import asyncio
import os

import httpx

from progress_note_module.ingestion import optimize_ingestion
from progress_note_module.note_assembler import assemble_note
from progress_note_module.section_generators import (
    build_section_prompts,
    deterministic_sections,
    parse_snapshot_context,
)
from progress_note_module.template_contract import (
    compile_template_contract,
    validate_rendered_note,
)
from progress_note_module.template_db import (
    compose_template_body_from_sections,
    default_template_sections,
    get_default_template,
    get_template,
    init_db,
)


class ProgressNoteModule:
    def __init__(
        self,
        modal_raw_api_url: str | None = None,
        use_remote_section_generation: bool = True,
    ) -> None:
        init_db()
        self.modal_raw_api_url = modal_raw_api_url or os.getenv(
            "MODAL_RAW_API_URL",
            "https://hamilton65--generate-raw.modal.run",
        )
        self.use_remote_section_generation = use_remote_section_generation

    async def _generate_section_via_modal(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        resp = await client.post(self.modal_raw_api_url, json=payload)
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    @staticmethod
    def _repair_sections_with_deterministic_fallback(
        sections: dict[str, str],
        deterministic: dict[str, str],
    ) -> dict[str, str]:
        repaired = dict(sections)
        for key in ["hpi", "pmh", "allergies", "medications", "ros", "vitals", "labs", "pe", "ap"]:
            txt = (repaired.get(key) or "").strip()
            if len(txt) < 20:
                repaired[key] = deterministic.get(key, "Not provided")
        return repaired

    async def generate_note_async(
        self,
        patient_context: str,
        provider_notes: str,
        template_id: int | None = None,
        template_sections: dict[str, str] | None = None,
    ) -> str:
        template_name = "Custom Template"
        template_text = ""

        if template_id:
            record = get_template(template_id)
            if record is None:
                raise ValueError("Selected template not found")
            template_name = record.name
            template_text = record.body
        elif template_sections:
            merged = default_template_sections()
            for key, value in template_sections.items():
                if key in merged and isinstance(value, str):
                    merged[key] = value
            template_text = compose_template_body_from_sections(merged)
        else:
            default_tpl = get_default_template()
            template_name = default_tpl.name
            template_text = default_tpl.body

        optimized_context = optimize_ingestion(patient_context, provider_notes)
        encounter = parse_snapshot_context(optimized_context)
        contract = compile_template_contract(template_name, template_text)

        deterministic = deterministic_sections(encounter)
        sections = dict(deterministic)

        if self.use_remote_section_generation:
            prompts = build_section_prompts(encounter)
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    sections["hpi"] = await self._generate_section_via_modal(client, prompts["hpi"], 280, 0.1)
                    sections["ros"] = await self._generate_section_via_modal(client, prompts["ros"], 340, 0.1)
                    sections["pe"] = await self._generate_section_via_modal(client, prompts["pe"], 320, 0.1)
                    sections["ap"] = await self._generate_section_via_modal(client, prompts["ap"], 420, 0.1)
            except httpx.HTTPError:
                sections = dict(deterministic)

        sections = self._repair_sections_with_deterministic_fallback(sections, deterministic)
        note = assemble_note(encounter, sections, contract)

        missing = validate_rendered_note(note, contract)
        if missing:
            note = assemble_note(encounter, deterministic, contract)

        return note

    def generate_note(
        self,
        patient_context: str,
        provider_notes: str,
        template_id: int | None = None,
        template_sections: dict[str, str] | None = None,
    ) -> str:
        return asyncio.run(
            self.generate_note_async(
                patient_context=patient_context,
                provider_notes=provider_notes,
                template_id=template_id,
                template_sections=template_sections,
            )
        )


def generate_progress_note(
    patient_context: str,
    provider_notes: str,
    template_id: int | None = None,
    template_sections: dict[str, str] | None = None,
    use_remote_section_generation: bool = True,
) -> str:
    module = ProgressNoteModule(use_remote_section_generation=use_remote_section_generation)
    return module.generate_note(
        patient_context=patient_context,
        provider_notes=provider_notes,
        template_id=template_id,
        template_sections=template_sections,
    )
