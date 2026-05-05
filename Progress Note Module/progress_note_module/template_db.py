import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "templates.db"

DEFAULT_TEMPLATE_NAME = "Progress Note, Acute APRN"


def default_template_sections() -> dict[str, str]:
    current_user = os.getenv("USER", "Provider")
    return {
        "header": "Progress Note",
        "date_of_service": "Date of Service: [MM/DD/YYYY]",
        "patient_type": "Patient Type: Established patient",
        "visit_type": "Visit Type: Acute Visit",
        "demographics": "Auto-filled from patient data: Patient Name, DOB, Sex, Facility, Room, Code Status. Document DOS, Facility (1–4 words), Room number, Patient Name (2 words), DOB, Code Status (Full Code/DNR).",
        "chief_complaint": "Auto-extract Chief Complaint (CC:) from input. Document as search for 'CC:' in patient transcript.",
        "hpi": "Write in a single paragraph using complex medical terminology. Begin: '[Age] year old [if confused: \"confused\"] [race if known: use \"African American\" not \"Black\", \"Caucasian\" not \"White\"] [sex] with a medical history of [only those chronic or acute diagnoses directly relevant to the chief complaint, abnormal vital signs, abnormal labs, positive ROS, or abnormal PE findings] who resides at [facility, location] seen today [date] in his/her room lying in bed for [CC].' Do not use terms 'A+Ox1', 'A+Ox2-3', or 'A+Ox4'; interpret appropriately (e.g., A+Ox1=Alert with Confusion, A+Ox4=Alert and oriented, A+Ox2-3=Alert to person and place). Document provider-performed exam findings in first person. Explicitly state facility/location and patient positioning. Analyze and correlate labs/radiology with clinical status. Review chronic disease management only if related to CC or abnormal findings. Include only diagnoses directly relevant to current clinical context; omit all others.",
        "past_medical_history": "Bullet list. Include all chronic conditions from medical history and any recent acute conditions. Extract only those diagnoses relevant to CC or abnormal findings. Consolidate related diagnoses; use standardized, concise clinical terms. Remove non-essential descriptors. Omit chronic diagnoses unrelated to current encounter.",
        "allergies": "Auto-extract allergies from records as bullets.",
        "medications": "Bullet list with dose/frequency. Include only active medications; consolidate duplicates. Each medication on separate line.",
        "ros": "Structured review with each system as separate bullet: General, HEENT, Neck, CV, Resp, GI, GU, MSK, Integ, Neuro, Psych, Endo, Heme/Lymph, Allergic/Immunologic. Use focused, advanced-level language. Include positives and minimum pertinent negatives. Exclude incontinence unless positive. Do not combine systems. Never include orientation or 'no acute distress' in ROS.",
        "vital_signs": "Bullet list; omit all date stamps except for Weight. Include: BP, Pulse, RR, Temp, SpO2, Weight (with date if relevant), Height, BMI (if relevant), Glucose (only if diabetic—never label as 'most recent'). Document only most recent complete set. Omit note if any required vital sign is missing.",
        "laboratory_results": "Include only if reviewed/relevant per transcript. Bullet list by date. Exclude blood glucose. Omit section if no valid results. Only include clinically relevant labs.",
        "radiology_results": "Include relevant radiology by date if present; otherwise state 'Not provided.'",
        "physical_exam": "System-based bullet list. Each system as separate bullet; advanced-level objective findings only. Include baseline weakness. Only include findings on or before DOS. Include orientation status in Neuro section. Omit 'no acute distress' from PE.",
        "assessment_plan": "Bullet list. Ensure at least 4 active diagnoses. Prioritize chief-complaint-related diagnoses first, then add unrelated diagnoses. Each diagnosis on separate line unless clinically interrelated and managed identically. For 'acute on chronic' conditions, list only chronic unless acute exacerbation is documented. Omit irrelevant acute diagnoses. For each diagnosis: specify condition, trajectory, pertinent findings, interventions, monitoring. Append specialty service statements as appropriate. Include lifestyle, preventive, and education recommendations. Ensure all listed diagnoses have active management. Remove unsupported diagnoses.",
        "ending_phrase": "Recent laboratory/diagnostic results have been reviewed if present. Appropriate lab work and medication ordered as necessary. Risks and benefits of controlled substances use and side effects have been discussed if applicable. Continue current treatment as discussed with the patient, family, and nursing staff. Total time spent caring for the resident today was [30-35 minutes]. This includes time spent before the visit reviewing the chart, time spent during the visit face to face, time spent after the visit on documentation. Patient's code status is [Code Status]. (Note: Do NOT include 'Summary' or 'Signature' after Assessment and Plan.)",
        "provider_name": current_user,
        "dictation_disclaimer": "EMR Dragon/Transcription Disclaimer: Marked this encounter note is an electronic transcription/translation of spoken language to printed text. The electronic translation of spoken language may permit erroneous, or at times, nonsensical words or phrases to be inadvertently transcribed. Although I have reviewed the note for such errors, some may still exist.",
    }


def compose_template_body_from_sections(sections: dict[str, str]) -> str:
    # Composed instruction body for contract compiler / prompts.
    order = [
        "header",
        "date_of_service",
        "patient_type",
        "visit_type",
        "demographics",
        "chief_complaint",
        "hpi",
        "past_medical_history",
        "allergies",
        "medications",
        "ros",
        "vital_signs",
        "laboratory_results",
        "radiology_results",
        "physical_exam",
        "assessment_plan",
        "ending_phrase",
        "provider_name",
        "dictation_disclaimer",
    ]
    lines = ["Acute Progress Note Algorithm (Condensed, Advanced Version - HPI and Output Optimization) APRN", ""]
    label_map = {
        "header": "Header",
        "date_of_service": "Date of Service",
        "patient_type": "Patient Type",
        "visit_type": "Visit Type",
        "demographics": "Demographics",
        "chief_complaint": "Chief Complaint",
        "hpi": "History of Present Illness",
        "past_medical_history": "Past Medical History",
        "allergies": "Allergies",
        "medications": "Medications",
        "ros": "Review of Systems (ROS)",
        "vital_signs": "Vital Signs",
        "laboratory_results": "Laboratory Results",
        "radiology_results": "Radiology Results",
        "physical_exam": "Physical Exam",
        "assessment_plan": "Assessment and Plan",
        "ending_phrase": "Ending Phrase",
        "provider_name": "Provider Name",
        "dictation_disclaimer": "Dictation Disclaimer",
    }
    for key in order:
        val = (sections.get(key) or "").strip()
        if not val:
            continue
        lines.append(f"{label_map[key]}:")
        lines.append(val)
        lines.append("")
    return "\n".join(lines).strip()


@dataclass
class TemplateRecord:
    id: int
    name: str
    body: str
    is_default: bool
    sections: dict[str, str]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_sections(row: sqlite3.Row) -> dict[str, str]:
    base = default_template_sections()
    raw = row["sections_json"] if "sections_json" in row.keys() else None
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if k in base and isinstance(v, str):
                        base[k] = v
        except Exception:
            pass
    return base


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                body TEXT NOT NULL,
                is_default INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Migration for sectioned templates.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(templates)").fetchall()]
        if "sections_json" not in cols:
            conn.execute("ALTER TABLE templates ADD COLUMN sections_json TEXT")

        defaults = default_template_sections()
        default_body = compose_template_body_from_sections(defaults)

        row = conn.execute("SELECT id FROM templates WHERE name = ?", (DEFAULT_TEMPLATE_NAME,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO templates (name, body, is_default, sections_json) VALUES (?, ?, 1, ?)",
                (DEFAULT_TEMPLATE_NAME, default_body, json.dumps(defaults)),
            )
        else:
            conn.execute(
                "UPDATE templates SET is_default = CASE WHEN id = ? THEN 1 ELSE 0 END",
                (row["id"],),
            )
            existing = conn.execute("SELECT sections_json FROM templates WHERE id = ?", (row["id"],)).fetchone()
            if not existing["sections_json"]:
                conn.execute(
                    "UPDATE templates SET sections_json = ?, body = ? WHERE id = ?",
                    (json.dumps(defaults), default_body, row["id"]),
                )


def list_templates() -> list[TemplateRecord]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, body, is_default, sections_json FROM templates ORDER BY is_default DESC, id ASC"
        ).fetchall()
        out = []
        for r in rows:
            sections = _load_sections(r)
            body = compose_template_body_from_sections(sections)
            out.append(TemplateRecord(id=r["id"], name=r["name"], body=body, is_default=bool(r["is_default"]), sections=sections))
        return out


def get_template(template_id: int) -> TemplateRecord | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, body, is_default, sections_json FROM templates WHERE id = ?",
            (template_id,),
        ).fetchone()
        if row is None:
            return None
        sections = _load_sections(row)
        body = compose_template_body_from_sections(sections)
        return TemplateRecord(id=row["id"], name=row["name"], body=body, is_default=bool(row["is_default"]), sections=sections)


def create_template(name: str, body: str | None = None, sections: dict[str, str] | None = None) -> TemplateRecord:
    with _connect() as conn:
        sec = default_template_sections()
        if sections:
            for k, v in sections.items():
                if k in sec and isinstance(v, str):
                    sec[k] = v
        if body and not sections:
            sec["hpi"] = body.strip()[:2000]
        composed = compose_template_body_from_sections(sec)

        cur = conn.execute(
            "INSERT INTO templates (name, body, is_default, sections_json) VALUES (?, ?, 0, ?)",
            (name.strip(), composed, json.dumps(sec)),
        )
        tid = cur.lastrowid
        return get_template(tid)


def update_template(template_id: int, name: str, body: str | None = None, sections: dict[str, str] | None = None) -> TemplateRecord | None:
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id, is_default, sections_json FROM templates WHERE id = ?",
            (template_id,),
        ).fetchone()
        if existing is None:
            return None

        sec = default_template_sections()
        if existing["sections_json"]:
            try:
                parsed = json.loads(existing["sections_json"])
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if k in sec and isinstance(v, str):
                            sec[k] = v
            except Exception:
                pass

        if sections:
            for k, v in sections.items():
                if k in sec and isinstance(v, str):
                    sec[k] = v
        if body and not sections:
            sec["hpi"] = body.strip()[:2000]

        composed = compose_template_body_from_sections(sec)
        conn.execute(
            "UPDATE templates SET name = ?, body = ?, sections_json = ? WHERE id = ?",
            (name.strip(), composed, json.dumps(sec), template_id),
        )
    return get_template(template_id)


def delete_template(template_id: int) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT is_default FROM templates WHERE id = ?", (template_id,)).fetchone()
        if row is None:
            return False
        if int(row["is_default"]) == 1:
            raise ValueError("Default template cannot be deleted")
        conn.execute("DELETE FROM templates WHERE id = ?", (template_id,))
        return True


def get_default_template() -> TemplateRecord:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, body, is_default, sections_json FROM templates WHERE is_default = 1 LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Default template not found")
        sections = _load_sections(row)
        body = compose_template_body_from_sections(sections)
        return TemplateRecord(id=row["id"], name=row["name"], body=body, is_default=bool(row["is_default"]), sections=sections)
