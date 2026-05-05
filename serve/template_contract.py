from dataclasses import dataclass


@dataclass
class TemplateContract:
    name: str
    strict_aprn: bool
    required_sections: list[str]
    ros_systems: list[str]
    required_vitals: list[str]
    min_ap_diagnoses: int
    require_closing_block: bool


def compile_template_contract(template_name: str, template_body: str) -> TemplateContract:
    body = (template_body or "").lower()
    name = (template_name or "Custom Template").strip()

    strict_aprn = "acute progress note algorithm" in body and "aprn" in body

    required_sections = [
        "Progress Note",
        "Date of Service:",
        "Patient Type:",
        "Visit Type:",
        "Chief Complaint:",
        "History of Present Illness:",
        "Past Medical History:",
        "Allergies:",
        "Medications:",
        "Review of Systems (ROS):",
        "Vital Signs:",
        "Physical Exam:",
        "Assessment and Plan:",
    ]
    if "laboratory results" in body or "laboratory data" in body:
        required_sections.append("Laboratory Results:")

    ros_systems = [
        "General",
        "HEENT",
        "Neck",
        "CV",
        "Resp",
        "GI",
        "GU",
        "MSK",
        "Integ",
        "Neuro",
        "Psych",
        "Endo",
        "Heme/Lymph",
        "Allergic/Immunologic",
    ]

    required_vitals = ["Blood Pressure", "Pulse", "Respirations", "Temperature", "SpO2", "Weight", "Height"]

    min_ap_diagnoses = 4 if "at least 4 diagnoses" in body or strict_aprn else 2
    require_closing_block = strict_aprn

    return TemplateContract(
        name=name,
        strict_aprn=strict_aprn,
        required_sections=required_sections,
        ros_systems=ros_systems,
        required_vitals=required_vitals,
        min_ap_diagnoses=min_ap_diagnoses,
        require_closing_block=require_closing_block,
    )


def validate_rendered_note(note_text: str, contract: TemplateContract) -> list[str]:
    missing = []
    text = note_text or ""
    low = text.lower()

    for section in contract.required_sections:
        if section.lower() not in low:
            missing.append(f"missing section: {section}")

    for sys_name in contract.ros_systems:
        if f"- {sys_name.lower()}" not in low:
            missing.append(f"missing ROS system: {sys_name}")

    for v in contract.required_vitals:
        if f"- {v.lower()}" not in low:
            missing.append(f"missing vital: {v}")

    ap_block = ""
    if "Assessment and Plan:" in text:
        ap_block = text.split("Assessment and Plan:", 1)[1]
    ap_lines = [ln for ln in ap_block.splitlines() if ln.strip().startswith("-")]
    if len(ap_lines) < contract.min_ap_diagnoses:
        missing.append(f"assessment/plan diagnoses fewer than {contract.min_ap_diagnoses}")

    if contract.require_closing_block:
        if "Jeffery Hamilton APRN, FNP-C" not in text:
            missing.append("missing required provider signature line")
        if "EMR Dragon/Transcription Disclaimer" not in text:
            missing.append("missing required EMR disclaimer")

    return missing
