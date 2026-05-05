import random

from serve.template_contract import TemplateContract


def assemble_note(encounter: dict, sections: dict, contract: TemplateContract) -> str:
    snap = encounter.get("snapshot", {})

    dos = snap.get("date of service", "Not provided")
    cc = snap.get("chief complaint", "Not provided")
    name = snap.get("name", "Not provided")
    dob = snap.get("dob", "Not provided")
    sex = snap.get("sex", "Not provided")
    facility = snap.get("facility", "Not provided")
    room = snap.get("room", "Not provided")
    code_status = snap.get("code status", "Not provided")

    if name != "Not provided":
        parts = [p for p in name.split() if p]
        if len(parts) >= 2:
            name = f"{parts[0]} {parts[1]}"

    if code_status.lower() not in {"dnr", "full code"}:
        code_status = "DNR" if "dnr" in code_status.lower() else ("Full Code" if "full" in code_status.lower() else "Not provided")

    lines = [
        "Progress Note",
        "",
        f"Date of Service: {dos}",
        "Patient Type: Established patient",
        "Visit Type: Acute Visit",
        "",
        f"Patient Name: {name}   DOB: {dob}   Sex: {sex}",
        f"Facility: {facility}, Room #: {room}",
        f"Code Status: {code_status}",
        "",
        f"Chief Complaint: {cc}",
        "",
        "History of Present Illness:",
        sections.get("hpi", "Not provided").strip(),
        "",
        "Past Medical History:",
        "",
        sections.get("pmh", "- Not provided").strip(),
        "",
        "Allergies:",
        "",
        sections.get("allergies", "- Not provided").strip(),
        "",
        "Medications:",
        "",
        sections.get("medications", "- Not provided").strip(),
        "",
        "Review of Systems (ROS):",
        "",
        sections.get("ros", "- Not provided").strip(),
        "",
        "Vital Signs:",
        "",
        sections.get("vitals", "- Not provided").strip(),
    ]

    labs_block = sections.get("labs", "- Not provided").strip()
    if labs_block and labs_block != "- Not provided":
        lines.extend([
            "",
            "Laboratory Results:",
            "",
            labs_block,
        ])

    lines.extend([
        "",
        "Physical Exam:",
        "",
        sections.get("pe", "- Not provided").strip(),
        "",
        "Assessment and Plan:",
        "",
        sections.get("ap", "- Not provided").strip(),
    ])

    if contract.require_closing_block:
        minutes = random.randint(30, 35)
        closing = (
            "Recent laboratory/diagnostic results have been reviewed if present. Appropriate lab work and medication ordered as necessary. "
            "Risks and benefits of controlled substances use and side effects have been discussed if applicable. Continue current treatment as discussed with the patient, family, and nursing staff. "
            f"Total time spent caring for the resident today was {minutes} minutes. This includes time spent before the visit reviewing the chart, time spent during the visit face to face, time spent after the visit on documentation. "
            f"Patient's code status is {code_status}.\n\n"
            "Jeffery Hamilton APRN, FNP-C\n\n"
            "EMR Dragon/Transcription Disclaimer: Marked this encounter note is an electronic transcription/translation of spoken language to printed text. "
            "The electronic translation of spoken language may permit erroneous, or at times, nonsensical words or phrases to be inadvertently transcribed. "
            "Although I have reviewed the note for such errors, some may still exist."
        )
        lines.extend(["", closing])

    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()
