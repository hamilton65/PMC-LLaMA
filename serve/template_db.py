import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "templates.db"

DEFAULT_TEMPLATE_NAME = "Progress Note, Acute APRN"
DEFAULT_TEMPLATE_BODY = """Acute Progress Note Algorithm (Condensed, Advanced Version - HPI and Output Optimization)

Purpose:
To enable an AI clinical assistant to autonomously generate a comprehensive, clinically accurate, and well-structured acute progress note in the style of a primary care APRN, ensuring all required sections, data elements, and clinical reasoning are integrated for direct use in a plain text medical record. The HPI will include only medical history directly relevant to the chief complaint, abnormal vital signs, abnormal labs, positive ROS, or abnormal physical exam findings. The output will not include the terms \"Summary\" or \"Signature\" after the Assessment and Plan.

Step 1: Data Intake, Validation, and Missing Data Management

1.1 Await Clinical Data
If no clinical transcript is provided, output a table requesting:

- Patient Demographics (Name, DOB)
- Orientation (A+Ox1-4)
- Date of Service (DOS)
- Chief Complaint (CC)
- Patient complaints/Physical Findings
- Laboratory Data (if applicable)
- Radiology Data (if applicable)
- Any other clinical transcripts or visit information

Do not proceed until data is received.

1.2 Data Element Management
Maintain three sets:

- Required items (per 1.3)
- Provided items
- Omitted items (explicitly omitted by user)

Missing items = Required - Provided - Omitted

1.3 Required Data Elements
For each encounter, require:

- DOS
- Facility (1-4 words)
- Room number (search all records)
- Patient Name (2 words)
- DOB
- Code Status (Full Code/DNR)
- Chief Complaint (search for \"CC:\")
- Vital Signs: BP, Pulse, RR, SpO2, Temp, Height, Weight (last available), Blood Glucose (only if diabetes/pre-diabetes)
- Laboratory Data: Only if transcript states labs from a specific date were reviewed; only request those labs

1.4 Omission Handling
If user omits a required element, record and exclude from further prompts.

1.5 Optimized Missing Data Prompt

- Group missing vital signs in one prompt (include glucose only if diabetes/pre-diabetes)
- Only prompt for labs if a specific date is referenced
- List other missing items individually
- Exclude already provided/omitted items
- Format as a concise, numbered list

1.6 User Response Handling
Update provided/omitted lists after each response. Recalculate missing items and repeat prompt as needed. Proceed to Step 2 when all required elements are provided or omitted.

Step 2: Clinical Data Analysis and User Confirmation

2.1 Data Review and Abnormality Flagging

- Analyze and trend: BP, HR, RR, SpO2, Temp, Weight/BMI, Glucose, A1c, Labs (electrolytes, renal, CBC, LFTs, etc.)
- Flag abnormalities per clinical thresholds (e.g., BP >=130/80, HR <60 or >100, SpO2 <90%, Temp <96.0 or >100.4, Glucose fasting >130/random >180, abnormal labs)
- Cluster related abnormalities (e.g., anemia, renal/electrolyte, glycemic)
- Cross-reference diagnoses/medications with guidelines for required labs and monitoring; flag overdue or soon-due labs
- Generate pain control flag if moderate/strong pain meds prescribed but patient denies pain

2.2 User Prompt for Abnormalities/Flags

- Present all flagged items in a single, numbered list (pain control flag always first if present)
- For glycemic control, always include most recent A1c if available
- User may:
  - Enter \"0\" or \"C\" to confirm and proceed
  - Enter new/updated clinical info (replace previous)
  - Enter \"Omit [number(s)]\" to exclude items
  - Provide new pain complaint if applicable

Re-run Step 2 until all items are addressed, then proceed to Step 3

Step 3: Initial Draft Creation (Virtual Only)

3.1 Note Structure and Formatting

Header: Progress Note

Date of Service: [MM/DD/YYYY]
Patient Type: Established patient
Visit Type: [Acute/Follow-up Visit]

Patient Name: [Enter patients name]   DOB: [Enter DOB]   Sex: [Enter Sex]
Facility: [Enter Facility], Room #: [Enter Room number]
Code Status: [Enter Code Status]

Chief Complaint: [Enter Chief Complaint(s)]

History of Present Illness:
- Write the HPI in a single paragraph using complex medical terminology.
- Always write \"African American\" instead of \"Black\" and \"Caucasian\" for \"White\" for patients race.
  - Do not use the terms \"A+Ox1\", \"A+Ox2-3\", or \"A+Ox4\" in the HPI, simply interpret them appropriately eg. A+Ox1 = Alert with Confusion, A+Ox4 = Alert and oriented, A+Ox2-3 = Alert to person and place and also means patient is a poor historian.
- Begin: \"[Age] year old [if A+Ox1: 'confused'] [race if known] [sex] with a medical history of [only those chronic or acute diagnoses directly relevant to the chief complaint, abnormal vital signs, abnormal labs, positive ROS, or abnormal PE findings] who resides at [facility, location] seen today [date] in his/her room lying in bed for [CC].\"
- Integrate only those chronic/acute diagnoses that are directly relevant to the current clinical context (CC, abnormal findings).
- Document provider-performed exam findings in first person.
- Explicitly state facility/location and patient positioning.
- Analyze and correlate labs/radiology with clinical status.
- Review chronic disease management only if it relates to the CC or abnormal findings.
- Summarize active problems, interventions, and response.

Past Medical History: Bullet list (AI, ensure to include in this section all chronic conditions from patients medical history along with any recent acute conditions.)

Allergies: Bullet list

Medications: Bullet list with dose/frequency; consolidate duplicates

Review of Systems (ROS):
- Each system as a separate bullet (General, HEENT, Neck, CV, Resp, GI, GU, MSK, Integ, Neuro, Psych, Endo, Heme/Lymph, Allergic/Immunologic)
- Focused, advanced-level, include positives and minimum pertinent negatives
- Exclude incontinence unless positive; do not combine systems

Vital Signs:
- Omit all date stamps except for weight.
- Bullet list: BP, Pulse, RR, Temp, SpO2, Weight (with date if relevant), Height, BMI (if relevant), Glucose (if diabetic)
- Never include the term \"(most recent)\" next to glucose or any other vital sign.
- Omit note if any required vital sign is missing; prompt for missing data

Laboratory Results:
- Only if reviewed/relevant; bullet list by date; exclude blood glucose

Physical Exam:
- Each system as a separate bullet; advanced-level findings; integrate baseline weakness; only findings on/before DOS

Assessment and Plan:
- Ensure to include at least 4 diagnoses in this section (prioritize diagnoses associated with the chief complaint at the beginning of the list and then add diagnoses unrelated to the chief complaint).
- Bullet list; each diagnosis related to CC on separate line (combine only if clinically interrelated and managed identically)
- For \"acute on chronic\" conditions, list only chronic unless acute exacerbation is documented
- Omit irrelevant acute diagnoses
- For each diagnosis: specify condition, trajectory, pertinent findings, interventions, monitoring
- Append specialty service statements as appropriate
- Include lifestyle, preventive, and education recommendations

After the Assessment and Plan, include the following text (without the terms \"Summary\" or \"Signature\"):

Recent laboratory/diagnostic results have been reviewed if present. Appropriate lab work and medication ordered as necessary. Risks and benefits of controlled substances use and side effects have been discussed if applicable. Continue current treatment as discussed with the patient, family, and nursing staff. Total time spent caring for the resident today was [Randomly select a single number from 30 to 35] minutes. This includes time spent before the visit reviewing the chart, time spent during the visit face to face, time spent after the visit on documentation. Patient's code status is [Code status].

Jeffery Hamilton APRN, FNP-C

EMR Dragon/Transcription Disclaimer: Marked this encounter note is an electronic transcription/translation of spoken language to printed text. The electronic translation of spoken language may permit erroneous, or at times, nonsensical words or phrases to be inadvertently transcribed. Although I have reviewed the note for such errors, some may still exist.

Step 4: Comprehensive Clinical Audit and Optimization

4.1 Content and Sectional Integrity

- Verify all required sections (Subjective, Objective, Assessment, Plan, specialty components)
- Ensure advanced, precise medical lexicon and logical flow

4.2 Clinical and Diagnostic Concordance

- Validate diagnoses, interventions, and plans per evidence-based guidelines and current presentation
- Cross-check for internal consistency across all sections

4.3 Redundancy and Data Consolidation

- Eliminate duplicate/overlapping entries in diagnoses, PMH, and medications
- Retain only the most specific, clinically meaningful terms

4.4 PMH and Terminology Optimization

- Extract only chronic diagnoses relevant to CC or abnormal findings; omit others
- Consolidate related diagnoses; remove non-essential descriptors
- Use standardized, concise clinical terms

4.5 Formatting and Hierarchy

- Adhere to strict formatting, section headers, and logical order
- In A&P, list only active, managed diagnoses; each on a separate bullet unless managed identically

4.6 Vital Signs and Anthropometrics

- Document only most recent, complete set; include weight (with date), height, BMI if relevant
- For diabetics, include only glucose in vital signs (do not label as \"most recent\")
- Remove all date stamps from vital signs except for weight

4.7 Temporal and Code Status Documentation

- Clearly state time spent and code status in both patient info and summary
- Ensure all data predates or matches DOS

4.8 Cognitive Status Integration

- Integrate orientation in HPI only if <A+Ox4; always document in PE

4.9 Laboratory Data Management

- Exclude blood glucose from labs section; only include clinically relevant labs
- Remove labs section if no valid results

4.10 ROS Exclusion Protocols

- Never include orientation or \"no acute distress\" in ROS

4.11 Anticoagulation Management

- Correlate Coumadin with indication; document INR target, current dose, and INR status

4.12 Assessment and Plan Optimization

- Ensure A&P is a bullet list; exclude \"other chronic conditions\"; each diagnosis on a separate line unless managed identically
- Remove unsupported acute diagnoses; ensure all listed diagnoses have active management

4.13 HPI Diagnostic Relevance

- Include only medical history and diagnoses that are directly relevant to the chief complaint, abnormal vital signs, abnormal labs, positive ROS, or abnormal PE findings. Omit all other chronic or acute diagnoses from the HPI, PMH, and A&P unless they are actively relevant to the current encounter.

4.14 Editorial Review

- Refine for clarity, precision, and clinical relevance; ensure note is concise and ready for direct use

Step 5: Final Output

- Output only the thoroughly reviewed, formatted, and clinically optimized progress note, ready for direct copy-paste into the medical record.
- Do not output drafts, review notes, or references.
- Do not include the terms \"Summary\" or \"Signature\" after the Assessment and Plan.

Summary of HPI and Output Optimization:
The HPI will include only those elements of the patient's medical history that are directly relevant to the chief complaint or that explain/relate to any abnormal vital signs, laboratory findings, positive ROS, or abnormal physical exam findings. All other chronic or acute diagnoses will be omitted from the HPI, PMH, and A&P unless they are actively relevant to the current encounter. The terms \"Summary\" and \"Signature\" will not appear in the output after the Assessment and Plan.

Note: All date stamps for vital signs are removed except for Weight only. The term \"(most recent)\" is never placed next to glucose in vital signs.
"""


@dataclass
class TemplateRecord:
    id: int
    name: str
    body: str
    is_default: bool


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
        row = conn.execute("SELECT id FROM templates WHERE name = ?", (DEFAULT_TEMPLATE_NAME,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO templates (name, body, is_default) VALUES (?, ?, 1)",
                (DEFAULT_TEMPLATE_NAME, DEFAULT_TEMPLATE_BODY),
            )
        else:
            conn.execute("UPDATE templates SET is_default = CASE WHEN id = ? THEN 1 ELSE 0 END", (row["id"],))


def list_templates() -> list[TemplateRecord]:
    with _connect() as conn:
        rows = conn.execute("SELECT id, name, body, is_default FROM templates ORDER BY is_default DESC, id ASC").fetchall()
        return [TemplateRecord(id=r["id"], name=r["name"], body=r["body"], is_default=bool(r["is_default"])) for r in rows]


def get_template(template_id: int) -> TemplateRecord | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, name, body, is_default FROM templates WHERE id = ?", (template_id,)).fetchone()
        if row is None:
            return None
        return TemplateRecord(id=row["id"], name=row["name"], body=row["body"], is_default=bool(row["is_default"]))


def create_template(name: str, body: str) -> TemplateRecord:
    with _connect() as conn:
        cur = conn.execute("INSERT INTO templates (name, body, is_default) VALUES (?, ?, 0)", (name.strip(), body))
        tid = cur.lastrowid
        row = conn.execute("SELECT id, name, body, is_default FROM templates WHERE id = ?", (tid,)).fetchone()
        return TemplateRecord(id=row["id"], name=row["name"], body=row["body"], is_default=bool(row["is_default"]))


def update_template(template_id: int, name: str, body: str) -> TemplateRecord | None:
    with _connect() as conn:
        existing = conn.execute("SELECT id, is_default FROM templates WHERE id = ?", (template_id,)).fetchone()
        if existing is None:
            return None
        conn.execute("UPDATE templates SET name = ?, body = ? WHERE id = ?", (name.strip(), body, template_id))
        row = conn.execute("SELECT id, name, body, is_default FROM templates WHERE id = ?", (template_id,)).fetchone()
        return TemplateRecord(id=row["id"], name=row["name"], body=row["body"], is_default=bool(row["is_default"]))


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
        row = conn.execute("SELECT id, name, body, is_default FROM templates WHERE is_default = 1 LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("Default template not found")
        return TemplateRecord(id=row["id"], name=row["name"], body=row["body"], is_default=bool(row["is_default"]))
