import re
from collections import defaultdict
from datetime import datetime

KEY_LABS = {
    "POTASSIUM",
    "SODIUM",
    "CREATININE",
    "BUN (UREA NITROGEN)",
    "CARBON DIOXIDE (CO2)",
    "GLUCOSE",
    "GFR-NON-AFRICAN AMERICAN",
    "GFR-AFRICAN AMERICAN",
    "ALBUMIN",
    "PROTEIN, TOTAL",
    "HEMOGLOBIN",
}


def _extract(pattern: str, text: str, default: str = "Not provided") -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else default


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _parse_date(date_text: str) -> datetime | None:
    for fmt in ("%m/%d/%Y", "%m/%d/%Y %I:%M %p", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(date_text.strip(), fmt)
        except ValueError:
            continue
    return None


def _extract_active_problems(raw: str, limit: int = 12) -> list[str]:
    problems = []
    # Matches lines such as: ICD-10 <problem> <code> <date> Active
    pattern = re.compile(r"ICD-10\s+(.+?)\s+[A-Z]\d[\w\.]+\s+\d{2}/\d{2}/\d{4}\s+Active", re.IGNORECASE)
    for m in pattern.finditer(raw):
        p = _normalize_ws(m.group(1))
        if p and p not in problems:
            problems.append(p)
        if len(problems) >= limit:
            break
    return problems


def _extract_active_meds(raw: str, limit: int = 12) -> list[str]:
    meds = []
    seen = set()
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if "Active" not in s:
            continue
        if "Not Active" in s:
            continue
        if not any(k in s.lower() for k in ("tablet", "capsule", "suppository", "solution", "mg", "mcg", "mEq", "oral", "rectal".lower())):
            continue
        # Compact table-like medication rows.
        s = re.sub(r"\s{2,}", " | ", s)
        s = s.lstrip("- ").strip()
        key = s.split("|", 1)[0].strip().lower()
        if key not in seen:
            seen.add(key)
            meds.append(s)
        if len(meds) >= limit:
            break
    return meds


def _extract_vitals_snapshot(raw: str) -> dict[str, str]:
    # Pull best-effort values from any line; avoid date-like false positives for BP.
    bp = "Not provided"
    for m in re.finditer(r"\b(\d{2,3})/(\d{2,3})\b", raw):
        sys = int(m.group(1))
        dia = int(m.group(2))
        if 60 <= sys <= 240 and 30 <= dia <= 140:
            bp = f"{sys}/{dia}"
            break
    pulse = _extract(r"\b(?:Pulse|HR)\b[^\n\d]*([0-9]{2,3})", raw)
    rr = _extract(r"\b(?:Respirations|RR)\b[^\n\d]*([0-9]{1,2})", raw)
    temp = _extract(r"\b([0-9]{2,3}\.?[0-9]?\s*F)\b", raw)
    spo2 = _extract(r"\b(?:SpO2|O2 Saturation)\b[^\n\d]*([0-9]{2,3}(?:\.\d)?)", raw)
    height = _extract(r"\bHeight\s*:\s*([^\n]+)", raw)
    weight = _extract(r"\bWeight\s*:\s*([^\n]+)", raw)
    bmi = _extract(r"\bBMI\s*:\s*([^\n]+)", raw)
    glucose = _extract(r"\bGLUCOSE\s+\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\s*[AP]M\s+([0-9]+(?:\.[0-9]+)?)", raw)

    return {
        "bp": bp,
        "pulse": pulse,
        "rr": rr,
        "temp": temp,
        "spo2": spo2,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "glucose": glucose,
    }


def _extract_labs(raw: str, limit: int = 16) -> list[str]:
    labs = []
    # Name Date Time Value Unit
    pat = re.compile(
        r"(?m)^([A-Z][A-Z0-9\-\(\)/\s']{2,})\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}\s*[AP]M)\s+.*?([\-]?[0-9]+(?:\.[0-9]+)?)\s*([A-Za-z/%0-9\.\-\^]+)?"
    )
    grouped: dict[str, list[tuple[datetime | None, str]]] = defaultdict(list)

    for m in pat.finditer(raw):
        name = _normalize_ws(m.group(1))
        if name not in KEY_LABS:
            continue
        date_txt = m.group(2)
        time_txt = m.group(3)
        dt = _parse_date(f"{date_txt} {time_txt}")
        value = m.group(4)
        unit = (m.group(5) or "").strip()
        grouped[name].append((dt, f"{date_txt}: {name} {value} {unit}".strip()))

    for analyte, rows in grouped.items():
        rows.sort(key=lambda x: x[0] or datetime.min, reverse=True)
        for _, row in rows[:2]:
            labs.append(row)
            if len(labs) >= limit:
                return labs

    return labs


def optimize_ingestion(patient_context: str, provider_notes: str) -> str:
    raw = (patient_context or "").strip()
    provider = (provider_notes or "").strip()

    name = _extract(r"\bPatient\s*Name\s*[:\t]\s*([^\n]+)", raw)
    if name == "Not provided":
        name = _extract(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", raw)
    dob = _extract(r"\bD\.?O\.?B\.?\s*[:\t]\s*([0-9/\-]+)", raw)
    sex = _extract(r"\bSex\s*[:\t]\s*([^\n]+)", raw)
    facility = _extract(r"\bFacility\s*[:\t]\s*([^\n]+)", raw)
    room = _extract(r"\b(?:Room\s*(?:#|number|no\.)?)\s*[:\t]\s*([^\n]+)", raw)
    if room == "Not provided":
        room = _extract(r"\b([A-Za-z]+\s*-\s*[^\n]*/\d+/[A-Z])\b", raw)
    code_status = _extract(r"\bCode\s*Status\s*[:\t]\s*([^\n]+)", raw)
    if code_status == "Not provided":
        if re.search(r"\bDNR\b", raw, flags=re.IGNORECASE):
            code_status = "DNR"
        elif re.search(r"\bFull\s*Code\b", raw, flags=re.IGNORECASE):
            code_status = "Full Code"
    allergies = _extract(r"\bAllerg(?:y|ies)\s*[:\t]\s*([^\n]+)", raw)

    dos = _extract(r"\bDOS\s*:\s*([^\.\n]+)", provider)
    cc = _extract(r"\bCC\s*:\s*([^\.\n]+)", provider)

    vitals = _extract_vitals_snapshot(raw)
    problems = _extract_active_problems(raw)
    meds = _extract_active_meds(raw)
    labs = _extract_labs(raw)

    out = []
    out.append("PATIENT SNAPSHOT")
    out.append(f"- Date of Service: {dos}")
    out.append(f"- Chief Complaint: {cc}")
    out.append(f"- Name: {name}")
    out.append(f"- DOB: {dob}")
    out.append(f"- Sex: {sex}")
    out.append(f"- Facility: {facility}")
    out.append(f"- Room: {room}")
    out.append(f"- Code Status: {code_status}")
    out.append(f"- Allergies: {allergies}")

    out.append("\nCURRENT PROVIDER NOTE")
    out.append(provider[:3500])

    out.append("\nACTIVE PROBLEMS")
    if problems:
        out.extend([f"- {p}" for p in problems])
    else:
        out.append("- Not provided")

    out.append("\nACTIVE MEDICATIONS")
    if meds:
        out.extend([f"- {m}" for m in meds])
    else:
        out.append("- Not provided")

    out.append("\nLATEST VITALS SNAPSHOT")
    out.append(f"- BP: {vitals['bp']}")
    out.append(f"- Pulse: {vitals['pulse']}")
    out.append(f"- Respirations: {vitals['rr']}")
    out.append(f"- Temperature: {vitals['temp']}")
    out.append(f"- SpO2: {vitals['spo2']}")
    out.append(f"- Weight: {vitals['weight']}")
    out.append(f"- Height: {vitals['height']}")
    out.append(f"- BMI: {vitals['bmi']}")
    out.append(f"- Glucose: {vitals['glucose']}")

    out.append("\nKEY LAB TREND")
    if labs:
        out.extend([f"- {l}" for l in labs])
    else:
        out.append("- Not provided")

    return "\n".join(out)
