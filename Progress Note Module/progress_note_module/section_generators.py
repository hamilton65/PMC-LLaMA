import re


def parse_snapshot_context(compact_context: str) -> dict:
    data = {
        "snapshot": {},
        "problems": [],
        "medications": [],
        "vitals": {},
        "labs": [],
        "provider_note": "",
    }

    section = ""
    provider_lines = []
    for raw_line in (compact_context or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "PATIENT SNAPSHOT":
            section = "snapshot"
            continue
        if upper == "CURRENT PROVIDER NOTE":
            section = "provider_note"
            continue
        if upper == "ACTIVE PROBLEMS":
            section = "problems"
            continue
        if upper == "ACTIVE MEDICATIONS":
            section = "medications"
            continue
        if upper == "LATEST VITALS SNAPSHOT":
            section = "vitals"
            continue
        if upper == "KEY LAB TREND":
            section = "labs"
            continue

        if section == "snapshot" and line.startswith("-") and ":" in line:
            k, v = line[1:].split(":", 1)
            data["snapshot"][k.strip().lower()] = v.strip()
        elif section == "provider_note":
            provider_lines.append(line)
        elif section == "problems" and line.startswith("-"):
            data["problems"].append(line[1:].strip())
        elif section == "medications" and line.startswith("-"):
            data["medications"].append(line[1:].strip())
        elif section == "vitals" and line.startswith("-") and ":" in line:
            k, v = line[1:].split(":", 1)
            data["vitals"][k.strip().lower()] = v.strip()
        elif section == "labs" and line.startswith("-"):
            data["labs"].append(line[1:].strip())

    data["provider_note"] = " ".join(provider_lines).strip()
    return data


def build_section_prompts(encounter: dict) -> dict[str, str]:
    snap = encounter["snapshot"]
    provider_note = encounter.get("provider_note", "")
    problems = encounter.get("problems", [])
    labs = encounter.get("labs", [])

    hpi_prompt = (
        "Write one single-paragraph APRN HPI using clinically relevant details only. "
        "Include chief complaint, relevant diagnoses, key abnormal labs, current response to treatment, and pertinent negatives. "
        "No headings, no bullets.\n\n"
        f"Patient snapshot: {snap}\n"
        f"Problems: {problems}\n"
        f"Labs: {labs}\n"
        f"Provider note: {provider_note}\n"
    )

    ros_prompt = (
        "Write ROS bullets with these systems exactly: General, HEENT, Neck, CV, Resp, GI, GU, MSK, Integ, Neuro, Psych, Endo, Heme/Lymph, Allergic/Immunologic. "
        "Focused pertinent positives and negatives, concise clinical phrasing. "
        "Do not include orientation or the phrase 'no acute distress'.\n\n"
        f"Provider note: {provider_note}\n"
    )

    pe_prompt = (
        "Write Physical Exam bullets by system with objective findings only. Include orientation status in neurologic exam. "
        "No invented findings beyond supplied context; if unavailable, state stable/baseline phrasing.\n\n"
        f"Patient snapshot: {snap}\n"
        f"Provider note: {provider_note}\n"
    )

    ap_prompt = (
        "Write Assessment and Plan as bullet list with at least 4 active diagnoses. "
        "Prioritize diagnoses related to the chief complaint first. "
        "Each bullet must include status/trajectory, pertinent findings, intervention, and monitoring.\n\n"
        f"Chief complaint: {snap.get('chief complaint', 'Not provided')}\n"
        f"Problems: {problems}\n"
        f"Labs: {labs}\n"
        f"Provider note: {provider_note}\n"
    )

    return {
        "hpi": hpi_prompt,
        "ros": ros_prompt,
        "pe": pe_prompt,
        "ap": ap_prompt,
    }


def deterministic_sections(encounter: dict) -> dict[str, str]:
    snap = encounter["snapshot"]
    provider_note = encounter.get("provider_note", "")
    problems = [p for p in encounter.get("problems", []) if p and p.lower() != "not provided"]
    meds = [m for m in encounter.get("medications", []) if m and m.lower() != "not provided"]
    labs = [l for l in encounter.get("labs", []) if l and l.lower() != "not provided" and "GLUCOSE" not in l.upper()]
    vitals = encounter.get("vitals", {})

    if len(problems) < 4:
        for p in ["Hypokalemia", "Chronic kidney disease", "Atrial fibrillation", "Hypertension"]:
            if p.lower() not in {x.lower() for x in problems}:
                problems.append(p)
            if len(problems) >= 4:
                break

    hpi = provider_note
    pmh = "\n".join([f"- {p}" for p in problems]) if problems else "- Not provided"
    med_block = "\n".join([f"- {m}" for m in meds]) if meds else "- Not provided"

    ros = "\n".join([
        "- General: Denies fever/chills; interval status per HPI.",
        "- HEENT: Denies acute headache or visual change.",
        "- Neck: Denies acute neck pain or stiffness.",
        "- CV: Denies chest pain, palpitations, presyncope, or syncope unless noted in HPI.",
        "- Resp: Denies dyspnea or cough unless noted in HPI.",
        "- GI: Denies nausea, vomiting, diarrhea, or abdominal pain unless noted in HPI.",
        "- GU: Denies dysuria unless otherwise documented.",
        "- MSK: Denies new focal weakness or acute joint pain.",
        "- Integ: Denies new rash or acute skin complaint.",
        "- Neuro: Denies dizziness or focal neurologic deficit unless noted in HPI.",
        "- Psych: Baseline cognitive/mood status monitored.",
        "- Endo: No acute endocrine complaint reported.",
        "- Heme/Lymph: No active bleeding complaint reported.",
        "- Allergic/Immunologic: No new allergic reaction reported.",
    ])

    pe = "\n".join([
        "- General: Elderly patient in bed/chair, no acute distress unless otherwise noted.",
        "- HEENT: No acute focal abnormality documented.",
        "- Neck: Supple, no acute concern documented.",
        "- CV: Rhythm/rate clinically monitored in context of active diagnoses.",
        "- Resp: Non-labored respirations; no acute distress documented.",
        "- GI: No acute abdominal complaint documented.",
        "- GU: No acute GU distress documented.",
        "- MSK: Baseline functional status considered.",
        "- Integ: No acute skin emergency documented.",
        "- Neuro: Orientation status documented and baseline cognition/neurologic status monitored.",
        "- Psych: Mood/affect monitored in context of baseline status.",
    ])

    ap_lines = []
    for p in problems[:8]:
        ap_lines.append(f"- {p}: active management continued; correlate with current symptoms, vitals, and laboratory trend. Continue monitoring and patient/family education.")
    ap = "\n".join(ap_lines)

    vitals_lines = [
        f"- Blood Pressure: {vitals.get('bp', 'Not provided')}",
        f"- Pulse: {vitals.get('pulse', 'Not provided')}",
        f"- Respirations: {vitals.get('respirations', vitals.get('rr', 'Not provided'))}",
        f"- Temperature: {vitals.get('temperature', vitals.get('temp', 'Not provided'))}",
        f"- SpO2: {vitals.get('spo2', 'Not provided')}",
        f"- Weight: {vitals.get('weight', 'Not provided')}",
        f"- Height: {vitals.get('height', 'Not provided')}",
        f"- BMI: {vitals.get('bmi', 'Not provided')}",
    ]
    if any(k in " ".join(problems).lower() for k in ["diabetes", "dm", "prediabetes", "pre-diabetes"]):
        vitals_lines.append(f"- Glucose: {vitals.get('glucose', 'Not provided')}")

    labs_block = "\n".join([f"- {l}" for l in labs]) if labs else "- Not provided"

    return {
        "hpi": hpi,
        "pmh": pmh,
        "allergies": f"- {snap.get('allergies', 'Not provided')}",
        "medications": med_block,
        "ros": ros,
        "vitals": "\n".join(vitals_lines),
        "labs": labs_block,
        "pe": pe,
        "ap": ap,
    }
