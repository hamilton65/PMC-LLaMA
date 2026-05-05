# Progress Note Module

Portable module packaging for the PMC-LLaMA progress note pipeline.

## What To Copy

Copy this entire folder into your target repository as:

`Progress Note Module/`

The Python package is inside:

`Progress Note Module/progress_note_module/`

## Install

From inside `Progress Note Module/`:

```bash
pip install -r requirements.txt
```

## Use As A Python Module

```python
from progress_note_module import generate_progress_note, ProgressNoteModule

note = generate_progress_note(
    patient_context="58yo male...",
    provider_notes="Routine follow-up...",
)
print(note)

module = ProgressNoteModule(use_remote_section_generation=False)
note2 = module.generate_note(
    patient_context="...",
    provider_notes="...",
    template_id=1,
)
```

## Run Local UI

From inside `Progress Note Module/`:

```bash
bash run_ui.sh
```

Then open:

`http://localhost:8080`

## Optional Environment Variables

- `MODAL_RAW_API_URL` for section-wise model generation endpoint.

If this variable is not set, the module defaults to:

`https://hamilton65--generate-raw.modal.run`

## Notes

- The module writes `templates.db` next to `template_db.py`.
- Template sections remain editable and are assembled into one full progress note string in output.
