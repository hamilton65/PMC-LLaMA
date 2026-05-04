import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from serve.ingestion import optimize_ingestion
from serve.template_db import (
  create_template,
  delete_template,
  get_default_template,
  get_template,
  init_db,
  list_templates,
  update_template,
)

app = FastAPI(title="PMC-LLaMA Test UI")

# Default to your deployed Modal endpoint; override with MODAL_API_URL if needed.
MODAL_API_URL = os.getenv("MODAL_API_URL", "https://hamilton65--generate-note-v2.modal.run")

# Initialize template storage for environments where startup hooks are skipped.
init_db()


@app.on_event("startup")
def startup_event() -> None:
    init_db()


class TemplateCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)


class TemplateUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    patient_context: str
    provider_notes: str
    template_id: int | None = None
    template: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.3


class IngestionPreviewRequest(BaseModel):
    patient_context: str
    provider_notes: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PMC-LLaMA API Test UI</title>
  <style>
    :root {
      --bg: #f5f8f4;
      --panel: #ffffff;
      --ink: #1f2d1f;
      --muted: #5a6a5a;
      --line: #d7e3d4;
      --accent: #176b52;
      --accent-2: #0f4f3d;
      --warn: #ab3c2f;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 15% -10%, #d9ebd7 0%, transparent 55%),
        radial-gradient(900px 500px at 90% 0%, #e2efe6 0%, transparent 50%),
        var(--bg);
      min-height: 100vh;
      padding: 28px 16px;
    }

    .wrap {
      max-width: 1080px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 16px;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 6px 30px rgba(23, 50, 23, 0.06);
      padding: 16px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0.2px;
    }

    p.sub {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 14px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin: 12px 0 6px;
      color: #274327;
    }

    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      background: #fcfdfc;
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      color: inherit;
    }

    textarea { min-height: 110px; resize: vertical; }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .actions {
      margin-top: 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    button {
      border: none;
      border-radius: 10px;
      padding: 10px 16px;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      transition: transform 0.12s ease, filter 0.12s ease;
    }

    button:hover { transform: translateY(-1px); filter: brightness(1.03); }
    button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

    .status {
      font-size: 13px;
      color: var(--muted);
      min-height: 18px;
    }

    .status.error { color: var(--warn); }

    pre {
      margin: 0;
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #fbfdfb;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      min-height: 280px;
      font-size: 13px;
      line-height: 1.5;
    }

    .meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    @media (max-width: 920px) {
      .wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>PMC-LLaMA Note Generator</h1>
      <p class="sub">Local test UI that proxies to your deployed Modal endpoint.</p>

      <label for="patient_context">Patient Records (labs, vitals, meds)</label>
      <textarea id="patient_context">58yo male\nVitals: BP 142/88, HR 78, Temp 98.6F, SpO2 97%\nLabs: HbA1c 7.8, Creatinine 1.1\nMedications: Metformin 1000mg BID, Lisinopril 10mg daily</textarea>

      <label for="provider_notes">Provider Notes</label>
      <textarea id="provider_notes">Routine diabetes follow-up. Reports medication adherence. No chest pain or dyspnea.</textarea>

      <label for="template_select">Template</label>
      <div class="row">
        <div>
          <select id="template_select"></select>
        </div>
        <div>
          <input id="template_name" type="text" placeholder="Template name" />
        </div>
      </div>

      <div class="actions" style="margin-top:8px;">
        <button id="newTemplateBtn" type="button">New</button>
        <button id="saveTemplateBtn" type="button">Save</button>
        <button id="deleteTemplateBtn" type="button">Delete</button>
      </div>

      <label for="template_body">Template Instructions</label>
      <textarea id="template_body" style="min-height:220px;"></textarea>

      <div class="row">
        <div>
          <label for="max_new_tokens">Max New Tokens</label>
          <input id="max_new_tokens" type="number" min="64" max="2048" step="64" value="512" />
        </div>
        <div>
          <label for="temperature">Temperature</label>
          <input id="temperature" type="number" min="0" max="1.5" step="0.1" value="0.3" />
        </div>
      </div>

      <div class="actions">
        <button id="generateBtn">Generate Note</button>
        <div id="status" class="status"></div>
      </div>

      <div class="meta">
        Proxy target: <span id="target"></span>
      </div>
    </section>

    <section class="card">
      <label>Generated Progress Note</label>
      <pre id="output">Output will appear here...</pre>
    </section>
  </div>

  <script>
    const output = document.getElementById("output");
    const statusEl = document.getElementById("status");
    const btn = document.getElementById("generateBtn");
    const target = document.getElementById("target");
    const templateSelect = document.getElementById("template_select");
    const templateName = document.getElementById("template_name");
    const templateBody = document.getElementById("template_body");
    const newTemplateBtn = document.getElementById("newTemplateBtn");
    const saveTemplateBtn = document.getElementById("saveTemplateBtn");
    const deleteTemplateBtn = document.getElementById("deleteTemplateBtn");

    let templates = [];
    let selectedTemplateId = null;

    function setStatus(message, isError=false) {
      statusEl.className = isError ? "status error" : "status";
      statusEl.textContent = message;
    }

    function renderTemplateOptions() {
      templateSelect.innerHTML = "";
      for (const t of templates) {
        const opt = document.createElement("option");
        opt.value = String(t.id);
        opt.textContent = t.is_default ? `${t.name} (default)` : t.name;
        templateSelect.appendChild(opt);
      }
      if (templates.length > 0) {
        if (!selectedTemplateId) selectedTemplateId = templates[0].id;
        templateSelect.value = String(selectedTemplateId);
      }
    }

    function loadSelectedTemplateIntoEditor() {
      const t = templates.find((x) => x.id === Number(templateSelect.value));
      if (!t) return;
      selectedTemplateId = t.id;
      templateName.value = t.name;
      templateBody.value = t.body;
      deleteTemplateBtn.disabled = !!t.is_default;
    }

    async function refreshTemplates(preferredId = null) {
      const resp = await fetch("/templates");
      const data = await resp.json();
      templates = data.templates || [];
      if (preferredId) selectedTemplateId = preferredId;
      renderTemplateOptions();
      loadSelectedTemplateIntoEditor();
    }

    async function createTemplate() {
      const name = (templateName.value || "").trim();
      const body = (templateBody.value || "").trim();
      if (!name || !body) {
        setStatus("Template name and body are required.", true);
        return;
      }
      const resp = await fetch("/templates", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, body }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Failed to create template");
      }
      await refreshTemplates(data.template.id);
      setStatus("Template created.");
    }

    async function saveTemplate() {
      const id = Number(templateSelect.value || 0);
      const name = (templateName.value || "").trim();
      const body = (templateBody.value || "").trim();
      if (!id) {
        await createTemplate();
        return;
      }
      if (!name || !body) {
        setStatus("Template name and body are required.", true);
        return;
      }
      const resp = await fetch(`/templates/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, body }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Failed to update template");
      }
      await refreshTemplates(id);
      setStatus("Template updated.");
    }

    async function deleteCurrentTemplate() {
      const id = Number(templateSelect.value || 0);
      if (!id) return;
      const ok = window.confirm("Delete this template?");
      if (!ok) return;
      const resp = await fetch(`/templates/${id}`, { method: "DELETE" });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Failed to delete template");
      }
      await refreshTemplates();
      setStatus("Template deleted.");
    }

    async function generate() {
      setStatus("Sending request...");
      btn.disabled = true;

      const payload = {
        patient_context: document.getElementById("patient_context").value,
        provider_notes: document.getElementById("provider_notes").value,
        template_id: Number(templateSelect.value || 0),
        template: templateBody.value,
        max_new_tokens: Number(document.getElementById("max_new_tokens").value || 512),
        temperature: Number(document.getElementById("temperature").value || 0.3),
      };

      output.textContent = "Generating...";
      const started = performance.now();

      try {
        const resp = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || "Request failed.");
        }

        output.textContent = data.progress_note || "No text generated.";
        const sec = ((performance.now() - started) / 1000).toFixed(1);
        setStatus(`Done in ${sec}s`);
      } catch (err) {
        output.textContent = "";
        setStatus(err.message || "Unexpected error.", true);
      } finally {
        btn.disabled = false;
      }
    }

    btn.addEventListener("click", generate);
    templateSelect.addEventListener("change", loadSelectedTemplateIntoEditor);

    newTemplateBtn.addEventListener("click", () => {
      templateSelect.value = "";
      selectedTemplateId = null;
      templateName.value = "";
      templateBody.value = "";
      deleteTemplateBtn.disabled = true;
      setStatus("Enter a new template name and body, then click Save.");
    });

    saveTemplateBtn.addEventListener("click", async () => {
      try {
        await saveTemplate();
      } catch (err) {
        setStatus(err.message || "Save failed.", true);
      }
    });

    deleteTemplateBtn.addEventListener("click", async () => {
      try {
        await deleteCurrentTemplate();
      } catch (err) {
        setStatus(err.message || "Delete failed.", true);
      }
    });

    (async () => {
      try {
        await refreshTemplates();
      } catch (err) {
        setStatus("Template loading failed.", true);
      }
    })();

    fetch("/target")
      .then((r) => r.json())
      .then((d) => { target.textContent = d.modal_api_url; })
      .catch(() => { target.textContent = "Unavailable"; });
  </script>
</body>
</html>
    """


@app.get("/target")
def get_target() -> dict[str, str]:
    return {"modal_api_url": MODAL_API_URL}


@app.get("/templates")
def get_templates() -> dict[str, Any]:
    records = list_templates()
    return {
        "templates": [
            {
                "id": r.id,
                "name": r.name,
                "body": r.body,
                "is_default": r.is_default,
            }
            for r in records
        ]
    }


@app.post("/templates")
def post_template(payload: TemplateCreateRequest) -> dict[str, Any]:
    try:
        record = create_template(payload.name, payload.body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Create failed: {exc}") from exc
    return {
        "template": {
            "id": record.id,
            "name": record.name,
            "body": record.body,
            "is_default": record.is_default,
        }
    }


@app.put("/templates/{template_id}")
def put_template(template_id: int, payload: TemplateUpdateRequest) -> dict[str, Any]:
    try:
        record = update_template(template_id, payload.name, payload.body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Update failed: {exc}") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "template": {
            "id": record.id,
            "name": record.name,
            "body": record.body,
            "is_default": record.is_default,
        }
    }


@app.delete("/templates/{template_id}")
def delete_template_route(template_id: int) -> dict[str, Any]:
    try:
        deleted = delete_template(template_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Delete failed: {exc}") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"deleted": True}


@app.post("/generate")
async def generate(payload: GenerateRequest) -> dict[str, Any]:
    template_text = (payload.template or "").strip()
    if payload.template_id:
        record = get_template(payload.template_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Selected template not found")
        template_text = record.body
    if not template_text:
        template_text = get_default_template().body

    optimized_context = optimize_ingestion(payload.patient_context, payload.provider_notes)

    upstream_payload = {
        "patient_context": optimized_context,
        "provider_notes": payload.provider_notes,
        "template": template_text,
        "max_new_tokens": payload.max_new_tokens,
        "temperature": payload.temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(MODAL_API_URL, json=upstream_payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text
        raise HTTPException(status_code=502, detail=f"Upstream returned {response.status_code}: {detail}")

    data = response.json()
    return {"progress_note": data.get("progress_note", "")}


@app.post("/ingestion-preview")
def ingestion_preview(payload: IngestionPreviewRequest) -> dict[str, str]:
    return {"optimized_context": optimize_ingestion(payload.patient_context, payload.provider_notes)}
