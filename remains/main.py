# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import datetime as dt
import os

from config.settings import DEBUG_ROOT
from config.quota import bump_and_check_limit
from models.schemas import OCRReq
from services.ocr_service import run_pta_free_ocr
from utils.io_utils import ensure_dir

app = FastAPI(title="Agnel OCR — Google Cloud Vision (with free fallback)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr/pta_free")
def ocr_pta_free(req: OCRReq):
    ok, usage = bump_and_check_limit(limit=1000)
    if not ok:
        return JSONResponse(
            status_code=429,
            content={"error": "Monthly free-tier limit reached.", "usage": usage},
        )

    debug_dir = None
    if req.debug:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        debug_dir = ensure_dir(os.path.join(DEBUG_ROOT, f"{stamp}_pta_free"))

    try:
        result = run_pta_free_ocr(req.imageBase64, debug_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision failed: {e}")

    return result


@app.get("/", response_class=HTMLResponse)
def index():
    HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"/><title>Agnel OCR — Vision</title>
<style>
body{font-family:system-ui,Arial;margin:20px}
.card{max-width:980px;margin:auto;border:1px solid #ddd;border-radius:12px;padding:16px}
.row{display:flex;gap:16px}
img{max-height:420px;border:1px solid #ddd;border-radius:8px}
.btn{background:#2563eb;color:#fff;border:0;padding:8px 12px;border-radius:8px;cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word}
label{font-weight:600}
small{color:#555}
</style></head>
<body>
<div class="card">
  <h2>Agnel OCR — Google Vision <small>(with free fallback)</small></h2>
  <div>
    <label>Mode</label>
    <select id="mode">
      <option value="pta_free" selected>PTA (label-based)</option>
    </select>
  </div>
  <div style="margin-top:8px;">
    <label><input type="checkbox" id="debugFlag"/> Save debug images</label>
  </div>
  <div style="margin-top:8px;">
    <input type="file" id="file" accept="image/*,pdf"/>
  </div>
  <div style="margin-top:12px;">
    <button class="btn" onclick="run()">Run OCR</button>
  </div>
  <div id="preview" style="margin-top:16px;"></div>
  <h3>Fields</h3><pre id="fields"></pre>
  <h3>Usage</h3><pre id="conf"></pre>
  <h3>Raw text</h3><pre id="text"></pre>
</div>
<script>
  function fileToDataURL(f){
    return new Promise(res=>{
      const r=new FileReader();
      r.onload=()=>res(r.result);
      r.readAsDataURL(f);
    });
  }

  async function run(){
    const f = document.getElementById('file').files[0];
    if(!f){ alert('Choose an image'); return; }
    const b64 = await fileToDataURL(f);
    document.getElementById('preview').innerHTML = '<img src="'+b64+'"/>';

    const dbg = document.getElementById('debugFlag').checked;
    const payload = { imageBase64: b64, template: null, debug: dbg };
    const resp = await fetch('/ocr/pta_free', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const out = await resp.json();

    document.getElementById('fields').textContent = JSON.stringify(out.fields || out.error || {}, null, 2);
    document.getElementById('conf').textContent   = JSON.stringify(out.usage || {}, null, 2);
    document.getElementById('text').textContent   = out.text || '';
  }
</script>
</body></html>
"""
    return HTMLResponse(HTML_PAGE)
