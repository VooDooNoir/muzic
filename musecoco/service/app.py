"""
MuseCoco Service — Text-to-Attribute (Stage 1 only)
  Stage 1: Text-to-Attribute — BERT classifies descriptions into music attributes
  Stage 2: Stubbed (returns 501)

Deploy target: Render (Docker, free tier, CPU-only)
"""
import os
import json
import base64
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, BertForSequenceClassification

app = FastAPI(
    title="MuseCoco Stage-1 (Text-to-Attribute)",
    description="Text-to-Music Attribute Generation (Microsoft Muzic)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ──────────────────────────────────────────────────────────────
API_SECRET = os.getenv("MUSECOGO_API_SECRET", "")
T2A_MODEL = os.getenv("T2A_MODEL", "XinXuNLPer/MuseCoco_text2attribute")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths to local MuseCoco repo files needed for attribute mapping
ROOT = Path(__file__).parent.parent.resolve()
NUM_LABELS_PATH = ROOT / "1-text2attribute_model" / "num_labels.json"
ATT_KEY_PATH = ROOT / "1-text2attribute_model" / "data" / "att_key.json"

# ── Load models at startup ─────────────────────────────────────────────────────
print(f"[startup] Loading T2A model {T2A_MODEL} on {DEVICE}...")
_tokenizer = AutoTokenizer.from_pretrained(T2A_MODEL)
_model = BertForSequenceClassification.from_pretrained(T2A_MODEL)
_model.to(DEVICE)
_model.eval()
print("[startup] T2A model loaded.")

with open(NUM_LABELS_PATH) as f:
    NUM_LABELS = json.load(f)
with open(ATT_KEY_PATH) as f:
    ATT_KEY = json.load(f)
# ───────────────────────────────────────────────────────────────────────────────


# ── Schemas ────────────────────────────────────────────────────────────────────
class TextToAttributeRequest(BaseModel):
    text: str


class TextToAttributeResponse(BaseModel):
    status: str
    text: str
    attributes: dict
    stage2_available: bool = False


class Stage2Request(BaseModel):
    attributes: dict
    temperature: float = 1.0
    top_k: int = 15
    num_samples: int = 2


class Stage2Response(BaseModel):
    status: str
    midi_files: list[str] = []
    midi_base64: list[str] = []
# ───────────────────────────────────────────────────────────────────────────────


# ── Helpers ────────────────────────────────────────────────────────────────────
def text_to_attributes(text: str) -> dict:
    """Stage 1: Convert a text description into structured music attributes."""
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    attributes = {}
    start = 0
    for attr_name, count in NUM_LABELS.items():
        end = start + count
        attr_logits = logits[0, start:end]
        pred = torch.argmax(attr_logits).item()
        values = ATT_KEY.get(attr_name, [])
        attributes[attr_name] = values[pred] if pred < len(values) else "unknown"
        start = end

    return attributes
# ───────────────────────────────────────────────────────────────────────────────


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": T2A_MODEL,
        "stage2": False,
    }


@app.post("/generate", response_model=TextToAttributeResponse)
def generate(req: TextToAttributeRequest, x_api_key: str = Header(None)):
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text description is required")

    try:
        attributes = text_to_attributes(req.text)
        return TextToAttributeResponse(
            status="success",
            text=req.text,
            attributes=attributes,
            stage2_available=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-stage2", response_model=Stage2Response)
def generate_stage2(req: Stage2Request, x_api_key: str = Header(None)):
    """Stage 2 (attribute-to-MIDI) — not yet deployed."""
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")

    raise HTTPException(
        status_code=501,
        detail="Stage 2 (attribute-to-music) is not implemented in this deployment. "
        "The fairseq generation model requires GPU resources not available on the free plan.",
    )
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port)
