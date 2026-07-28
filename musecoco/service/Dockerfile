# MuseCoco Stage-1 (Text-to-Attribute) — Render-compatible CPU build
# Build context: repo root (/Volumes/Builds/muzic/musecoco/)
#   docker build -f service/Dockerfile -t musecoco-t2a .

FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY service/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the full repo (local files needed: num_labels.json, att_key.json, stage2_pre.py)
COPY . .

# Pre-download the T2A model so first request is fast
RUN python3 -c "\
from transformers import AutoTokenizer, BertForSequenceClassification; \
AutoTokenizer.from_pretrained('XinXuNLPer/MuseCoco_text2attribute'); \
BertForSequenceClassification.from_pretrained('XinXuNLPer/MuseCoco_text2attribute'); \
"

ENV PYTHONPATH=/app

EXPOSE 8090

CMD uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8090}
