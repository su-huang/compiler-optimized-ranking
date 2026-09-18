"""FastAPI serving layer for the pairwise ranking cross-encoder.

Baseline (eager-mode, single-instance) serving for Phase 2 profiling/load-testing.
Compiler optimizations (torch.compile/TVM) and quantization land in later phases.
"""

import os
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints/distilbert-ranker-50k-1ep")

model_state: dict = {}
# FastAPI runs sync endpoints in a thread pool; PyTorch's MPS backend is not
# thread-safe for concurrent forward passes on the same model and segfaults under
# concurrent load without this. Also foreshadows the batching queue in Phase 5,
# which will replace this with real request batching instead of serialization.
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    model_state["device"] = device

    yield

    model_state.clear()


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    text_a: str
    text_b: str


class PredictResponse(BaseModel):
    preferred: str
    confidence: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "model" in model_state else "loading"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    device = model_state["device"]

    inputs = tokenizer(
        request.text_a,
        request.text_b,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with inference_lock, torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    label = int(torch.argmax(probs).item())
    confidence = float(probs[label].item())

    return PredictResponse(
        preferred="text_a" if label == 1 else "text_b",
        confidence=confidence,
    )
