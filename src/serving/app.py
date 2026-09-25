"""FastAPI serving layer for the pairwise ranking cross-encoder.

Phase 2: baseline eager-mode single-instance serving.
Phase 5: adaptive request batching + canary routing with automatic rollback.
Compiler optimizations (torch.compile/TVM) and quantization land in earlier
benchmark scripts (Phase 3/4) but aren't wired into this service by default.
"""

import logging
import os
import random
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.serving.batching import AdaptiveBatcher
from src.serving.canary import CanaryRouter

logging.basicConfig(level=logging.INFO)

STABLE_MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints/distilbert-ranker-50k-1ep")
CANARY_MODEL_DIR = os.environ.get("CANARY_MODEL_DIR", STABLE_MODEL_DIR)
CANARY_TRAFFIC_FRACTION = float(os.environ.get("CANARY_TRAFFIC_FRACTION", "0.05"))
# For demoing automatic rollback: forces this fraction of canary requests to fail
# outright, simulating a broken deployment. 0.0 in normal operation.
CANARY_CHAOS_FAILURE_RATE = float(os.environ.get("CANARY_CHAOS_FAILURE_RATE", "0.0"))

model_state: dict = {}
# AdaptiveBatcher now runs predict_fn via asyncio.to_thread (so the event loop isn't
# blocked during inference), which means the stable and canary batchers' forward
# passes can now land on different OS threads at the same time. PyTorch's MPS
# backend isn't thread-safe for concurrent forward passes (same segfault as Phase 2),
# so both batchers share this lock to serialize actual GPU/CPU compute.
inference_lock = threading.Lock()


def _load_model(model_dir: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def _predict_batch(pairs: list[tuple[str, str]], model, tokenizer, device) -> list[tuple[str, float]]:
    text_a_list = [p[0] for p in pairs]
    text_b_list = [p[1] for p in pairs]

    inputs = tokenizer(
        text_a_list,
        text_b_list,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with inference_lock, torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    results = []
    for row in probs:
        label = int(torch.argmax(row).item())
        confidence = float(row[label].item())
        results.append(("text_a" if label == 1 else "text_b", confidence))
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    stable_model, stable_tokenizer = _load_model(STABLE_MODEL_DIR, device)
    canary_model, canary_tokenizer = _load_model(CANARY_MODEL_DIR, device)

    stable_batcher = AdaptiveBatcher(
        predict_fn=lambda pairs: _predict_batch(pairs, stable_model, stable_tokenizer, device)
    )
    canary_batcher = AdaptiveBatcher(
        predict_fn=lambda pairs: _predict_batch(pairs, canary_model, canary_tokenizer, device)
    )
    stable_batcher.start()
    canary_batcher.start()

    model_state["stable_batcher"] = stable_batcher
    model_state["canary_batcher"] = canary_batcher
    model_state["canary_router"] = CanaryRouter(canary_traffic_fraction=CANARY_TRAFFIC_FRACTION)
    model_state["ready"] = True

    yield

    await stable_batcher.stop()
    await canary_batcher.stop()
    model_state.clear()


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    text_a: str
    text_b: str


class PredictResponse(BaseModel):
    preferred: str
    confidence: float
    served_by: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model_state.get("ready") else "loading"}


@app.get("/canary/status")
def canary_status() -> dict:
    return model_state["canary_router"].status()


@app.post("/canary/reset")
def canary_reset() -> dict:
    model_state["canary_router"].reset()
    return model_state["canary_router"].status()


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    router: CanaryRouter = model_state["canary_router"]
    use_canary = router.should_route_to_canary()

    if use_canary:
        start = time.perf_counter()
        try:
            if random.random() < CANARY_CHAOS_FAILURE_RATE:
                raise RuntimeError("simulated canary failure (chaos mode)")
            preferred, confidence = await model_state["canary_batcher"].submit((request.text_a, request.text_b))
            latency_ms = (time.perf_counter() - start) * 1000
            router.record_canary_result(success=True, latency_ms=latency_ms)
            return PredictResponse(preferred=preferred, confidence=confidence, served_by="canary")
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            router.record_canary_result(success=False, latency_ms=latency_ms)
            # fall through to stable so the caller still gets a real answer

    preferred, confidence = await model_state["stable_batcher"].submit((request.text_a, request.text_b))
    return PredictResponse(preferred=preferred, confidence=confidence, served_by="stable")
