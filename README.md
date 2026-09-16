# Compiler-Optimized Pairwise Ranking Pipeline

Fine-tune a DistilBERT cross-encoder to rank pairs of Yelp reviews by user
preference (Beli-style "which would you rank higher?"), then take it through
a full production path: profiling, compiler-level optimization
(`torch.compile` / TVM), IR-level INT8 quantization, adaptive-batching +
canary serving, and Kubernetes autoscaling.

## Status

Phase 1 (fine-tuning) — in progress. See `configs/phase1.yaml` and
`src/ranking/`.

## Roadmap

1. **Fine-tune** a DistilBERT cross-encoder on Yelp review pairs (this repo's
   current focus).
2. **Baseline serving + profiling** — FastAPI + Docker, Locust load test,
   `torch.profiler` bottleneck report.
3. **Compiler optimization** — `torch.compile` and TVM (LLVM/CPU target;
   Triton is out of scope, no CUDA on this hardware) vs. eager mode.
4. **IR-level quantization** — INT8 via TVM `relay.quantize` (or PyTorch FX
   graph-mode quantization as fallback), with a latency/memory/accuracy
   trade-off table.
5. **Adaptive batching + canary deployment** — dynamic batching window,
   full-precision vs. quantized model behind a routing layer, automatic
   rollback on error-rate/latency breach.
6. **Kubernetes autoscaling** — full stack on k3d/minikube, HPA on
   queue-depth, Prometheus + Grafana dashboards.

## Data

This project uses the raw [Yelp Open Dataset](https://www.yelp.com/dataset)
(not the `yelp_review_full` HF dataset, which lacks `user_id` grouping).
See `data/README.md` for expected file layout — the dataset itself is not
committed to this repo.

## Setup

```bash
pip install -r requirements.txt
```
