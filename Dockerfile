FROM python:3.11-slim

WORKDIR /app

# torch.compile's Inductor CPU backend generates and compiles C++ at runtime for
# the fused kernels -- python:3.11-slim has no compiler by default, and without
# one COMPILE_MODEL=true fails at startup (InvalidCxxCompiler), not at import time.
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
# CPU-only wheel index: the default PyPI torch wheel for linux/aarch64 pulls in
# multiple GB of NVIDIA CUDA libraries (nvidia_cudnn_cu13 alone is 650MB+) even
# though this cluster is CPU-only -- pure dead weight without the CPU-only index.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-serving.txt

COPY src/ src/
COPY checkpoints/ checkpoints/

ENV MODEL_DIR=checkpoints/distilbert-ranker-50k-1ep

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
