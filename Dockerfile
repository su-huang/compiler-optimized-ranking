FROM python:3.11-slim

WORKDIR /app

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
