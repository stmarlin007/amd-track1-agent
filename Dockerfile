# AMD Developer Hackathon — Track 1 General-Purpose AI Agent
# Build for the grading VM's architecture explicitly:
#   docker buildx build --platform linux/amd64 -t <you>/amd-track1-agent:latest --push .
FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU-only build of llama-cpp-python (grading VM has no GPU: 4GB RAM / 2 vCPU)
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.sh .
RUN chmod +x download_model.sh && ./download_model.sh

COPY main.py categories.py local_backend.py fireworks_backend.py ./

# /input and /output are provided by the grading harness at runtime; create
# them here too so the image also runs standalone for local testing.
RUN mkdir -p /input /output

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
