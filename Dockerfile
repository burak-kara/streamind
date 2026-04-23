FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/root/.ollama/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        ffmpeg \
        build-essential \
        pkg-config \
        git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY pyproject.toml ./
RUN uv venv --python 3.12 .venv \
    && uv sync --extra dev --no-install-project

RUN ollama serve & \
    OLLAMA_PID=$! ; \
    for i in $(seq 1 30); do \
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break ; \
        sleep 1 ; \
    done ; \
    ollama pull qwen3.5:4b ; \
    kill "$OLLAMA_PID" ; \
    wait "$OLLAMA_PID" 2>/dev/null || true

COPY plugins/ ./plugins/
COPY pipelines/ ./pipelines/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY docker/entrypoint-pipeline.sh /usr/local/bin/entrypoint-pipeline.sh
RUN chmod +x /usr/local/bin/entrypoint-pipeline.sh tools/run_pipeline.sh

RUN mkdir -p /app/tmp /app/results

EXPOSE 8888/udp

ENV SUMMARIZER_PROFILE=ollama-qwen3.5-4b \
    WINDOW_SECONDS=300

ENTRYPOINT ["/usr/local/bin/entrypoint-pipeline.sh"]
