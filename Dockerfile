FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    VLLM_USE_FLASHINFER_SAMPLER=0

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

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.12 .venv \
    && uv sync --no-install-project

# Bake LLM weights into image (no runtime download).
COPY tools/fetch_models.sh ./tools/fetch_models.sh
RUN chmod +x tools/fetch_models.sh \
    && ./tools/fetch_models.sh cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4 qwen3.5-4b-awq

# Pre-populate faster-whisper model (submission container has no network).
# faster-whisper accepts a local dir path as model_name; config-base.json
# points to this path inside the container.
RUN uv run python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('deepdml/faster-whisper-large-v3-turbo-ct2', local_dir='./models/faster-whisper-large-v3-turbo')"


COPY plugins/ ./plugins/
COPY pipelines/ ./pipelines/
COPY tools/ ./tools/
COPY docker/entrypoint-pipeline.sh /usr/local/bin/entrypoint-pipeline.sh
RUN chmod +x /usr/local/bin/entrypoint-pipeline.sh tools/run_pipeline.sh

RUN mkdir -p /app/tmp /app/results

EXPOSE 8888/udp

ENV SUMMARIZER_PROFILE=vllm-qwen3.5-4b-awq \
    WINDOW_SECONDS=300

ENTRYPOINT ["/usr/local/bin/entrypoint-pipeline.sh"]
