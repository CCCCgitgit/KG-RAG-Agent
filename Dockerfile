# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

# -----------------------------------------------------------------------------
# Build stage
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip wheel \
        --wheel-dir /wheels \
        ".[all]"


# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    KG_RAG_AGENT_ROOT=/app \
    HF_HOME=/app/outputs/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/outputs/cache/sentence_transformers

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgomp1 \
    && groupadd --gid "${APP_GID}" kgagent \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --create-home \
        --shell /usr/sbin/nologin \
        kgagent \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels

RUN python -m pip install \
        --no-index \
        --find-links=/wheels \
        "kg-rag-agent[all]==0.1.0" \
    && rm -rf /wheels

COPY --chown=kgagent:kgagent configs ./configs
COPY --chown=kgagent:kgagent scripts ./scripts

RUN mkdir -p \
        /app/data/kg \
        /app/data/vector_store \
        /app/logs \
        /app/outputs/evaluation \
        /app/outputs/memory \
        /app/outputs/cache \
    && chown -R kgagent:kgagent \
        /app/data \
        /app/logs \
        /app/outputs

USER kgagent

EXPOSE 8000

STOPSIGNAL SIGTERM

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()" || exit 1

CMD [
    "python",
    "-m",
    "kg_rag_agent",
    "--serve",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
