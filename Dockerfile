FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG CODEX_VERSION=0.171.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    nodejs \
    npm \
    git \
    && npm install -g "@openai/codex@${CODEX_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

RUN mkdir -p /data/recovery /workspace

EXPOSE 8000

CMD ["uvicorn", "app.server:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
