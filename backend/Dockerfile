# FinRAG 2.0 backend image.
# Build context is the backend/ directory (see infra/docker-compose.yml).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# pdf2image shells out to pdftoppm and pytesseract to tesseract; both are
# needed for the ingestion path, so they belong in the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer survives source-only edits.
COPY pyproject.toml ./
RUN pip install --upgrade pip

COPY app ./app
RUN pip install .

# Run as an unprivileged user and give it the writable data volume.
RUN useradd --create-home --shell /usr/sbin/nologin finrag \
    && mkdir -p /app/data/storage \
    && chown -R finrag:finrag /app
USER finrag

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
