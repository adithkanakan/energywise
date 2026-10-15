FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt requirements.txt
# The cache mount keeps already-downloaded wheels between attempts, so a
# retry after a network failure doesn't start from zero.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 10 -r requirements.txt

# LightGBM needs the OpenMP runtime, which the slim base image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ ./backend/
COPY src/ ./src/

ENV PYTHONPATH=/app/src:/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
