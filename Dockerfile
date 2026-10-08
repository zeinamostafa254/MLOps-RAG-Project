
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml .
COPY src ./src
COPY scripts ./scripts
COPY evaluation ./evaluation
COPY tests ./tests
COPY data/processed ./data/processed

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
