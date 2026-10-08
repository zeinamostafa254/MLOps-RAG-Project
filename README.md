
# Egyptian Legal RAG — MLOps Final Project

Arabic/English Retrieval-Augmented Generation over the Egyptian Civil Code.

## Business problem

A legal organization needs answers grounded in Egyptian Civil Code articles. A legally useful answer must expose the article citation used to support it; hallucinated legal provisions are unacceptable.

The supplied `orig_data.json` is already article-structured and bilingual. It contains an `Article N` key, Arabic text, English text, and hierarchical metadata. The project therefore uses a **normalization/validation ingestion stage** instead of pretending that JSON is raw PDF text.

For full Track B provenance, place the original bilingual PDF at:

`data/raw/1576751803.pdf`

The handbook requires both PDF and derived JSON to be DVC tracked and `dvc repro` to rebuild the corpus. The supplied JSON is the authoritative ready-to-use dataset; the PDF extractor is included as a reproducibility path, but its output should be compared against the supplied JSON before replacing it.

## Architecture

```text
                         ┌─────────────────────┐
                         │ PDF / supplied JSON  │
                         └──────────┬──────────┘
                                    │
                         normalize + validate
                                    │
                                    ▼
                         data/processed/corpus.json
                                    │
                              DVC pipeline
                                    │
                                    ▼
                        article-aware chunking
                                    │
                         multilingual embeddings
                                    │
                                    ▼
                              ChromaDB
                                    │
 question ──► guardrails ──► hybrid retrieval ──► reranker
                                    │
                                    ▼
                               vLLM LLM
                                    │
                          grounded answer + citations
                                    │
                   ┌────────────────┼────────────────┐
                   ▼                ▼                ▼
               RAGAS             Langfuse       Prometheus
                   │                │                │
                   ▼                ▼                ▼
                MLflow          Grafana          alerts
```

## 3-command setup

```bash
pip install -e .
python scripts/prepare_corpus.py
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

For Docker:

```bash
docker compose up --build
curl http://localhost:8000/health
```

## API

`POST /ask`

```json
{
  "question": "ما هي شروط استعمال الحق بصورة غير مشروعة؟"
}
```

Response:

```json
{
  "answer": "...",
  "sources": ["Egyptian Civil Code, Article 5"]
}
```

`GET /health` returns the service status and number of indexed chunks.

`POST /ask/stream` returns newline-delimited JSON for progressive command-line consumption.

## Evaluation

Run:

```bash
python evaluation/build_questions.py
python evaluation/run_ragas.py
```

The evaluation script targets at least 50 questions and records all four RAGAS metrics:

- faithfulness
- answer relevancy
- context precision
- context recall

The CI quality gate fails if the 20-question smoke evaluation has faithfulness below `0.75`.

## MLOps commands

```bash
dvc repro
mlflow ui --port 5000
pytest -q
ruff check .
locust -f loadtest/locustfile.py --host http://localhost:8000
```

## Optimization

The optimization module compares:

1. baseline generator
2. AWQ 4-bit generator
3. baseline retriever
4. reranked retriever

Record actual results in `reports/benchmark.csv`. Never fabricate speed, memory, or quality improvements.

## Monitoring

- RAGAS faithfulness
- query-embedding cosine drift
- token usage / estimated cost
- request latency
- Langfuse trace per request
- PII guardrail
- Prometheus metrics
- Grafana dashboard and faithfulness alert

## Important data note

The handbook asks for PDF → structured JSON conversion. Your supplied JSON already contains article-level Arabic/English records, so this repository does not claim that the JSON was produced by our extractor. If you add the original PDF, run `scripts/extract_pdf.py`, compare the result with `data/processed/corpus.json`, and document discrepancies.

## Rubric evidence map

| Rubric | Evidence |
|---|---|
| 01 Code & packaging | `pyproject.toml`, `src/`, typed classes |
| 02 API | `src/api.py`, Pydantic validation, `/health`, `/ask` |
| 03 Docker | `Dockerfile`, `docker-compose.yml`, README |
| 04 MLflow | `src/mlflow_tracking.py`, `scripts/run_chunk_experiments.py` |
| 05 DVC | `dvc.yaml`, `dvc.yaml` stages, tracked data |
| 06 CI/CD | `.github/workflows/ci.yml`, quality gate |
| 07 Production serving | `src/bento_service.py`, `vllm` service, Locust |
| 08 Monitoring | RAGAS, Langfuse, Prometheus, Grafana |
| 09 Peer review | complete manually after assignment |
| 10 README/architecture | this README + `docs/architecture.md` |
