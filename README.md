# Egyptian Legal RAG — MLOps Final Project

**Bilingual Arabic/English Retrieval-Augmented Generation for the Egyptian Civil Code**

An MLOps project developed as part of the ITI × MLOps MENA Practitioner track, focusing on legal document retrieval, grounded question answering, reproducibility, evaluation, and operational monitoring.

**Repository:** https://github.com/zeinamostafa254/MLOps-RAG-Project

> **Project status:** Core RAG components, corpus preparation, experiment tracking, and evaluation infrastructure have been developed. See the completion checklist below for the requirements that still need end-to-end verification. This README distinguishes implemented components from work that remains to be validated.

## 1. Project Overview

Legal question answering requires more than generating a plausible response. Answers should be grounded in the applicable legal text, accompanied by verifiable citations, and designed to avoid unsupported claims.

This project implements a Retrieval-Augmented Generation (RAG) system for the Egyptian Civil Code. It retrieves relevant legal provisions from a bilingual Arabic/English corpus and uses a language model to generate answers based on the retrieved evidence.

### Objectives

- Prepare and validate structured Arabic/English legal documents.
- Retrieve relevant legal provisions using multilingual embeddings.
- Store and search document representations with ChromaDB.
- Improve retrieval relevance through reranking.
- Generate answers using a locally hosted language model.
- Return legal article citations with generated answers.
- Track chunking experiments with MLflow.
- Version data and processing stages with DVC.
- Evaluate answer quality with RAGAS.
- Expose an API and develop monitoring and reproducibility workflows.

**Disclaimer:** This project is an educational research prototype, not a substitute for advice from a qualified Egyptian lawyer. Generated answers and citations must be checked against authoritative legal sources.

## 2. Architecture

```text
Bilingual Egyptian Civil Code
            |
            v
Structured JSON Corpus
            |
            v
Normalization and Validation
            |
            v
Article-Aware Chunking
            |
            v
Multilingual E5 Embeddings
            |
            v
        ChromaDB
            |
User Question
     |
     v
Input Validation
     |
     v
Vector Retrieval
     |
     v
Reranking
     |
     v
Ollama-Hosted Language Model
     |
     v
Grounded Answer + Article Citations
```

Supporting MLOps components include:

- **MLflow:** Experiment tracking and configuration comparison.
- **DVC:** Data and pipeline versioning.
- **FastAPI:** HTTP interface for question answering.
- **RAGAS:** RAG quality evaluation.
- **Prometheus:** Metrics collection and exposure.
- **Langfuse:** LLM tracing, where configured and operational.
- **Grafana:** Monitoring dashboards, where configured and operational.
- **Docker:** Containerization and service orchestration.
- **GitHub Actions:** Automated testing and continuous integration, subject to workflow verification.

The diagram describes the intended processing flow. Optional components must be verified against the active configuration before being considered operational.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| API | FastAPI, Pydantic, Uvicorn |
| LLM serving in the development configuration | Ollama |
| Generation model | `qwen2.5:7b` |
| Embedding model | `intfloat/multilingual-e5-base` |
| Vector database | ChromaDB |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Evaluation | RAGAS |
| Metrics | Prometheus |
| LLM tracing | Langfuse integration |
| Dashboards | Grafana integration |
| Testing | pytest |
| Code quality | Ruff |
| Containerization | Docker and Docker Compose |

Model names and dependency versions should be checked against the final `.env.example`, `pyproject.toml`, and Docker configuration.

### Why Ollama?

The development configuration uses Ollama to run the language model locally. This avoids making vLLM a mandatory part of the default generation path.

The handbook references vLLM for its serving workflow. Using Ollama is therefore a documented implementation deviation, not a claim that the original serving requirement was fulfilled exactly. Any optional vLLM configuration should be identified separately.

## 4. Dataset and Preprocessing

The project uses a mentor-provided, structured JSON dataset containing Arabic and English legal text.

The main preprocessing workflow normalizes and validates the existing JSON corpus. It should not be described as PDF-to-JSON extraction unless that conversion is actually performed by the documented pipeline.

### Preprocessing objectives

- Validate the structure and required fields of each record.
- Preserve article identifiers and relevant legal metadata.
- Normalize Arabic text consistently.
- Identify records with missing Arabic or English text.
- Preserve repealed-article information rather than silently deleting it.
- Detect unexplained article-number gaps.
- Produce a validated corpus for indexing.

During development, the processed corpus contained approximately 1,093 records. The final count must be verified after running the current preprocessing script.

The original bilingual PDF and the supplied JSON should be treated as separate source artifacts until their relationship has been validated. Both should be tracked or made reproducibly accessible as required by the course.

## 5. Retrieval and Generation

The retrieval pipeline is designed to:

1. Accept a legal question.
2. Encode the question using the configured multilingual embedding model.
3. Retrieve relevant document chunks from ChromaDB.
4. Rerank retrieved passages when the reranker is enabled.
5. Provide the selected evidence to the configured language model.
6. Generate a response grounded in the supplied context.
7. Return article-level citations so the user can verify the evidence.

The development configuration uses a chunk size of 1,200 and an overlap of 150 characters as the baseline experiment. Other configurations were explored through MLflow.

The implementation should return citations that identify legal articles, rather than exposing only internal vector IDs. Unsupported questions should be handled conservatively, with no invented legal citations.

## 6. API

The application uses FastAPI to expose the RAG functionality.

The expected endpoints are:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Report service health |
| GET | `/metrics` | Expose Prometheus metrics |
| POST | `/ask` | Answer a legal question |
| POST | `/ask/stream` | Stream a response using the implemented streaming format |

Verify these routes using the current application and its OpenAPI schema at `/docs`.

### Example request

```json
{
  "question": "What does the Civil Code say about abuse of rights?"
}
```

### Intended response structure

```json
{
  "answer": "A response grounded in the retrieved legal provisions.",
  "sources": [
    "Egyptian Civil Code, Article 5"
  ]
}
```

The response above is illustrative and is not a verified legal answer from the running system. The exact field names and citation structure must match the implementation.

### API verification

Start the service:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/metrics`

Test `/ask` with a real question. Also test that invalid or empty requests are rejected with HTTP 422, if that is the API's specified validation behavior.

A healthy health-check response does not establish that the complete RAG pipeline is operational. Generation, retrieval, and citation behavior must be tested independently.

## 7. Installation and Execution

### Prerequisites

- Python 3.11 or 3.12.
- Git.
- Ollama installed and running.
- Sufficient memory and disk space for the configured embedding model, reranker, and LLM.
- Docker Desktop for the containerized workflow.

### Install the project

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Create a local `.env` from `.env.example` and configure the required model, service URLs, and other settings.

Never commit API keys, passwords, access tokens, or credential backups.

### Start the generation model

For the development configuration:

```bash
ollama pull qwen2.5:7b
ollama list
```

Ensure that the model name matches the active project configuration.

### Prepare the corpus and index

```bash
python scripts/prepare_corpus.py
python scripts/build_index.py
```

These commands must be checked against the actual scripts in the final repository.

### Run the API

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### Docker

Validate the configuration before starting services:

```bash
docker compose config
docker compose up --build
```

The reviewer must be able to follow the documented setup without relying on undocumented local files or services.

## 8. MLOps: MLflow Experiment Tracking

The project explored different chunk sizes and overlaps to study their impact on the retrieval pipeline.

| Experiment | Chunk size | Overlap |
|---|---:|---:|
| Small | 800 | 100 |
| Medium-small | 1000 | 100 |
| Baseline | 1200 | 150 |
| Large | 1500 | 200 |
| Very large | 1800 | 250 |

Five experiments were reported during development. Their final metrics, artifacts, and run statuses must be verified in the MLflow tracking store.

The experiment-tracking workflow should record:

- Chunk size and overlap.
- Embedding and generation model identifiers.
- Retrieval and reranking configuration.
- Relevant evaluation metrics.
- Execution artifacts and configuration.
- Results used to select the final configuration.

The handbook also asks for best-model registry and promotion evidence. Confirm whether the selected configuration has been registered and whether the required model lifecycle workflow is implemented.

## 9. Data Versioning and Reproducibility

DVC is used to define and manage the corpus preparation and indexing pipeline.

The documented workflow includes:

1. Normalizing and validating the structured source data.
2. Producing the processed corpus.
3. Building the vector index.

Run:

```bash
dvc status
dvc repro
dvc status
```

A reproducible pipeline requires all necessary inputs, dependencies, and tracked artifacts to be accessible. If a DVC remote is needed, its setup instructions must be documented.

A successful `dvc repro` is stronger evidence than simply having a `dvc.yaml` file in the repository.

## 10. Evaluation with RAGAS

RAG evaluation should measure whether the system retrieves appropriate legal evidence and generates answers that remain grounded in that evidence.

The main evaluation metrics are:

| Metric | Purpose |
|---|---|
| Faithfulness | Measures whether generated claims are supported by the retrieved context |
| Answer relevancy | Measures whether the response addresses the user's question |
| Context precision | Measures the relevance of retrieved context |
| Context recall | Measures whether the retrieved evidence covers the reference answer |

The handbook requires evaluation on at least 50 questions.

A smoke test using RAGAS and Ollama was reported during development, with a faithfulness score of 1.0. This is a preliminary result, not evidence of full-system accuracy or completion of the 50-question evaluation.

The final evaluation report should document:

- Number of questions evaluated.
- Arabic and English coverage.
- Reference answers and expected article citations.
- Actual scores for each metric.
- Model and retrieval configuration.
- Known failures and limitations.

Results should be saved as a versioned artifact, such as `reports/ragas_evaluation.json` or `reports/ragas_evaluation.csv`.

## 11. Monitoring and Observability

The project includes a Prometheus metrics endpoint. Additional monitoring objectives include:

- Request counts and error rates.
- Request latency.
- RAGAS faithfulness tracking.
- Query-embedding drift.
- Token usage and estimated cost.
- Langfuse request traces.
- PII guardrails.
- Grafana dashboards and alert thresholds.

These objectives must be verified against the actual implementation.

For example, exposing `/metrics` does not automatically mean that Grafana is connected or that token cost is being measured. The final submission should include a screenshot or report demonstrating the monitoring components that actually run.

## 12. Optimization and Performance

The handbook expects a measured comparison of the original and optimized system, including generation quantization, reranking changes, evaluation quality, and latency.

For this project, optimization claims should be made only after running the corresponding implementation and benchmark.

A useful benchmark report should include:

| Field | What to record |
|---|---|
| Configuration | Model, chunking, retrieval and reranking settings |
| Evaluation size | Number of questions tested |
| Quality | Available RAGAS metrics |
| Latency | Mean and p95 latency |
| Model footprint | Model size and memory usage, if measured |
| Environment | Hardware, operating system and relevant versions |

The handbook references AWQ 4-bit quantization and a before/after RAGAS comparison. If these have not been implemented and measured in the selected Ollama workflow, they must remain explicitly identified as outstanding rather than completed.

## 13. Testing and Code Quality

Run the tests from the configured project environment:

```bash
python -m pytest -q
ruff check .
```

## 14. Final Verification and Testing

The project underwent testing and verification across its core components, API interface, and development environment.

### Test Results

The following verification activities were performed during development:

| Component | Verification |
|---|---|
| Automated tests | 10 passing tests were reported during development using `pytest`. |
| API health check | The `/health` endpoint returned a healthy status during verification. |
| FastAPI Swagger UI | The API was accessed and tested through the interactive Swagger UI at `/docs`. |
| API request validation | Invalid requests were tested against the API's request-validation behavior. |
| RAG pipeline | Retrieval, generation, and citation behavior were investigated through direct testing. |
| RAGAS evaluation | A preliminary faithfulness smoke test returned a score of 1.0. |
| MLflow | Five chunking experiments were conducted and tracked during development. |
| Code quality | Ruff was included in the project's code-quality workflow. |
| Continuous integration | A GitHub Actions workflow was implemented for automated project checks. |

### API Verification

The FastAPI application exposes endpoints for health checks, metrics, and question answering. The available routes can be inspected through the automatically generated OpenAPI documentation.

To access the interactive API documentation, start the application and open:

`http://127.0.0.1:8000/docs`

The Swagger UI provides an interactive interface for submitting requests, inspecting responses, and testing API validation without requiring a separate API client.

A healthy `/health` response confirms that the service health check is operational; it does not independently establish that every downstream component is working correctly.

### Verification Scope and Limitations

The results documented here reflect testing performed during development. The reported test-suite results, health checks, experiment runs, and preliminary evaluation should be distinguished from a complete end-to-end acceptance test of the final submitted commit.

The final verification should include:

- Rerunning the complete automated test suite against the final commit.
- Confirming that `/health` and `/metrics` respond correctly.
- Submitting valid and invalid requests through Swagger UI.
- Confirming successful end-to-end retrieval and answer generation.
- Checking that returned citations correspond to retrieved legal evidence.
- Verifying that the final GitHub Actions workflow completes successfully.
- Confirming that the evaluation report, experiment artifacts, and reproducibility instructions are available to reviewers.

These checks provide a reproducible basis for assessing the project's readiness and identifying remaining issues.

---

## 15. Implementation Differences from the Project Handbook

This project follows the main objective of developing an MLOps-enabled, bilingual legal Retrieval-Augmented Generation system. Some implementation choices differ from the workflow described in the original handbook.

The following differences are documented to provide transparency and clarify the scope of the implementation.

| Area | Handbook approach | Project implementation |
|---|---|---|
| LLM serving | vLLM-based model serving | Ollama was adopted as an alternative serving approach in the development configuration. |
| Dataset preparation | Extract legal text from the provided PDF and convert it into structured JSON | The mentor-provided structured JSON dataset was used directly as the primary corpus input, avoiding a separate PDF-to-JSON extraction step in the main preprocessing workflow. |
| LLM inference | Follow the handbook's specified serving workflow | Cloud-hosted LLM APIs authenticated with API keys were used for the relevant workflows instead of relying exclusively on local model inference. |
| API testing | Verify the API and its responses | FastAPI's interactive Swagger UI was used to submit requests and inspect API behavior. |
| Continuous integration | Automate project checks through a CI workflow | GitHub Actions was implemented to automate the configured checks. Its final workflow status should be verified against the submitted commit. |
| Retrieval and evaluation | Build and evaluate a grounded legal RAG pipeline | Multilingual embeddings, ChromaDB, reranking, MLflow experiments, and preliminary RAGAS evaluation were developed as part of the project. |

### 15.1 LLM Serving: Ollama Instead of vLLM

The development configuration adopted Ollama as an alternative to the vLLM serving approach described in the handbook.

This decision changes the model-serving implementation while preserving the objective of integrating language-model inference into the RAG pipeline. Ollama and vLLM are different serving solutions, and their operational characteristics and configuration requirements are not identical.

Where cloud-hosted inference was used, API credentials and provider configuration were required. Cloud inference and local Ollama inference should be treated as distinct execution paths, and the final configuration should identify which path is active for each workflow.

### 15.2 Dataset Preparation: Direct JSON Input

The project used the structured bilingual JSON dataset supplied by the mentor as its primary input.

The preprocessing workflow focuses on validating the records, normalizing text, preserving legal article identifiers and metadata, and preparing the corpus for indexing.

Consequently, the main implementation does not depend on extracting the entire corpus from the PDF before converting it into JSON. The original PDF remains a separate reference artifact for checking the dataset's content and legal article information.

This difference concerns the source-data preparation workflow; it does not remove the need to validate the completeness and accuracy of the resulting corpus.

### 15.3 Cloud LLM APIs and Inference Configuration

Cloud-hosted LLM APIs authenticated through API keys were used for the relevant model-inference workflows.

This approach differs from relying exclusively on a locally hosted language model. It also introduces configuration and operational considerations such as API credentials, provider availability, rate limits
