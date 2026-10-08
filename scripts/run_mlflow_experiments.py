# made by ai for testing
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import chromadb
import mlflow
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

  
# 1. LOAD ENVIRONMENT
  

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "orig_data.json"
INDEX_ROOT = PROJECT_ROOT / "data" / "mlflow_indexes"
REPORT_ROOT = PROJECT_ROOT / "reports" / "mlflow"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-base",
)

TOP_K = int(os.getenv("TOP_K", "8"))

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "legal-rag",
)

MLFLOW_MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "LegalRAGConfig",
)


  
# 2. EXPERIMENT CONFIGURATIONS
  

EXPERIMENTS = [
    {
        "name": "run-01-small",
        "chunk_size": 800,
        "chunk_overlap": 100,
    },
    {
        "name": "run-02-medium-small",
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },
    {
        "name": "run-03-baseline",
        "chunk_size": 1200,
        "chunk_overlap": 150,
    },
    {
        "name": "run-04-large",
        "chunk_size": 1500,
        "chunk_overlap": 200,
    },
    {
        "name": "run-05-very-large",
        "chunk_size": 1800,
        "chunk_overlap": 250,
    },
]


  
# 3. REAL LEGAL QUESTIONS
  

QUESTIONS = [
    {
        "question": "ما هي شروط صحة العقد؟",
        "expected_article": None,
    },
    {
        "question": "ما هو تعريف العقد في القانون المدني؟",
        "expected_article": None,
    },
    {
        "question": "ما هي آثار العقد بين المتعاقدين؟",
        "expected_article": None,
    },
    {
        "question": "متى يكون العقد باطلاً؟",
        "expected_article": None,
    },
    {
        "question": "ما هي شروط الأهلية للتعاقد؟",
        "expected_article": None,
    },
    {
        "question": "ما هي القواعد المتعلقة بالالتزام؟",
        "expected_article": None,
    },
    {
        "question": "متى ينتهي الالتزام؟",
        "expected_article": None,
    },
    {
        "question": "ما هي أحكام التعويض عن الضرر؟",
        "expected_article": None,
    },
    {
        "question": "ما هي شروط المسؤولية عن الفعل الضار؟",
        "expected_article": None,
    },
    {
        "question": "ما هي أحكام الملكية؟",
        "expected_article": None,
    },
]


  
# 4. LOAD JSON
  

def load_articles() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find legal corpus:\n{DATA_PATH}"
        )

    with DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    articles = []

    # Mentor JSON is expected to look approximately like:
    #
    # {
    #   "Article 1": {
    #       "arabic": "...",
    #       "english": "...",
    #       "metadata": [...]
    #   }
    # }

    for key, value in data.items():

        if not isinstance(value, dict):
            continue

        arabic = (
            value.get("arabic")
            or value.get("text_ar")
            or value.get("ar_text")
            or ""
        )

        english = (
            value.get("english")
            or value.get("text_en")
            or ""
        )

        if not arabic:
            continue

        article_number = key

        if key.lower().startswith("article"):
            article_number = key.split()[-1]

        try:
            article_number = int(article_number)
        except ValueError:
            pass

        articles.append(
            {
                "article_number": article_number,
                "text_ar": arabic,
                "text_en": english,
                "metadata": value.get("metadata", []),
            }
        )

    if not articles:
        raise ValueError(
            "No articles were found in orig_data.json. "
            "Check the JSON structure."
        )

    return articles


  
# 5. ARTICLE-LEVEL CHUNKING
  

def chunk_article(
    article: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:

    text = article["text_ar"].strip()

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    chunk_number = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append(
                {
                    "id": (
                        f"article-{article['article_number']}"
                        f"-chunk-{chunk_number}"
                    ),
                    "text": chunk_text,
                    "article_number": article["article_number"],
                }
            )

        if end >= len(text):
            break

        start = end - chunk_overlap
        chunk_number += 1

    return chunks


def build_chunks(
    articles: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:

    all_chunks = []

    for article in articles:
        all_chunks.extend(
            chunk_article(
                article,
                chunk_size,
                chunk_overlap,
            )
        )

    return all_chunks


  
# 6. EMBEDDINGS
  

print(
    f"Loading embedding model: {EMBEDDING_MODEL}"
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


def embed_documents(
    texts: list[str],
) -> list[list[float]]:

    prepared = [
        f"passage: {text}"
        for text in texts
    ]

    embeddings = embedding_model.encode(
        prepared,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()


def embed_query(
    question: str,
) -> list[float]:

    embedding = embedding_model.encode(
        [f"query: {question}"],
        normalize_embeddings=True,
    )

    return embedding[0].tolist()


  
# 7. CHROMA
  

def create_vector_store(
    chunks: list[dict[str, Any]],
    index_path: Path,
) -> chromadb.Collection:

    if index_path.exists():
        shutil.rmtree(index_path)

    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(index_path)
    )

    collection = client.get_or_create_collection(
        name="legal_rag_experiment",
        metadata={
            "hnsw:space": "cosine"
        },
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_documents(texts)

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "article_number": str(
                chunk["article_number"]
            )
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


 
# 8. OLLAMA
 

llm_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

# RAGAS evaluator using the same local Ollama model
ragas_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

ragas_llm = llm_factory(
    OLLAMA_MODEL,
    client=ragas_client,
)

faithfulness_metric = Faithfulness(
    llm=ragas_llm
)
# TEMPORARY RAGAS TEST
test_score = faithfulness_metric.score(
    user_input="ما هي شروط صحة العقد؟",
    response="اختبار تقييم الإجابة.",
    retrieved_contexts=[
        "هذا نص قانوني تجريبي لاختبار RAGAS."
    ],
)

print("RAGAS TEST SCORE:", test_score.value)

def generate_answer(
    question: str,
    contexts: list[str],
) -> str:

    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
You are an Egyptian legal document assistant.

Answer the user's question ONLY using the provided
Egyptian Civil Code context.

If the context does not contain enough information,
say that the available documents do not provide
enough information.

Do not invent legal rules.

Question:
{question}

Context:
{context_text}

Give a concise answer and mention the relevant
article number when it is available.
"""

    response = llm_client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual legal RAG assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content


  
# 9. RETRIEVE + GENERATE
  

def answer_question(
    collection: chromadb.Collection,
    question: str,
) -> dict[str, Any]:

    query_embedding = embed_query(question)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    answer = generate_answer(
        question,
        documents,
    )

    articles = [
        metadata["article_number"]
        for metadata in metadatas
    ]

    return {
        "question": question,
        "answer": answer,
        "contexts": documents,
        "articles": articles,
    }


  
# 10. SIMPLE RETRIEVAL METRIC
  

def calculate_retrieval_coverage(
    results: list[dict[str, Any]],
) -> float:

    successful = 0

    for result in results:

        if result["contexts"]:
            successful += 1

    return successful / len(results)

def calculate_faithfulness(
    results: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """
    Calculate RAGAS faithfulness for every generated answer.

    Faithfulness asks:
    Is the generated answer supported by the retrieved legal context?
    """

    scores = []

    for i, result in enumerate(results, start=1):
        print()
        print(f"Evaluating faithfulness {i}/{len(results)}...")

        try:
            score = faithfulness_metric.score(
                user_input=result["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
            )

            value = float(score.value)

            scores.append(value)

            print(f"Faithfulness: {value:.4f}")

        except Exception as exc:
            raise RuntimeError(
                f"RAGAS faithfulness failed for question {i}: "
                f"{result['question']}"
            ) from exc

    if not scores:
        raise RuntimeError(
            "RAGAS faithfulness evaluation produced no valid scores."
        )

    average_score = sum(scores) / len(scores)

    return average_score, scores

  
# 11. MAIN MLflow EXPERIMENT
  

def main():

    mlflow.set_tracking_uri(
        MLFLOW_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    articles = load_articles()

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loaded {len(articles)} legal articles."
    )

    all_run_results = []

    for config in EXPERIMENTS:

        print()
        print("=" * 70)
        print(
            f"STARTING {config['name']}"
        )
        print("=" * 70)

        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]

        index_path = (
            INDEX_ROOT
            / config["name"]
        )

        start_time = time.perf_counter()

        with mlflow.start_run(
            run_name=config["name"]
        ) as run:

             
            # PARAMETERS
             

            mlflow.log_params(
                {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": EMBEDDING_MODEL,
                    "reranker_model": os.getenv(
                        "RERANKER_MODEL",
                        "BAAI/bge-reranker-v2-m3",
                    ),
                    "top_k": TOP_K,
                    "llm_model": OLLAMA_MODEL,
                    "llm_provider": "ollama",
                    "dataset": "Egyptian Civil Code",
                }
            )

             
            # BUILD INDEX
             

            chunks = build_chunks(
                articles,
                chunk_size,
                chunk_overlap,
            )

            mlflow.log_metric(
                "num_articles",
                len(articles),
            )

            mlflow.log_metric(
                "num_chunks",
                len(chunks),
            )

            print(
                f"Created {len(chunks)} chunks."
            )

            collection = create_vector_store(
                chunks,
                index_path,
            )

             
            # REAL QUESTIONS
             

            results = []

            for item in QUESTIONS:

                result = answer_question(
                    collection,
                    item["question"],
                )

                results.append(result)

                print()
                print(
                    "QUESTION:",
                    item["question"],
                )

                print(
                    "ARTICLES:",
                    result["articles"],
                )

                print(
                    "ANSWER:",
                    result["answer"][:300],
                )

             
            # METRICS
           

            retrieval_coverage = calculate_retrieval_coverage(
                results
            )

# RAGAS faithfulness
            faithfulness, individual_faithfulness = calculate_faithfulness(
                results
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            # Log standard metrics
            mlflow.log_metric(
                "retrieval_coverage",
                retrieval_coverage,
            )

            mlflow.log_metric(
                "faithfulness",
                faithfulness,
            )

            mlflow.log_metric(
                "experiment_runtime_seconds",
                elapsed,
            )

            # Also log latency as the total experiment runtime
            mlflow.log_metric(
                "latency_seconds",
                elapsed,
            )

            print()
            print(f"Retrieval coverage: {retrieval_coverage:.4f}")
            print(f"Faithfulness:       {faithfulness:.4f}")
            print(f"Runtime:            {elapsed:.2f} seconds")

            
             
            # SAVE RESULTS
             

            result_path = (
                REPORT_ROOT
                / f"{config['name']}_results.json"
            )

            with result_path.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    results,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            mlflow.log_artifact(
                str(result_path),
                artifact_path="evaluation",
            )

            # Save experiment configuration

            config_path = (
                REPORT_ROOT
                / f"{config['name']}_config.json"
            )

            with config_path.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    config,
                    f,
                    indent=2,
                )

            mlflow.log_artifact(
                str(config_path),
                artifact_path="config",
            )

            mlflow.set_tags(
                {
                    "project": "egyptian-legal-rag",
                    "experiment_type": (
                        "chunking_ablation"
                    ),
                    "track": "LLM-RAG",
                }
            )

            all_run_results.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": config["name"],
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "retrieval_coverage": retrieval_coverage,
                    "faithfulness": faithfulness,
                    "experiment_runtime_seconds": elapsed,
                }
)

        print(
            f"FINISHED {config['name']}"
        )

     
    # SAVE SUMMARY
     

    summary_path = (
        REPORT_ROOT
        / "experiment_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_run_results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("ALL 5 MLFLOW RUNS FINISHED")
    print("=" * 70)

    for result in all_run_results:

        print(
            result["run_name"],
            "->",
            result["retrieval_coverage"],
        )

ragas_results = []

for result, score in zip(
    results,
    individual_faithfulness
):
    ragas_results.append(
        {
            "question": result["question"],
            "answer": result["answer"],
            "retrieved_articles": result["articles"],
            "faithfulness": score,
        }
    )

ragas_path = (
    REPORT_ROOT
    / f"{config['name']}_ragas.json"
)

with ragas_path.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        ragas_results,
        f,
        ensure_ascii=False,
        indent=2,
    )

mlflow.log_artifact(
    str(ragas_path),
    artifact_path="ragas",
)

if __name__ == "__main__":
    main()