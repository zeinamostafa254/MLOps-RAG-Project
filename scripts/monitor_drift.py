
"""
Query-embedding drift monitor.

Compare the current batch mean query embedding against a baseline embedding
centroid. Store the score in Prometheus/MLflow in a production deployment.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_settings


def cosine_drift(baseline: np.ndarray, current: np.ndarray) -> float:
    similarity = cosine_similarity(
        baseline.reshape(1, -1),
        current.reshape(1, -1),
    )[0, 0]
    return float(1.0 - similarity)


def main():
    settings = get_settings()
    encoder = SentenceTransformer(settings.embedding_model)

    questions = json.loads(
        Path("evaluation/questions.json").read_text(encoding="utf-8")
    )
    embeddings = encoder.encode(
        [q["question"] for q in questions],
        normalize_embeddings=True,
    )
    baseline = embeddings.mean(axis=0)
    current = embeddings[:10].mean(axis=0)

    score = cosine_drift(baseline, current)
    print(f"cosine drift={score:.6f}")


if __name__ == "__main__":
    main()
