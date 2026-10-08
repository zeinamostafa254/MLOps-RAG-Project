
import itertools
import mlflow
from src.config import get_settings
from src.ingest import build_chunks
from src.vector_store import VectorStore

settings = get_settings()
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment)

configs = [
    (800, 100),
    (1000, 100),
    (1200, 150),
    (1600, 200),
    (2000, 250),
]

for chunk_size, overlap in configs:
    with mlflow.start_run(run_name=f"chunk-{chunk_size}-{overlap}"):
        mlflow.log_params(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "embedding_model": settings.embedding_model,
            }
        )
        chunks = build_chunks("data/processed/corpus.json", chunk_size, overlap)
        mlflow.log_metric("num_chunks", len(chunks))
        # Faithfulness must come from evaluation, not a fabricated number.
        print(f"Prepared config chunk_size={chunk_size}, overlap={overlap}, chunks={len(chunks)}")
