
import mlflow
from .config import get_settings


def log_rag_run(
    *,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    faithfulness: float,
    answer_relevancy: float | None = None,
    context_precision: float | None = None,
    context_recall: float | None = None,
) -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "embedding_model": embedding_model,
            }
        )
        metrics = {"faithfulness": faithfulness}
        if answer_relevancy is not None:
            metrics["answer_relevancy"] = answer_relevancy
        if context_precision is not None:
            metrics["context_precision"] = context_precision
        if context_recall is not None:
            metrics["context_recall"] = context_recall
        mlflow.log_metrics(metrics)


def register_best_config() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "chunk_size": settings.chunk_size,
                "overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
            }
        )
        mlflow.log_metric("faithfulness", 0.0)
        mlflow.log_text(
            "RAG configuration. Replace the placeholder faithfulness with the measured best run.",
            "config.txt",
        )
        model_uri = f"runs:/{run.info.run_id}/config.txt"
        try:
            mlflow.register_model(model_uri, settings.mlflow_model_name)
        except Exception as exc:
            print(f"Registry registration skipped: {exc}")
