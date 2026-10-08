import mlflow
import json
from pathlib import Path

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("legal-rag")

Path("mlflow_test_artifact.json").write_text(
    json.dumps({
        "test": "MLflow is working",
        "project": "Legal RAG"
    }, indent=2),
    encoding="utf-8"
)

with mlflow.start_run(run_name="mlflow-test"):

    mlflow.log_params({
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "embedding_model": "intfloat/multilingual-e5-base",
    })

    mlflow.log_metrics({
        "faithfulness": 0.80,
        "latency_seconds": 2.5,
    })

    mlflow.log_artifact("mlflow_test_artifact.json")

    print("Run ID:", mlflow.active_run().info.run_id)