
"""
Benchmark harness.

Populate actual measurements by running this script against your chosen
generator/retriever configurations. It intentionally never fabricates results.
"""

import csv
import time
from pathlib import Path


def write_row(name: str, latency_ms: float, quality: float, size_mb: float) -> None:
    path = Path("reports/benchmark.csv")
    path.parent.mkdir(exist_ok=True)

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["configuration", "p95_latency_ms", "faithfulness", "size_mb"])
        writer.writerow([name, latency_ms, quality, size_mb])


if __name__ == "__main__":
    print(
        "Run actual baseline/AWQ/reranker measurements and call write_row(). "
        "Do not enter invented values."
    )
