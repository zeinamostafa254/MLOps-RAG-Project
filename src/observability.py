
import time
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from .config import get_settings

REQUESTS = Counter("rag_requests_total", "Total RAG requests")
LATENCY = Histogram("rag_request_latency_seconds", "RAG request latency")
TOKENS = Counter("rag_tokens_total", "Estimated generation tokens")
FAITHFULNESS = Gauge("rag_faithfulness", "Latest RAGAS faithfulness")
QUERY_DRIFT = Gauge("rag_query_cosine_drift", "Query embedding cosine drift")


@asynccontextmanager
async def observe_request():
    start = time.perf_counter()
    REQUESTS.inc()
    try:
        yield
    finally:
        LATENCY.observe(time.perf_counter() - start)


def metrics_response():
    return generate_latest()
