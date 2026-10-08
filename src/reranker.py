
from sentence_transformers import CrossEncoder
from .config import get_settings


class Reranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(get_settings().reranker_model)

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits:
            return []
        pairs = [(query, hit["metadata"]["text_ar"]) for hit in hits]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(hits, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            {**hit, "reranker_score": float(score)}
            for hit, score in ranked[:top_k]
        ]
