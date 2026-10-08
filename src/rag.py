
from .config import get_settings
from .guardrails import is_legal_question, validate_request
from .llm import LLMClient
from .reranker import Reranker
from .vector_store import VectorStore


class LegalRAG:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = VectorStore()
        self.reranker = Reranker()
        self.llm = LLMClient()

    async def ask(self, question: str) -> dict:
        validate_request(question)
        if not is_legal_question(question):
            raise ValueError("Only legal-document questions are supported.")

        hits = self.store.search(question, self.settings.top_k * 2)
        hits = self.reranker.rerank(question, hits, self.settings.top_k)
        answer = await self.llm.generate(question, hits)

        sources = []
        for hit in hits:
            citation = hit["metadata"]["citation"]
            if citation not in sources:
                sources.append(citation)

        return {
            "answer": answer,
            "sources": sources,
            "source_details": [
                {
                    "citation": h["metadata"]["citation"],
                    "article_number": int(h["metadata"]["article_number"]),
                    "score": float(h["score"]),
                    "text": h["metadata"]["text_ar"],
                }
                for h in hits
            ],
            "model_version": self.settings.vllm_model,
        }
