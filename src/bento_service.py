
import bentoml
from .rag import LegalRAG


@bentoml.service(name="egyptian_legal_rag")
class LegalRAGService:
    def __init__(self) -> None:
        self.rag = LegalRAG()

    @bentoml.api
    async def ask(self, question: str) -> dict:
        return await self.rag.ask(question)
