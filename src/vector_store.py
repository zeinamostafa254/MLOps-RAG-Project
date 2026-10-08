
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from .config import get_settings
from .models import LegalChunk


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.encoder = SentenceTransformer(settings.embedding_model)

    def count(self) -> int:
        return self.collection.count()

    def index(self, chunks: list[LegalChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index")

        ids = [c.chunk_id for c in chunks]
        texts = [c.embedding_text for c in chunks]
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        metadatas = [
            {
                "article_number": c.article_number,
                "citation": c.citation,
                "text_ar": c.text_ar,
                "text_en": c.text_en,
                "is_repealed": bool(c.metadata.get("is_repealed", False)),
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int) -> list[dict]:
        q = self.encoder.encode([f"query: {query}"], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=q,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "score": 1.0 - float(distance),
                }
            )
        return hits
