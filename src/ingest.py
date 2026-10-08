
import json
from pathlib import Path
from .models import LegalChunk


def load_articles(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["records"]


def chunk_article(article: dict, chunk_size: int, overlap: int) -> list[LegalChunk]:
    text = article["text_ar"]
    words = text.split()

    if len(words) <= chunk_size:
        return [
            LegalChunk(
                chunk_id=f"article-{article['article_number']}-0",
                article_number=article["article_number"],
                citation=article["citation"],
                text_ar=text,
                text_en=article["text_en"],
                metadata=article,
            )
        ]

    chunks = []
    start = 0
    part = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        ar = " ".join(words[start:end])
        chunks.append(
            LegalChunk(
                chunk_id=f"article-{article['article_number']}-{part}",
                article_number=article["article_number"],
                citation=article["citation"],
                text_ar=ar,
                text_en=article["text_en"],
                metadata=article,
            )
        )
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
        part += 1

    return chunks


def build_chunks(path: str | Path, chunk_size: int, overlap: int) -> list[LegalChunk]:
    chunks = []
    for article in load_articles(path):
        chunks.extend(chunk_article(article, chunk_size, overlap))
    return chunks
