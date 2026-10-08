
from src.ingest import chunk_article


def test_article_is_atomic_when_short():
    article = {
        "article_number": 1,
        "citation": "Egyptian Civil Code, Article 1",
        "text_ar": "نص قانوني قصير",
        "text_en": "Short legal text",
        "is_repealed": False,
    }
    chunks = chunk_article(article, 100, 10)
    assert len(chunks) == 1
    assert chunks[0].article_number == 1


def test_long_article_keeps_article_number():
    article = {
        "article_number": 999,
        "citation": "Egyptian Civil Code, Article 999",
        "text_ar": " ".join(["نص"] * 250),
        "text_en": "Long text",
        "is_repealed": False,
    }
    chunks = chunk_article(article, 100, 20)
    assert len(chunks) > 1
    assert all(c.article_number == 999 for c in chunks)
