
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u0640", "")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def article_number(key: str) -> int:
    match = re.search(r"(\d+|[٠-٩]+)", key)
    if not match:
        raise ValueError(f"Cannot parse article number from {key!r}")
    return int(match.group(1).translate(ARABIC_DIGITS))


def normalize_source(raw: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, value in raw.items():
        if not key.lower().startswith("article "):
            continue
        number = article_number(key)
        metadata = value.get("metadata", [])
        if not isinstance(metadata, list):
            metadata = [str(metadata)]

        records.append(
            {
                "article_number": number,
                "book": metadata[1] if len(metadata) > 1 else "",
                "chapter": metadata[2] if len(metadata) > 2 else "",
                "section": metadata[4] if len(metadata) > 4 else "",
                "topic": metadata[-1] if metadata else "",
                "text_ar": normalize_arabic(value.get("arabic", "")),
                "text_en": " ".join((value.get("english") or "").split()),
                "is_repealed": number in range(54, 81),
                "source_page": None,
                "citation": f"Egyptian Civil Code, Article {number}",
            }
        )

    records.sort(key=lambda x: x["article_number"])
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No article records found")
    if any(not r["text_ar"] for r in records):
        raise ValueError("Every article must have non-empty Arabic text")
    if any(len(r["text_ar"]) > 25000 for r in records):
        raise ValueError("An article is suspiciously large; extraction/splitting likely failed")

    numbers = [r["article_number"] for r in records]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate article numbers found")

    repealed = [r["article_number"] for r in records if r["is_repealed"]]
    if repealed != list(range(54, 81)):
        raise ValueError("Expected Articles 54–80 to be flagged as repealed")


def normalize_file(source: Path, output: Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    records = normalize_source(raw)
    validate_records(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "corpus_name": "Egyptian Civil Code",
                "language": ["ar", "en"],
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
