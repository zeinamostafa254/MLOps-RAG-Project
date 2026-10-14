
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


# Arabic-Indic digits: ٠١٢٣٤٥٦٧٨٩
# Eastern Arabic/Persian digits: ۰۱۲۳۴۵۶۷۸۹
ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

REPEALED_START = 54
REPEALED_END = 80

REPEAL_NOTICE_PATTERN = re.compile(
    r"articles?\s+54\s*[-–—]\s*80\s+have\s+been\s+repealed",
    re.IGNORECASE,
)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text while preserving its original meaning."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u0640", "")  # Remove Arabic tatweel.
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize common Arabic letter variants.
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def article_number(key: str) -> int:
    """Extract an article number from keys such as Article 12 or Article ١٢."""
    normalized_key = unicodedata.normalize("NFKC", key)
    normalized_key = normalized_key.translate(ARABIC_DIGITS)

    match = re.search(r"(\d+)", normalized_key)
    if not match:
        raise ValueError(f"Cannot parse article number from {key!r}")

    return int(match.group(1))


def normalize_source(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert the original keyed JSON into normalized article records."""
    if not isinstance(raw, dict):
        raise ValueError("Expected the source JSON to contain an object.")

    records: list[dict[str, Any]] = []

    for key, value in raw.items():
        if not key.lower().replace(" ", "").startswith("article"):
            continue

        if not isinstance(value, dict):
            raise ValueError(
                f"Expected a dictionary for {key!r}, "
                f"got {type(value).__name__}."
            )

        number = article_number(key)

        metadata = value.get("metadata", [])
        if not isinstance(metadata, list):
            metadata = [str(metadata)]

        arabic_text = normalize_arabic(value.get("arabic", ""))
        english_text = " ".join(
            str(value.get("english") or "").split()
        )

        is_repealed = REPEALED_START <= number <= REPEALED_END

        records.append(
            {
                "article_number": number,
                "book": metadata[1] if len(metadata) > 1 else "",
                "chapter": metadata[2] if len(metadata) > 2 else "",
                "section": metadata[4] if len(metadata) > 4 else "",
                "topic": metadata[-1] if metadata else "",
                "text_ar": arabic_text,
                "text_en": english_text,
                "is_repealed": is_repealed,
                "source_page": None,
                "citation": f"Egyptian Civil Code, Article {number}",
            }
        )

    records.sort(key=lambda record: record["article_number"])
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    """Validate source records without fabricating repealed article text."""
    if not records:
        raise ValueError("No article records found.")

    numbers = [record["article_number"] for record in records]

    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate article numbers found.")

    for record in records:
        number = record["article_number"]
        text_ar = str(record.get("text_ar") or "").strip()
        text_en = str(record.get("text_en") or "").strip()
        is_repealed = bool(record.get("is_repealed"))

        if len(text_ar) > 25000:
            raise ValueError(
                f"Article {number} is suspiciously large; "
                "check extraction or splitting."
            )

        if len(text_en) > 25000:
            raise ValueError(
                f"Article {number} English text is suspiciously large."
            )

        # An article with no Arabic text is acceptable only when the
        # source explicitly identifies it as repealed and provides
        # a repeal notice in English.
        if not text_ar:
            if not is_repealed:
                raise ValueError(
                    f"Article {number} has no Arabic text "
                    "and is not marked as repealed."
                )

            if not text_en:
                raise ValueError(
                    f"Article {number} has no Arabic text or "
                    "English repeal notice."
                )

            if number == REPEALED_START:
                if not REPEAL_NOTICE_PATTERN.search(text_en):
                    raise ValueError(
                        "Article 54 has no recognizable "
                        "Articles 54-80 repeal notice."
                    )

        expected_repealed = (
            REPEALED_START <= number <= REPEALED_END
        )
        if is_repealed != expected_repealed:
            raise ValueError(
                f"Incorrect repeal status for Article {number}."
            )

    # The source contains one consolidated repeal notice at Article 54.
    # Articles 55-80 need not exist as individual source records.
    if REPEALED_START not in numbers:
        raise ValueError("The consolidated Article 54 repeal notice is missing.")


def normalize_file(source: Path, output: Path) -> None:
    """Read the original JSON, normalize it, validate it, and save it."""
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
