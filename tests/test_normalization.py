
import json
from pathlib import Path
import pytest

from src.normalization import normalize_source, validate_records

SOURCE = Path("data/raw/orig_data.json")


@pytest.mark.skipif(not SOURCE.exists(), reason="copy supplied orig_data.json into data/raw/")
def test_normalization_has_articles():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = normalize_source(raw)
    validate_records(records)
    assert records[0]["article_number"] == 1


@pytest.mark.skipif(not SOURCE.exists(), reason="copy supplied orig_data.json into data/raw/")
def test_repealed_articles_are_flagged():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = normalize_source(raw)
    repealed = [r["article_number"] for r in records if r["is_repealed"]]
    assert repealed == list(range(54, 81))
