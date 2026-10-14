import json
from pathlib import Path

import pytest

from src.normalization import normalize_source, validate_records


SOURCE = Path("data/raw/orig_data.json")


@pytest.mark.skipif(
    not SOURCE.exists(),
    reason="Copy supplied orig_data.json into data/raw/",
)
def test_normalization_has_articles():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = normalize_source(raw)

    assert records
    assert len(records) == len(
        {record["article_number"] for record in records}
    )


@pytest.mark.skipif(
    not SOURCE.exists(),
    reason="Copy supplied orig_data.json into data/raw/",
)
def test_repealed_articles_are_flagged():
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = normalize_source(raw)

    article_54 = next(
        record for record in records
        if record["article_number"] == 54
    )

    assert article_54["is_repealed"] is True
    assert "Articles 54-80 have been repealed" in (
        article_54["text_en"].replace("\\n", " ")
    )

    # The source contains a consolidated repeal notice, not
    # 27 individual article records.
    assert not any(
        55 <= record["article_number"] <= 80
        for record in records
    )

    validate_records(records)
