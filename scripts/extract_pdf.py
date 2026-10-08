
"""
Reference PDF extractor.

The handbook's ideal pipeline is PDF -> article JSON. The supplied GitHub JSON is
already extracted, so use this script to create a candidate extraction and compare
it against data/processed/corpus.json before treating it as authoritative.

This intentionally avoids silently claiming perfect extraction: bilingual legal PDFs
can interleave Arabic/English columns and Arabic glyph order.
"""

from pathlib import Path
import json
import re
import fitz


ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"


def to_western(value: str) -> int:
    trans = str.maketrans(ARABIC_INDIC, "0123456789")
    return int(value.translate(trans))


def reverse_arabic_line(line: str) -> str:
    # The supplied PDF's extracted Arabic lines are commonly word-order reversed.
    return " ".join(reversed(line.split()))


def extract(pdf_path: Path) -> dict:
    articles = {}
    current = None

    for page_number, page in enumerate(fitz.open(pdf_path), start=1):
        lines = page.get_text("text").splitlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            english = re.match(r"Article\s+(\d+)\s*$", line, re.I)
            arabic = re.search(r"مادة\s*[\(\s]*([٠-٩]+)", line)

            if english:
                current = int(english.group(1))
                articles.setdefault(current, {"arabic": [], "english": [], "source_page": page_number})
                continue

            if current is None:
                continue

            if arabic:
                continue

            # Heuristic: Arabic Unicode range => Arabic candidate; otherwise English.
            if re.search(r"[\u0600-\u06ff]", line):
                articles[current]["arabic"].append(reverse_arabic_line(line))
            elif re.search(r"[A-Za-z]", line):
                articles[current]["english"].append(line)

    return {
        "articles": [
            {
                "article_number": number,
                "text_ar": " ".join(value["arabic"]).strip(),
                "text_en": " ".join(value["english"]).strip(),
                "source_page": value["source_page"],
            }
            for number, value in sorted(articles.items())
        ]
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/processed/pdf_candidate.json"))
    args = parser.parse_args()

    result = extract(args.pdf)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
