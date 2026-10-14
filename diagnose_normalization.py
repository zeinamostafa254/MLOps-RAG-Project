import json
from pathlib import Path
from src.normalization import normalize_source

source = Path("data/raw/orig_data.json")
raw = json.loads(source.read_text(encoding="utf-8"))
records = normalize_source(raw)

print("\n=== RECORDS WITH EMPTY ARABIC TEXT ===")
for r in records:
    if not str(r.get("text_ar") or "").strip():
        print({
            "article_number": r.get("article_number"),
            "text_ar": repr(r.get("text_ar")),
            "text_en": repr(r.get("text_en")),
            "keys": list(r.keys()),
        })

print("\n=== REPEAL STATUS: ARTICLES 54-80 ===")
for r in records:
    number = r.get("article_number")
    if isinstance(number, int) and 54 <= number <= 80:
        print(number, "repealed =", r.get("is_repealed"),
              "text_ar =", repr(str(r.get("text_ar") or "")[:100]))

print("\n=== NORMALIZATION SOURCE ===")
print("Input type:", type(raw).__name__)
if isinstance(raw, dict):
    print("Input keys:", list(raw.keys()))
elif isinstance(raw, list) and raw:
    print("First record keys:", list(raw[0].keys()))
