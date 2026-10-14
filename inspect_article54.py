import json
from pathlib import Path

data = json.loads(
    Path("data/raw/orig_data.json").read_text(encoding="utf-8")
)

print("=== RAW ARTICLE 54 ===")
print(json.dumps(data.get("Article 54"), ensure_ascii=False, indent=2))

print("\n=== METADATA ===")
print(json.dumps(data.get("__metadata__"), ensure_ascii=False, indent=2))

print("\n=== UNMATCHED TEXT ===")
print(json.dumps(data.get("__unmatched_text__"), ensure_ascii=False, indent=2)[:5000])
