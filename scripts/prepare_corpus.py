
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from normalization import normalize_file

source = Path("data/raw/orig_data.json")
output = Path("data/processed/corpus.json")

if not source.exists():
    raise SystemExit("Missing data/raw/orig_data.json. Copy the supplied JSON there.")

normalize_file(source, output)
print(f"Wrote {output}")
