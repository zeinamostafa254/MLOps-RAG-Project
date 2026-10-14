from pathlib import Path
from src.normalization import normalize_file

source = Path("data/raw/orig_data.json")
output = Path("data/processed/corpus.json")

normalize_file(source, output)
print(f"Normalization successful: {output}")
