
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import get_settings
from ingest import build_chunks
from vector_store import VectorStore

settings = get_settings()
chunks = build_chunks(
    "data/processed/corpus.json",
    settings.chunk_size,
    settings.chunk_overlap,
)

store = VectorStore()
store.index(chunks)

print(f"Indexed {len(chunks)} chunks; collection count={store.count()}")
