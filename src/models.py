
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    article_number: int
    citation: str
    text_ar: str
    text_en: str
    metadata: dict[str, Any]

    @property
    def embedding_text(self) -> str:
        return f"passage: {self.text_ar}\n{self.text_en}"
