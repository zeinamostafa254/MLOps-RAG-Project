
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class Source(BaseModel):
    citation: str
    article_number: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    source_details: list[Source]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
