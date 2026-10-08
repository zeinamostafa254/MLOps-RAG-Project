
import re


LEGAL_TERMS = {
    "قانون", "مادة", "عقد", "التزام", "دعوى", "محكمة", "حق", "ملكية",
    "تشريع", "مسؤولية", "تعويض", "قانون مدني", "civil", "contract", "law",
    "article", "court", "liability", "property",
}

PII_PATTERNS = [
    re.compile(r"\b\d{14}\b"),  # Egyptian national ID shape
    re.compile(r"\b01\d{9}\b"),  # common Egyptian mobile shape
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
]


def is_legal_question(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in LEGAL_TERMS)


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in PII_PATTERNS)


def validate_request(question: str) -> None:
    if contains_pii(question):
        raise ValueError("PII detected. Remove personal identifiers and retry.")
