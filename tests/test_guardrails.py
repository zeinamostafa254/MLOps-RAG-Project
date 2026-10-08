
import pytest
from src.guardrails import contains_pii, is_legal_question, validate_request


def test_legal_question():
    assert is_legal_question("ما هو أثر العقد؟")


def test_non_legal_question():
    assert not is_legal_question("ما هي عاصمة فرنسا؟")


def test_pii():
    assert contains_pii("رقمي 01012345678")


def test_pii_is_blocked():
    with pytest.raises(ValueError):
        validate_request("العقد ورقم الهاتف 01012345678")
