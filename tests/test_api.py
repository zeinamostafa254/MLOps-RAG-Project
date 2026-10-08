
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_empty_question_rejected():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_health_contract():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "documents_indexed" in body
