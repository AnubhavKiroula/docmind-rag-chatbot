import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app
from main import app
from api.routes import conversation_store

client = TestClient(app)

def test_health_endpoint():
    """
    Test GET /health returns 200 and status ok.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data

def test_root_endpoint():
    """
    Test GET / returns welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Welcome to DocMind" in data["message"]

@patch("api.routes.retriever.retrieve")
@patch("api.routes.synthesizer.synthesize")
def test_query_endpoint(mock_synthesize, mock_retrieve):
    """
    Test POST /query handles requests and coordinates retriever and synthesizer.
    """
    # Mock retriever output: list of (chunk_text, score)
    mock_retrieve.return_value = [("Mock document chunk", 0.95)]
    
    # Mock synthesizer output: dict
    mock_synthesize.return_value = {
        "answer": "This is a mock answer based on the document.",
        "confidence_score": 0.95,
        "sources": ["Mock document chunk"]
    }
    
    payload = {
        "query": "What is the mock question?",
        "conversation_id": "test-session-123"
    }
    
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["answer"] == "This is a mock answer based on the document."
    assert data["confidence_score"] == 0.95
    assert len(data["sources"]) == 1
    assert data["sources"][0]["text"] == "Mock document chunk"
    assert data["sources"][0]["relevance_score"] == 0.95
    assert data["conversation_id"] == "test-session-123"
    
    # Verify that the session is stored in conversation history
    assert "test-session-123" in conversation_store
    history = conversation_store["test-session-123"]
    assert len(history) == 1
    assert history[0]["query"] == "What is the mock question?"
    assert history[0]["answer"] == "This is a mock answer based on the document."

def test_history_endpoint():
    """
    Test GET /history/{conversation_id} returns history of existing session
    and returns 404 for non-existent session.
    """
    # Add dummy history to the store
    conversation_store["test-session-abc"] = [
        {
            "query": "Hello",
            "answer": "Hi there",
            "timestamp": "2026-06-30T10:00:00"
        }
    ]
    
    # Test valid session
    response = client.get("/history/test-session-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "test-session-abc"
    assert len(data["history"]) == 1
    assert data["history"][0]["query"] == "Hello"
    
    # Test invalid session
    response = client.get("/history/invalid-session-xyz")
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation session not found."
