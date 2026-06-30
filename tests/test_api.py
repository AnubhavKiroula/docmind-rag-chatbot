import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import FastAPI app and database components
from main import app
from db.database import Base, get_db
from db.models import Conversation, Message

# 1. Setup in-memory SQLite database using StaticPool for thread-safe test isolation
TEST_DATABASE_URL = "sqlite://"
test_engine = create_engine(
    TEST_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create all tables in the shared in-memory database
Base.metadata.create_all(bind=test_engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Override FastAPI's get_db dependency with our test database generator
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_db():
    """
    Cleans database tables before and after each individual test run.
    """
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield

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
    Test POST /query handles requests and saves queries/answers to the SQLite database.
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
    
    # Verify that the session is stored in the database
    db = TestingSessionLocal()
    try:
        db_conversation = db.query(Conversation).filter(Conversation.id == "test-session-123").first()
        assert db_conversation is not None
        assert len(db_conversation.messages) == 2
        
        # User message assertion
        assert db_conversation.messages[0].role == "user"
        assert db_conversation.messages[0].content == "What is the mock question?"
        
        # Assistant response assertion
        assert db_conversation.messages[1].role == "assistant"
        assert db_conversation.messages[1].content == "This is a mock answer based on the document."
        assert db_conversation.messages[1].confidence_score == 0.95
    finally:
        db.close()

def test_history_endpoint():
    """
    Test GET /conversations/{conversation_id} and GET /history/{conversation_id}
    returns history of existing session and returns 404 for non-existent session.
    """
    # Pre-populate the test DB with a conversation session
    db = TestingSessionLocal()
    try:
        conv = Conversation(id="test-session-abc")
        db.add(conv)
        db.commit()
        
        msg = Message(
            conversation_id="test-session-abc",
            role="user",
            content="Hello"
        )
        db.add(msg)
        db.commit()
    finally:
        db.close()
    
    # Test valid session via history endpoint
    response = client.get("/conversations/test-session-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "test-session-abc"
    assert len(data["history"]) == 1
    assert data["history"][0]["content"] == "Hello"
    assert data["history"][0]["role"] == "user"
    
    # Test legacy alias route
    response_legacy = client.get("/history/test-session-abc")
    assert response_legacy.status_code == 200
    assert response_legacy.json() == response.json()
    
    # Test invalid session
    response_invalid = client.get("/conversations/invalid-session-xyz")
    assert response_invalid.status_code == 404
    assert response_invalid.json()["detail"] == "Conversation session not found."
