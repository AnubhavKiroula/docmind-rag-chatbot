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
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool
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


@patch("api.routes.get_retriever")
@patch("api.routes.get_synthesizer")
def test_query_endpoint(mock_get_synthesizer, mock_get_retriever):
    """
    Test POST /query handles requests and saves queries/answers to the SQLite database.
    """
    # Mock retriever output: list of (chunk_text, score)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [("Mock document chunk", 0.95)]
    mock_get_retriever.return_value = mock_retriever

    # Mock synthesizer output: dict
    mock_synthesizer = MagicMock()
    mock_synthesizer.synthesize.return_value = {
        "answer": "This is a mock answer based on the document.",
        "confidence_score": 0.95,
        "sources": ["Mock document chunk"],
    }
    mock_get_synthesizer.return_value = mock_synthesizer

    payload = {"query": "What is the mock question?", "conversation_id": "test-session-123"}

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
        db_conversation = (
            db.query(Conversation).filter(Conversation.id == "test-session-123").first()
        )
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

        msg = Message(conversation_id="test-session-abc", role="user", content="Hello")
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


@patch("api.routes.get_retriever")
@patch("api.routes.get_synthesizer")
def test_query_endpoint_with_top_k(mock_get_synthesizer, mock_get_retriever):
    """
    Test POST /query with an explicit top_k parameter passes the value to the retriever.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [("Mock chunk", 0.9)]
    mock_get_retriever.return_value = mock_retriever

    mock_synthesizer = MagicMock()
    mock_synthesizer.synthesize.return_value = {
        "answer": "Mock answer",
        "confidence_score": 0.9,
        "sources": ["Mock chunk"],
    }
    mock_get_synthesizer.return_value = mock_synthesizer

    payload = {"query": "Hello", "top_k": 5}

    response = client.post("/query", json=payload)
    assert response.status_code == 200

    # Assert retriever.retrieve was called with top_k=5
    mock_retriever.retrieve.assert_called_once_with("Hello", top_k=5)


def test_ingest_rejects_non_pdf():
    """
    Test POST /ingest returns 400 for a non-PDF upload.
    """
    response = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_ingest_rejects_oversized_file(tmp_path):
    """
    Test POST /ingest returns 413 when the upload exceeds MAX_UPLOAD_BYTES,
    and that no partial file is left behind.
    """
    with patch("api.routes.UPLOAD_DIR", str(tmp_path)), patch("api.routes.MAX_UPLOAD_BYTES", 10):
        response = client.post(
            "/ingest", files={"file": ("big.pdf", b"x" * 100, "application/pdf")}
        )
    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_ingest_filename_cannot_escape_upload_dir(tmp_path):
    """
    Test that a traversal filename is reduced to its basename and saved inside UPLOAD_DIR.
    The pipeline itself fails on the fake PDF (expected 500). We disable the route's failure
    cleanup (os.remove) so the saved file survives and its location can be asserted.
    """
    upload_dir = tmp_path / "uploads"
    with patch("api.routes.UPLOAD_DIR", str(upload_dir)), patch("api.routes.os.remove"):
        client.post(
            "/ingest",
            files={"file": ("../../evil.pdf", b"%PDF-1.4 not a real pdf", "application/pdf")},
        )
    assert (upload_dir / "evil.pdf").exists()
    assert not (tmp_path.parent / "evil.pdf").exists()
