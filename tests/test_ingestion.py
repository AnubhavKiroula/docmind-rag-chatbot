import os
import pytest
from unittest.mock import MagicMock, patch

# Import LlamaIndex components needed for verification and mocking
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import MockEmbedding

# Import the functions we want to test
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model
from ingestion.store import build_vector_index

@pytest.fixture
def temp_pdf_dir(tmp_path):
    """
    Fixture to create a temporary directory containing a valid minimal PDF file.
    """
    pdf_dir = tmp_path / "sample"
    pdf_dir.mkdir()
    pdf_file = pdf_dir / "test.pdf"
    
    # A basic minimal valid PDF file structure (binary representation)
    # containing the text "Hello World"
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 48 >>\nstream\nBT\n/F1 12 Tf\n72 712 Td\n(Hello World) Tj\nET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000111 00000 n\n0000000212 00000 n\n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n311\n%%EOF"
    )
    pdf_file.write_bytes(minimal_pdf)
    return str(pdf_dir)

def test_load_documents(temp_pdf_dir):
    """
    Verifies that load_documents properly reads PDF files in a directory 
    and returns LlamaIndex Document objects.
    """
    docs = load_documents(temp_pdf_dir)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    # Verify that the parsed content contains our dummy string
    assert "Hello World" in docs[0].text

def test_chunk_documents():
    """
    Verifies that chunk_documents correctly splits a list of documents
    into smaller overlapping text nodes.
    """
    docs = [
        Document(text="This is a test document to verify the chunker. It has some text contents that need to be split into nodes.")
    ]
    # Set chunk_size small to force splitting the document
    nodes = chunk_documents(docs, chunk_size=15, chunk_overlap=3)
    assert len(nodes) > 1
    # Check that they are LlamaIndex TextNodes
    assert all(isinstance(node, TextNode) for node in nodes)

def test_get_embedding_model():
    """
    Verifies that get_embedding_model initializes the HuggingFaceEmbedding class.
    We patch it to avoid downloading heavy weights during the unit test.
    """
    with patch("ingestion.embedder.HuggingFaceEmbedding") as mock_hf:
        mock_hf.return_value = "mock_model"
        model = get_embedding_model("dummy_model_name")
        mock_hf.assert_called_once_with(model_name="dummy_model_name")
        assert model == "mock_model"

def test_build_vector_index():
    """
    Verifies that build_vector_index properly connects to Qdrant, registers
    the collection, and constructs a VectorStoreIndex.
    """
    nodes = [TextNode(text="chunk 1"), TextNode(text="chunk 2")]
    # Using LlamaIndex MockEmbedding to avoid calling any remote or heavy local model APIs
    mock_embed = MockEmbedding(embed_dim=384)
    
    with patch("ingestion.store.qdrant_client.QdrantClient") as mock_client_cls, \
         patch("ingestion.store.QdrantVectorStore") as mock_store_cls, \
         patch("ingestion.store.StorageContext.from_defaults") as mock_storage_ctx_cls, \
         patch("ingestion.store.VectorStoreIndex") as mock_index_cls:
         
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        
        build_vector_index(
            nodes=nodes, 
            embed_model=mock_embed, 
            qdrant_url="http://localhost:6333", 
            collection_name="test_collection"
        )
        
        # Verify Qdrant connection URL
        mock_client_cls.assert_called_once_with(url="http://localhost:6333")
        # Verify collection setup
        mock_store_cls.assert_called_once_with(client=mock_client, collection_name="test_collection")
        # Verify Index creation
        mock_index_cls.assert_called_once()
