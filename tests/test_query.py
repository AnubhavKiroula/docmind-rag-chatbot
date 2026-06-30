import pytest
from unittest.mock import patch, MagicMock

# Import LlamaIndex components needed for mocked data representation
from llama_index.core.schema import TextNode, NodeWithScore

# Import query engine components to test
from query.retriever import QdrantRetriever
from query.llm_client import LLMClient
from query.synthesizer import ResponseSynthesizer

# Unit tests verify that individual code components work correctly in isolation, 
# preventing changes in one module from silently breaking another. By mocking external connections 
# (like Qdrant databases or remote LLM APIs), we can run fast, deterministic tests without relying 
# on internet access or active local servers.

def test_retriever_returns_chunks():
    """
    Verifies that QdrantRetriever successfully retrieves semantic chunks from Qdrant 
    and transforms LlamaIndex NodeWithScore outputs into raw (text, score) tuples.
    We mock Qdrant connections and vector index construction.
    """
    with patch("query.retriever.qdrant_client.QdrantClient") as mock_client_cls, \
         patch("query.retriever.QdrantVectorStore") as mock_store_cls, \
         patch("query.retriever.StorageContext.from_defaults") as mock_storage_ctx_cls, \
         patch("query.retriever.VectorStoreIndex.from_vector_store") as mock_index_cls:
         
        # Mock index behavior
        mock_retriever = MagicMock()
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_index_cls.return_value = mock_index
        
        # Mock retriever result nodes
        mock_node = TextNode(text="Simulated document chunk content.")
        mock_node_with_score = NodeWithScore(node=mock_node, score=0.825)
        mock_retriever.retrieve.return_value = [mock_node_with_score]
        
        # Instantiate retriever and query it
        retriever = QdrantRetriever()
        results = retriever.retrieve("search query text")
        
        # Verify formatting matches expected Tuple structures
        assert len(results) == 1
        assert results[0] == ("Simulated document chunk content.", 0.825)


def test_synthesizer_returns_response():
    """
    Verifies that ResponseSynthesizer constructs prompts correctly, invokes 
    the generator, and correctly averages similarity scores to yield confidence ratings.
    We stub the LLMClient using MagicMock.
    """
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = "Answer based on document text."
    
    synthesizer = ResponseSynthesizer()
    retrieved_chunks = [
        ("Source document chunk A.", 0.90),
        ("Source document chunk B.", 0.70)
    ]
    
    response = synthesizer.synthesize("User question?", retrieved_chunks, mock_llm_client)
    
    # Assert answer construction matches mock return value
    assert response["answer"] == "Answer based on document text."
    # Assert confidence score is computed as average: (0.90 + 0.70) / 2 = 0.80
    assert response["confidence_score"] == 0.80
    # Assert sources list is correctly populated
    assert response["sources"] == ["Source document chunk A.", "Source document chunk B."]


def test_llm_client_fallback():
    """
    Verifies the automatic local-to-cloud fallback mechanism in LLMClient.
    If Ollama local inference throws a connection error, it must catch the error 
    and request completion from Groq API if an API key is configured.
    """
    with patch("query.llm_client.httpx.post") as mock_post:
        # Simulate local Ollama being offline by raising a connection exception
        mock_post.side_effect = Exception("Ollama HTTP connection timed out")
        
        with patch("query.llm_client.Groq") as mock_groq_cls:
            mock_groq = MagicMock()
            mock_groq_cls.return_value = mock_groq
            
            # Setup mock Groq response structures
            mock_chat = MagicMock()
            mock_groq.chat.completions.create.return_value = mock_chat
            mock_choice = MagicMock()
            mock_choice.message.content = "Response from Groq Cloud API"
            mock_chat.choices = [mock_choice]
            
            # Patch settings to configure 'ollama' mode with fallback credentials
            with patch("query.llm_client.settings") as mock_settings:
                mock_settings.llm_mode = "ollama"
                mock_settings.groq_api_key = "gsk_dummy_api_key"
                mock_settings.local_llm_model = "llama3.2"
                mock_settings.ollama_base_url = "http://localhost:11434"
                
                client = LLMClient()
                result = client.generate("Fallback prompt query")
                
                # Verify Groq completion API was indeed called as fallback
                mock_groq.chat.completions.create.assert_called_once()
                assert result == "Response from Groq Cloud API"
