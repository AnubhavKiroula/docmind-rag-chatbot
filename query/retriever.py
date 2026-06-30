import qdrant_client
# We import HuggingFaceEmbedding to convert the user's search query into the same vector space as the documents.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# We import QdrantVectorStore, VectorStoreIndex, StorageContext, and Settings from LlamaIndex core to access the DB.
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from typing import List, Tuple

# We import the instantiated settings object to retrieve configuration values (URLs, model names, etc.).
from config.settings import settings

# Semantic similarity search means finding document chunks whose core meaning is closest to the query's meaning.
# Unlike keyword matching (which looks for exact words), semantic search maps both the query and the documents 
# into vectors. Distance is calculated between these vectors, allowing us to find relevant content even if
# different words are used (e.g. searching 'automobile' can match 'car').
#
# top_k=3 is selected as a default because:
# 1. It provides enough background information (context) for the LLM to draft a rich, accurate answer.
# 2. It keeps the context size small enough to avoid exceeding the LLM's prompt window and prevents latency lag.
# 3. It filters out low-relevance results that might confuse the generator.

class QdrantRetriever:
    """
    Retriever class that connects to Qdrant and retrieves relevant document chunks based on a query.
    """
    def __init__(self):
        print(f"Loading embedding model for retriever: {settings.embed_model}...")
        # Load the Hugging Face sentence-transformers model.
        # This is needed to convert raw user queries into semantic embeddings.
        self.embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
        
        # Set the default embedding model in LlamaIndex global settings.
        # This makes sure that index queries use our sentence-transformer model.
        Settings.embed_model = self.embed_model
        
        print(f"Connecting retriever to Qdrant at {settings.qdrant_url}...")
        # Establish a client connection to the running Qdrant instance.
        self.client = qdrant_client.QdrantClient(url=settings.qdrant_url)
        
        # Configure the vector store link pointing to our specific collection.
        self.vector_store = QdrantVectorStore(client=self.client, collection_name=settings.qdrant_collection_name)
        
        # Set up a storage context referencing the vector store.
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Load the VectorStoreIndex. Instead of rebuilding it from source documents, 
        # we load it directly from our Qdrant vector database (very fast).
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )

    def retrieve(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Performs a semantic similarity search on Qdrant and returns the top_k relevant chunks.
        
        Parameters:
            query_text (str): The search question or statement.
            top_k (int): Number of top documents to retrieve.
            
        Returns:
            List[Tuple[str, float]]: A list of tuples containing (chunk_text, similarity_score).
        """
        print(f"Retrieving top {top_k} chunk(s) for query: '{query_text}'...")
        
        # Generate a retriever object from our loaded index, configured with similarity_top_k.
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        # Execute the semantic query retrieval. This returns a list of NodeWithScore objects.
        results = retriever.retrieve(query_text)
        
        # Format the LlamaIndex retrieval nodes into simple, clean (text, score) tuples.
        extracted_chunks = []
        for node_with_score in results:
            chunk_text = node_with_score.node.get_content()
            similarity_score = float(node_with_score.score)
            extracted_chunks.append((chunk_text, similarity_score))
            
        # Return the retrieved chunks list.
        return extracted_chunks
