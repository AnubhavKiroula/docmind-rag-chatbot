import qdrant_client
# We import QdrantVectorStore which acts as the bridge between LlamaIndex and Qdrant.
from llama_index.vector_stores.qdrant import QdrantVectorStore
# We import VectorStoreIndex and StorageContext to construct and manage our vector database index.
# We also import Settings from LlamaIndex core to set the default embedding model.
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from typing import List, Optional
from llama_index.core.schema import BaseNode
# Import instantiated settings to retrieve dynamic Qdrant connection configuration.
from config.settings import settings

# A vector store is a specialized database (like Qdrant) optimized for storing and querying high-dimensional vectors.
# Instead of doing traditional keyword matching, it allows us to perform similarity searches to find text chunks 
# that are closest in meaning to a user's query.
#
# An index (specifically VectorStoreIndex) is a lookup structure built on top of the vector store.
# It links each vector embedding back to its original raw text chunk and metadata, so that when a match is found,
# the corresponding text can be retrieved and sent to the LLM.


def build_vector_index(
    nodes: List[BaseNode], 
    embed_model, 
    qdrant_url: Optional[str] = None, 
    collection_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> VectorStoreIndex:
    """
    Connects to Qdrant (local or cloud), registers the vector store, creates/populates the collection, and builds a VectorStoreIndex.
    
    Parameters:
        nodes (List[BaseNode]): The text chunks/nodes to store and index.
        embed_model: The LlamaIndex embedding model to generate embeddings for the nodes.
        qdrant_url (str, optional): The URL of the target Qdrant instance. Defaults to settings value.
        collection_name (str, optional): The name of the collection in Qdrant. Defaults to settings value.
        api_key (str, optional): Authentication API key for Qdrant Cloud. Defaults to settings value.
        
    Returns:
        VectorStoreIndex: The generated and connected index.
    """
    # Environment-driven configuration pattern:
    # If parameters are not explicitly passed, retrieve target connection details from settings.get_qdrant_connection().
    # This allows the exact same code to work with local Docker Qdrant (api_key=None) or Qdrant Cloud (api_key=secret).
    if qdrant_url is None:
        target_url, target_api_key = settings.get_qdrant_connection()
    else:
        target_url = qdrant_url
        target_api_key = api_key

    if collection_name is None:
        collection_name = settings.qdrant_collection_name

    print(f"Connecting to Qdrant at {target_url} (mode: {settings.qdrant_mode})...")
    # (1) We establish a connection to the Qdrant service using qdrant-client.
    # The api_key parameter is passed to enable Qdrant Cloud authentication (ignored when None for local Qdrant).
    client = qdrant_client.QdrantClient(url=target_url, api_key=target_api_key)
    
    # We update LlamaIndex global Settings to use our loaded embedding model.
    # This guarantees that the indexing process generates embeddings using our specified sentence-transformers model.
    Settings.embed_model = embed_model
    
    print(f"Configuring vector store for collection: '{collection_name}'...")
    # (2) We initialize QdrantVectorStore, passing it our connected client and specifying the collection name.
    # If the collection does not exist, QdrantVectorStore will automatically attempt to create it.
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    
    # We define a StorageContext to tell LlamaIndex that it should store the index elements
    # inside our Qdrant vector database instead of in-memory or on local disk.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"Building VectorStoreIndex with {len(nodes)} node(s)...")
    # (3) We build the VectorStoreIndex. LlamaIndex will automatically use the embed_model
    # to compute embeddings for each node, push the embeddings and text to Qdrant, and set up the index.
    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # (4) Return the fully initialized index.
    return index
