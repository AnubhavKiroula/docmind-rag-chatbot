import qdrant_client
# We import QdrantVectorStore which acts as the bridge between LlamaIndex and Qdrant.
from llama_index.vector_stores.qdrant import QdrantVectorStore
# We import VectorStoreIndex and StorageContext to construct and manage our vector database index.
# We also import Settings from LlamaIndex core to set the default embedding model.
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from typing import List
from llama_index.core.schema import BaseNode

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
    qdrant_url: str, 
    collection_name: str
) -> VectorStoreIndex:
    """
    Connects to Qdrant, registers the vector store, creates/populates the collection, and builds a VectorStoreIndex.
    
    Parameters:
        nodes (List[BaseNode]): The text chunks/nodes to store and index.
        embed_model: The LlamaIndex embedding model to generate embeddings for the nodes.
        qdrant_url (str): The URL of the running Qdrant instance.
        collection_name (str): The name of the collection in Qdrant.
        
    Returns:
        VectorStoreIndex: The generated and connected index.
    """
    print(f"Connecting to Qdrant at {qdrant_url}...")
    # (1) We establish a connection to the Qdrant service using the qdrant-client.
    client = qdrant_client.QdrantClient(url=qdrant_url)
    
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
