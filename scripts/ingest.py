import os
import sys
# We import load_dotenv to read configuration settings from the local .env file.
from dotenv import load_dotenv

# We add the parent directory (project root) to sys.path so we can import our custom ingestion package 
# without encountering ModuleNotFoundError, regardless of where this script is executed from.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# We import the pipeline stage functions from our ingestion package.
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model
from ingestion.store import build_vector_index

def main():
    print("==================================================")
    print("Starting DocMind Document Ingestion Pipeline...")
    print("==================================================")
    
    # (1) Load environment variables from the .env file.
    load_dotenv()
    
    # We retrieve the configuration variables with safe fallback defaults.
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "docmind_docs")
    embed_model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
    
    # Define paths to sample data.
    sample_data_dir = os.path.join(project_root, "data", "sample")
    
    # Ensure that the data/sample directory exists.
    # If it does not exist, we create it so the user can easily drop PDFs in it.
    if not os.path.exists(sample_data_dir):
        print(f"Creating directory: {sample_data_dir}")
        os.makedirs(sample_data_dir, exist_ok=True)
        
    # (2) Load documents using loader.py.
    print("\n--- Phase 2.1: Loading Documents ---")
    try:
        documents = load_documents(sample_data_dir)
        print(f"Success: Loaded {len(documents)} document pages/objects.")
    except Exception as e:
        print(f"Error during document loading: {e}")
        sys.exit(1)
        
    if not documents:
        print("\n[Warning] No documents were found to ingest!")
        print(f"Please drop one or more PDF files in '{sample_data_dir}' and run this script again.")
        sys.exit(0)
        
    # (3) Chunk documents using chunker.py.
    print("\n--- Phase 2.2: Chunking Documents ---")
    nodes = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"Success: Created {len(nodes)} text chunks.")
    
    # (4) Load the embedding model using embedder.py.
    print("\n--- Phase 2.3: Loading Embedding Model ---")
    embed_model = get_embedding_model(embed_model_name)
    
    # (5) Connect to Qdrant, save the nodes, and build the index using store.py.
    print("\n--- Phase 2.4: Building Vector Index & Storing in Qdrant ---")
    try:
        index = build_vector_index(nodes, embed_model, qdrant_url, collection_name)
        print("\n==================================================")
        print("Success: Document ingestion pipeline completed!")
        print(f"Loaded {len(documents)} documents, split into {len(nodes)} chunks.")
        print(f"Indexed vectors stored in Qdrant collection: '{collection_name}'")
        print(f"Access Qdrant dashboard at: {qdrant_url}/dashboard")
        print("==================================================")
    except Exception as e:
        print(f"Error during database indexing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
