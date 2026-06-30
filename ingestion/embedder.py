# We import HuggingFaceEmbedding from LlamaIndex's HuggingFace integration package.
# This integrates local Hugging Face sentence-transformers models directly into LlamaIndex.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# An embedding is a vector representation of text (a list of numbers, e.g., 384 dimensions for all-MiniLM-L6-v2).
# It captures the semantic meaning and context of a text segment. Words or sentences with similar meanings
# are placed closer together in this high-dimensional vector space, allowing for fast mathematical similarity searches.
#
# We use sentence-transformers/all-MiniLM-L6-v2 because:
# 1. It is a lightweight model (only ~80MB), making it run extremely fast even on standard CPUs.
# 2. It runs completely offline without requiring any external APIs or internet access after download.
# 3. It provides high-quality retrieval accuracy for general English texts.

def get_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    """
    Loads a HuggingFace sentence-transformer embedding model.
    
    Parameters:
        model_name (str): The name of the HuggingFace model (e.g. 'sentence-transformers/all-MiniLM-L6-v2').
        
    Returns:
        HuggingFaceEmbedding: The initialized embedding model object.
    """
    # Print status indicating we are loading the embedding model
    print(f"Loading embedding model: {model_name}...")
    
    # We initialize the HuggingFaceEmbedding class with the specified model name.
    # When this code runs for the first time, it downloads the model weights from Hugging Face hub.
    # On subsequent runs, it loads the model weights directly from the local disk cache.
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    
    # Return the loaded embedding model instance
    return embed_model
