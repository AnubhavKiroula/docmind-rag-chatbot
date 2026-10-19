import os

# We import load_dotenv to locate and load the .env file from the root directory.
from dotenv import load_dotenv

# We import BaseModel and Field from pydantic to construct and document our validated Settings model.
from pydantic import BaseModel, Field
from typing import Tuple, Optional

# Load environment variables from the .env file.
# If .env does not exist, os.getenv will fall back to default values or system variables.
load_dotenv()


class Settings(BaseModel):
    """
    Settings class that loads and validates environment variables.
    Provides fallback defaults so the application can start under default conditions.
    Supports both local execution (Docker Qdrant) and cloud deployment (Qdrant Cloud).
    """

    # URL of the running local Qdrant instance
    qdrant_url: str = Field(
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        description="The HTTP endpoint URL where the local Qdrant container is hosted.",
    )

    # Qdrant Cloud cluster endpoint URL
    # WHY: Allows cloud deployment to connect to managed Qdrant Cloud without changing application code.
    qdrant_cloud_url: str = Field(
        default=os.getenv("QDRANT_CLOUD_URL", ""),
        description="The HTTPS cluster endpoint URL provided by Qdrant Cloud.",
    )

    # Qdrant Cloud authentication API key
    # WHY: Required for authenticating requests sent to managed Qdrant Cloud clusters.
    qdrant_api_key: str = Field(
        default=os.getenv("QDRANT_API_KEY", ""),
        description="The secret API key used for Qdrant Cloud authentication.",
    )

    # Vector store execution mode: 'local' or 'cloud'
    # WHY: Enables seamless mode switching. Local dev uses Docker Qdrant (no API key required),
    # while cloud deployment uses Qdrant Cloud (API key required). Same codebase, environment-driven configuration.
    qdrant_mode: str = Field(
        default=os.getenv("QDRANT_MODE", "local"),
        description="Vector storage mode: 'local' connects to local container, 'cloud' connects to Qdrant Cloud.",
    )

    # Application deployment environment: 'development' or 'production'
    # WHY: Allows the app to adjust logging, error verbosity, and security defaults based on deployment context.
    environment: str = Field(
        default=os.getenv("ENVIRONMENT", "development"),
        description="Deployment stage: 'development' for local work, 'production' for live hosting.",
    )

    # Qdrant collection name
    qdrant_collection_name: str = Field(
        default=os.getenv("QDRANT_COLLECTION_NAME", "docmind_docs"),
        description="The name of the vector database collection containing our document embeddings.",
    )

    # Embedding model name
    embed_model: str = Field(
        default=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        description="The Hugging Face sentence-transformers model used to embed text chunks.",
    )

    # HuggingFace API token for cloud inference
    # WHY: When EMBED_MODE=api, embeddings are generated via the HuggingFace Inference API
    # instead of loading a local model. This eliminates PyTorch from the server's RAM footprint,
    # making it suitable for Render's 512MB free tier.
    hf_token: str = Field(
        default=os.getenv("HF_TOKEN", ""),
        description="HuggingFace API token for Inference API access. Required when EMBED_MODE=api.",
    )

    # Embedding execution mode: 'local' (default) or 'api'
    # WHY: 'local' loads the model into RAM (good for dev/Docker); 'api' uses HF Inference API (good for free cloud hosting).
    embed_mode: str = Field(
        default=os.getenv("EMBED_MODE", "local"),
        description="'local' loads sentence-transformers model into RAM. 'api' calls HuggingFace Inference API.",
    )

    # Target token size for text chunking
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "512")),
        description="The maximum number of tokens allowed in an individual text chunk.",
    )

    # Token overlap size between adjacent chunks
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "64")),
        description="The number of tokens that overlap from one chunk into the next.",
    )

    # Groq Cloud API Key
    groq_api_key: str = Field(
        default=os.getenv("GROQ_API_KEY", ""),
        description="API key for Groq Cloud. Required only if using Groq cloud inference.",
    )

    # Groq Cloud model name
    # WHY configurable: this was previously hardcoded to 'llama-3.1-8b-instant'. Groq retires
    # models without notice, which breaks generation at runtime with a 404 and needs a code
    # change to fix. Reading it from the environment lets the model be swapped via config.
    groq_model: str = Field(
        default=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        description="The model name requested from the Groq Cloud API.",
    )

    # Local Ollama LLM Model name
    local_llm_model: str = Field(
        default=os.getenv("LOCAL_LLM_MODEL", "llama3.2"),
        description="The model name to request from Ollama (e.g. 'llama3.2').",
    )

    # Local Ollama endpoint URL
    ollama_base_url: str = Field(
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="The API URL where local Ollama is hosted.",
    )

    # Active LLM Mode: 'ollama' or 'groq'
    llm_mode: str = Field(
        default=os.getenv("LLM_MODE", "ollama"),
        description="Controls backend generation routing. 'ollama' selects local LLM, 'groq' selects Groq Cloud API.",
    )

    # Default retrieval top_k limit
    top_k: int = Field(
        default=int(os.getenv("DEFAULT_TOP_K", "3")),
        description="The default number of document chunks to retrieve during a semantic query.",
    )

    def get_qdrant_connection(self) -> Tuple[str, Optional[str]]:
        """
        Dynamically returns the target Qdrant connection endpoint and API key.

        WHY: Centralizes vector store connection logic so database client instantiation code
        (in ingestion/store.py and query/retriever.py) does not need conditional checks.
        If QDRANT_MODE is 'cloud', returns (QDRANT_CLOUD_URL, QDRANT_API_KEY).
        If QDRANT_MODE is 'local', returns (QDRANT_URL, None).

        Returns:
            Tuple[str, Optional[str]]: A tuple containing (target_url, target_api_key).
        """
        if self.qdrant_mode.lower() == "cloud":
            return self.qdrant_cloud_url, self.qdrant_api_key
        return self.qdrant_url, None


# Instantiate Settings globally. Any module importing settings will access this validated object.
settings = Settings()
