# WHY two modes:
# 'local' mode (default): Loads the sentence-transformers model directly into server RAM via PyTorch.
#   Good for: local development, Docker, machines with 1GB+ RAM.
#   Bad for: Render free tier (512MB RAM limit — model alone can use 300MB+).
#
# 'api' mode: Sends text to HuggingFace's Inference API to get embeddings back.
#   Good for: cloud free-tier hosting (no model in RAM, no PyTorch initialization).
#   Bad for: rate-limited on HF free tier (~requests per hour limit).
#
# The active mode is controlled by the EMBED_MODE environment variable.

from config.settings import settings


def get_embedding_model():
    """
    Returns the appropriate embedding model based on the EMBED_MODE environment variable.

    Returns:
        An embedding model compatible with LlamaIndex (HuggingFaceEmbedding or HuggingFaceInferenceAPIEmbedding).
    """
    if settings.embed_mode.lower() == "api":
        # WHY HuggingFaceInferenceAPIEmbedding: Calls HuggingFace's hosted inference endpoints instead
        # of loading the model locally. This keeps server RAM usage very low (no PyTorch model in memory).
        from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding

        print(f"[EmbedMode=api] Using HuggingFace Inference API for model: {settings.embed_model}")
        return HuggingFaceInferenceAPIEmbedding(
            model_name=settings.embed_model,
            token=settings.hf_token or None,
        )
    else:
        # WHY HuggingFaceEmbedding: Loads the full model locally via PyTorch.
        # Faster after warm-up, no rate limits, no API token needed. Default for local dev and Docker.
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        print(f"[EmbedMode=local] Loading model into RAM: {settings.embed_model}")
        return HuggingFaceEmbedding(model_name=settings.embed_model)
