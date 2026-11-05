from typing import Type
from app.core.config import settings, Settings
from app.core.exceptions import LLMProviderError, APIKeyMissingError
from app.domain.interfaces.embeddings_interface import BaseEmbeddings
from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddingsModel
from app.infrastructure.embeddings.google_embeddings import GoogleEmbeddingsModel

# --- The Registry Pattern ---------------------------------------------------
EMBEDDING_REGISTRY: dict[str, Type[BaseEmbeddings]] = {
    "openai": OpenAIEmbeddingsModel,
    "google": GoogleEmbeddingsModel,
}

# --- Provider-specific API key requirements ---------------------------------
EMBEDDING_API_KEY_REQUIREMENTS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def build_embeddings(s: Settings = settings) -> BaseEmbeddings:
    """
    Factory function that returns the correct embedding model based on settings.
    Uses the Registry Pattern (Open/Closed Principle).
    """
    provider = s.EMBEDDING_PROVIDER.lower()

    print(f"🔧 [Embeddings Factory] Building embeddings for provider: '{provider}'")

    embeddings_class = EMBEDDING_REGISTRY.get(provider)

    if embeddings_class is None:
        available = ", ".join(EMBEDDING_REGISTRY.keys())
        raise LLMProviderError(
            f"Unknown embedding provider: '{provider}'. "
            f"Available providers: {available}. "
            f"Check your EMBEDDING_PROVIDER setting in .env file."
        )

    required_key = EMBEDDING_API_KEY_REQUIREMENTS.get(provider)
    if required_key:
        api_key = getattr(s, required_key, None)
        if not api_key:
            raise APIKeyMissingError(
                f"API key '{required_key}' is missing for embedding provider '{provider}'. "
                f"Please add it to your .env file."
            )
        print(f"Properly worked [Embeddings Factory] API key '{required_key}' found")

    try:
        embeddings_instance = embeddings_class(s)
        print(
            f"Properly worked [Embeddings Factory] Successfully initialized {embeddings_class.__name__}"
        )
        return embeddings_instance
    except Exception as e:
        raise LLMProviderError(
            f"Failed to initialize embedding provider '{provider}': {str(e)}"
        ) from e
