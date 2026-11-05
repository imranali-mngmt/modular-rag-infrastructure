from typing import Type
from app.core.config import settings, Settings
from app.core.exceptions import LLMProviderError, APIKeyMissingError
from app.domain.interfaces.llm_interface import BaseLLM
from app.infrastructure.llm.openai_llm import OpenAILLM
from app.infrastructure.llm.gemini_llm import GeminiLLM
from app.infrastructure.llm.openrouter_llm import OpenRouterLLM
from app.infrastructure.llm.groq_llm import GroqLLM

# --- The Registry Pattern ---------------------------------------------------
# Maps provider names (strings) to their corresponding classes.
# This dictionary is the ONLY place you need to touch when adding a new provider.
LLM_REGISTRY: dict[str, Type[BaseLLM]] = {
    "openai": OpenAILLM,
    "gemini": GeminiLLM,
    "openrouter": OpenRouterLLM,  # <-- Just add it here!
    "groq": GroqLLM,  # <-- And here!
}

# --- Provider-specific API key requirements ---------------------------------
API_KEY_REQUIREMENTS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",  # <-- Add the requirement
    "groq": "GROQ_API_KEY",  # <-- Add the requirement
}


def build_llm(s: Settings = settings) -> BaseLLM:
    """
    Factory function that returns the correct LLM based on settings.
    Uses the Registry Pattern to avoid if/else chains (Open/Closed Principle).

    Raises:
        LLMProviderError: If the provider is unknown or not in registry.
        APIKeyMissingError: If the required API key for the provider is missing.
    """
    # 1. Get the provider from settings
    provider = s.LLM_PROVIDER.lower()

    # 2. Fetch the class from the registry (O(1) lookup)
    llm_class = LLM_REGISTRY.get(provider)

    # 3. Handle unknown providers with clear error message
    if llm_class is None:
        available = ", ".join(LLM_REGISTRY.keys())
        raise LLMProviderError(
            f"Unknown LLM provider: '{provider}'. "
            f"Available providers: {available}. "
            f"Check your LLM_PROVIDER setting in .env file."
        )

    # 4. Validate API key requirements
    required_key = API_KEY_REQUIREMENTS.get(provider)
    if required_key:
        api_key = getattr(s, required_key, None)
        if not api_key:
            raise APIKeyMissingError(
                f"API key '{required_key}' is missing for provider '{provider}'. "
                f"Please add it to your .env file."
            )

    # 5. Instantiate and return
    try:
        return llm_class(s)
    except Exception as e:
        raise LLMProviderError(
            f"Failed to initialize LLM provider '{provider}': {str(e)}"
        ) from e
