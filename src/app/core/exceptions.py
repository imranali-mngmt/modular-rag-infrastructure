"""Custom exceptions for the application."""


class ConfigurationError(Exception):
    """Raised when there's a configuration issue (missing API keys, invalid settings, etc.)."""

    pass


class LLMProviderError(Exception):
    """Raised when there's an issue with the LLM provider (unknown provider, initialization failure, etc.)."""

    pass


class APIKeyMissingError(ConfigurationError):
    """Raised when a required API key is missing."""

    pass
