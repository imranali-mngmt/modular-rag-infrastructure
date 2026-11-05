import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- Provider Selection ---
    USE_MOCKS: bool = field(
        default_factory=lambda: os.getenv("USE_MOCKS", "false").lower() == "true"
    )
    LLM_PROVIDER: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower()
    )
    EMBEDDING_PROVIDER: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "google").lower()
    )
    # --- OpenAI Settings ---
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    )

    # --- Google Gemini Settings ---
    GOOGLE_API_KEY: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    GOOGLE_MODEL: str = field(
        default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")
    )
    GOOGLE_EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2"
        )
    )

    # --- OpenRouter Settings ---
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    OPENROUTER_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL", "inclusionai/ling-3.0-tiny"
        )
    )

    OPENAI_EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )

    # --- Groq Settings ---
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GROQ_MODEL: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    )

    # --- Qdrant Settings ---
    QDRANT_URL: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    QDRANT_API_KEY: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    QDRANT_COLLECTION_NAME: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "documents")
    )
    QDRANT_MODE: str = field(
        default_factory=lambda: os.getenv("QDRANT_MODE", "memory").lower()
    )
    QDRANT_LOCAL_PATH: str = field(
        default_factory=lambda: os.getenv("QDRANT_LOCAL_PATH", "./qdrant_data")
    )

    # --- Document Processing Settings ---
    # --- Document Processing & Retrieval Settings ---
    CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )

    DEFAULT_RETRIEVAL_K: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_RETRIEVAL_K", "4"))
    )
    HYBRID_FETCH_K: int = field(
        default_factory=lambda: int(os.getenv("HYBRID_FETCH_K", "20"))
    )  # Fetch 20, rerank to 4
    USE_RERANKER: bool = field(
        default_factory=lambda: os.getenv("USE_RERANKER", "true").lower() == "true"
    )

    # --- Cohere Settings ---
    COHERE_API_KEY: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    COHERE_RERANK_MODEL: str = field(
        default_factory=lambda: os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
    )


settings = Settings()
