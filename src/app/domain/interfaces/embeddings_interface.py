from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Single text string to embed

        Returns:
            Embedding vector (list of floats)
        """
        pass
