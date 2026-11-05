from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseSparseEmbeddings(ABC):
    """Abstract interface for sparse (BM25/keyword) embeddings."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Embed documents into sparse vector format (indices and values)."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> Dict[str, Any]:
        """Embed a query into sparse vector format."""
        pass
