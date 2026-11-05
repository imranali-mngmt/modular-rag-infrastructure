from abc import ABC, abstractmethod
from typing import List
from app.domain.interfaces.vectorstore_interface import RetrievalResult


class BaseReranker(ABC):
    """Abstract interface for reranking retrieved documents."""

    @abstractmethod
    def rerank(
        self, query: str, documents: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank documents based on relevance to the query."""
        pass
