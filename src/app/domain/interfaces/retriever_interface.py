from abc import ABC, abstractmethod
from typing import List
from app.domain.interfaces.vectorstore_interface import RetrievalResult


class BaseRetriever(ABC):
    """Abstract interface for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        pass
