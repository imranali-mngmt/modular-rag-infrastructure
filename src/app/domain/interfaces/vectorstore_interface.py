from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: dict


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        dense_embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> int:
        pass

    @abstractmethod
    def hybrid_search(
        self,
        query_text: str,
        dense_query_embedding: List[float],
        fetch_k: int = 20,
        return_k: int = 4,
    ) -> List[RetrievalResult]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
