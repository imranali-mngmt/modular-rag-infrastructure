from typing import List
from dataclasses import dataclass
from app.domain.interfaces.retriever_interface import BaseRetriever
from app.domain.interfaces.vectorstore_interface import RetrievalResult


@dataclass
class RetrievalResponse:
    """Response from a retrieval operation."""

    query: str
    results: List[RetrievalResult]
    total_found: int


class RetrievalService:
    """Service for retrieving relevant documents."""

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def retrieve(self, query: str, k: int = None) -> RetrievalResponse:
        """Retrieve relevant documents for a query."""
        results = self.retriever.retrieve(query, k=k)

        return RetrievalResponse(query=query, results=results, total_found=len(results))
