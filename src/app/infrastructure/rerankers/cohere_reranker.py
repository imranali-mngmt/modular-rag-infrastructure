from typing import List
import cohere
from app.core.config import Settings
from app.domain.interfaces.reranker_interface import BaseReranker
from app.domain.interfaces.vectorstore_interface import RetrievalResult


class CohereReranker(BaseReranker):
    """Production-grade reranker using Cohere's API."""

    def __init__(self, settings: Settings):
        if not settings.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is missing in .env")

        self.client = cohere.Client(settings.COHERE_API_KEY)
        self.model = settings.COHERE_RERANK_MODEL

    def rerank(
        self, query: str, documents: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        if not documents:
            return []

        # Cohere expects a list of strings
        texts = [doc.content for doc in documents]

        response = self.client.rerank(
            model=self.model, query=query, documents=texts, top_n=len(documents)
        )

        reranked_results = []
        for res in response.results:
            original_doc = documents[res.index]
            reranked_results.append(
                RetrievalResult(
                    content=original_doc.content,
                    score=res.relevance_score,  # Cohere returns a float between 0.0 and 1.0
                    metadata=original_doc.metadata,
                )
            )

        # Sort by score descending (Cohere usually returns them sorted, but good to be safe)
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results
