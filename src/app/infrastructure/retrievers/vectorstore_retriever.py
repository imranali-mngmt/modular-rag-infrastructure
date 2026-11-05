from typing import List
from app.core.config import Settings
from app.domain.interfaces.retriever_interface import BaseRetriever
from app.domain.interfaces.vectorstore_interface import BaseVectorStore, RetrievalResult
from app.domain.interfaces.embeddings_interface import BaseEmbeddings
from app.domain.interfaces.reranker_interface import BaseReranker
from app.infrastructure.vectorstores.qdrant_store import QdrantHybridVectorStore


class HybridRerankingRetriever(BaseRetriever):
    """Retriever that uses Qdrant Hybrid Search + Cohere Reranking."""

    def __init__(
        self,
        vector_store: QdrantHybridVectorStore,
        embeddings: BaseEmbeddings,
        reranker: BaseReranker,
        settings: Settings,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.reranker = reranker
        self.settings = settings

    def retrieve(self, query: str, k: int = None) -> List[RetrievalResult]:
        k = k or self.settings.DEFAULT_RETRIEVAL_K
        fetch_k = self.settings.HYBRID_FETCH_K

        # 1. Get dense embedding for the query
        dense_query_embedding = self.embeddings.embed_query(query)

        # 2. Perform Hybrid Search (fetch more than we need for the reranker)
        print(f"   🔍 [Retriever] Running hybrid search (fetching top {fetch_k})...")
        initial_results = self.vector_store.hybrid_search(
            query_text=query,
            dense_query_embedding=dense_query_embedding,
            fetch_k=fetch_k,
            return_k=fetch_k,
        )

        print(
            f"   🔄 [Retriever] Found {len(initial_results)} initial results. Reranking with Cohere..."
        )

        # 3. Rerank the results
        if self.settings.USE_RERANKER and initial_results:
            final_results = self.reranker.rerank(query, initial_results)
            return final_results[:k]  # Return only the top K after reranking

        return initial_results[:k]
