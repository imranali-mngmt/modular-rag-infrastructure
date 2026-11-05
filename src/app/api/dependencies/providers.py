from functools import lru_cache
from app.core.config import settings
from app.domain.interfaces.llm_interface import BaseLLM
from app.domain.interfaces.embeddings_interface import BaseEmbeddings
from app.domain.interfaces.sparse_embeddings_interface import BaseSparseEmbeddings
from app.domain.interfaces.vectorstore_interface import BaseVectorStore
from app.domain.interfaces.retriever_interface import BaseRetriever
from app.domain.interfaces.reranker_interface import BaseReranker

from app.infrastructure.llm.factory import build_llm
from app.infrastructure.embeddings.factory import build_embeddings
from app.infrastructure.embeddings.sparse_embeddings import FastEmbedSparseEmbeddings
from app.infrastructure.vectorstores.qdrant_store import QdrantHybridVectorStore
from app.infrastructure.retrievers.vectorstore_retriever import HybridRerankingRetriever
from app.infrastructure.rerankers.cohere_reranker import CohereReranker
from app.application.services.indexing_service import IndexingService
from app.application.services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_embeddings() -> BaseEmbeddings:
    return build_embeddings()


@lru_cache(maxsize=1)
def get_sparse_embeddings() -> BaseSparseEmbeddings:
    return FastEmbedSparseEmbeddings()


@lru_cache(maxsize=1)
def get_vector_store() -> BaseVectorStore:
    return QdrantHybridVectorStore(
        settings=settings,
        dense_embeddings=get_embeddings(),
        sparse_embeddings=get_sparse_embeddings(),
    )


@lru_cache(maxsize=1)
def get_reranker() -> BaseReranker:
    return CohereReranker(settings)


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    return HybridRerankingRetriever(
        vector_store=get_vector_store(),
        embeddings=get_embeddings(),
        reranker=get_reranker(),
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_indexing_service() -> IndexingService:
    return IndexingService(
        vector_store=get_vector_store(),
        dense_embeddings=get_embeddings(),
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(retriever=get_retriever())


def reset_all_caches() -> None:
    for fn in [
        get_embeddings,
        get_sparse_embeddings,
        get_vector_store,
        get_reranker,
        get_retriever,
        get_indexing_service,
        get_retrieval_service,
    ]:
        fn.cache_clear()
