import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import Settings
from app.domain.interfaces.vectorstore_interface import BaseVectorStore, RetrievalResult
from app.domain.interfaces.embeddings_interface import BaseEmbeddings
from app.domain.interfaces.sparse_embeddings_interface import BaseSparseEmbeddings


class QdrantHybridVectorStore(BaseVectorStore):
    """Qdrant vector store with native Hybrid Search (Dense + Sparse) support."""

    def __init__(
        self,
        settings: Settings,
        dense_embeddings: BaseEmbeddings,
        sparse_embeddings: BaseSparseEmbeddings,
    ):
        self.settings = settings
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.dense_embeddings = dense_embeddings
        self.sparse_embeddings = sparse_embeddings

        if settings.QDRANT_USE_IN_MEMORY:
            self.client = QdrantClient(":memory:")
            print("📦 [Qdrant] Using in-memory storage")
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            )
            print(f"📦 [Qdrant] Connected to {settings.QDRANT_URL}")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            sample_dense = self.dense_embeddings.embed_query("test")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=len(sample_dense), distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
            print(
                f"Properly worked [Qdrant] Created hybrid collection '{self.collection_name}'"
            )
        else:
            print(
                f"Properly worked [Qdrant] Hybrid collection '{self.collection_name}' already exists"
            )

    def add_documents(
        self,
        texts: List[str],
        dense_embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> int:
        if not texts:
            return 0

        # Generate sparse embeddings locally (free and fast)
        sparse_embeddings = self.sparse_embeddings.embed_documents(texts)

        points = []
        for text, dense_vec, sparse_vec, metadata in zip(
            texts, dense_embeddings, sparse_embeddings, metadatas
        ):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec["indices"], values=sparse_vec["values"]
                        ),
                    },
                    payload={"content": text, **metadata},
                )
            )

        # Batch upload
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name, points=points[i : i + batch_size]
            )

        return len(points)

    def hybrid_search(
        self,
        query_text: str,
        dense_query_embedding: List[float],
        fetch_k: int = 20,
        return_k: int = 4,
    ) -> List[RetrievalResult]:
        """Perform Hybrid Search using Qdrant's native Reciprocal Rank Fusion (RRF)."""
        sparse_query = self.sparse_embeddings.embed_query(query_text)

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query_embedding, using="dense", limit=fetch_k
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query["indices"], values=sparse_query["values"]
                    ),
                    using="sparse",
                    limit=fetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=return_k,
            with_payload=True,
        )

        retrieval_results = []
        for point in results.points:
            payload = point.payload or {}
            content = payload.pop("content", "")
            retrieval_results.append(
                RetrievalResult(
                    content=content, score=point.score, metadata=payload  # RRF score
                )
            )

        return retrieval_results

    def count(self) -> int:
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:
            return 0

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
