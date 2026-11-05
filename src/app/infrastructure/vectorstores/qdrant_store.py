import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import Settings
from app.domain.interfaces.vectorstore_interface import BaseVectorStore, RetrievalResult
from app.domain.interfaces.embeddings_interface import BaseEmbeddings
from app.domain.interfaces.sparse_embeddings_interface import BaseSparseEmbeddings


class QdrantHybridVectorStore(BaseVectorStore):
    """Qdrant vector store with Hybrid Search support (Dense + Sparse)."""

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

        # Initialize Qdrant client based on mode
        mode = settings.QDRANT_MODE.lower()

        if mode == "memory":
            self.client = QdrantClient(":memory:")
            print(f"📦 [Qdrant] Using IN-MEMORY storage (volatile)")
        elif mode == "local":
            self.client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
            print(
                f"📦 [Qdrant] Using LOCAL PERSISTENT storage at '{settings.QDRANT_LOCAL_PATH}'"
            )
        elif mode == "cloud":
            if not settings.QDRANT_URL:
                raise ValueError("QDRANT_URL is required for cloud mode")
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            )
            print(f"📦 [Qdrant] Connected to CLOUD instance at {settings.QDRANT_URL}")
        else:
            raise ValueError(
                f"Unknown QDRANT_MODE: {mode}. Use 'memory', 'local', or 'cloud'"
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            print(f"⚠️  Warning: Could not list collections: {e}")
            collections = []

        if self.collection_name not in collections:
            # Get sample embedding to determine dimension
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
            print(f"✅ [Qdrant] Created hybrid collection '{self.collection_name}'")
        else:
            print(f"✅ [Qdrant] Collection '{self.collection_name}' already exists")

    def add_documents(
        self,
        texts: List[str],
        dense_embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> int:
        """Add documents to the vector store."""
        if not texts:
            return 0

        # Generate sparse embeddings
        sparse_embeddings = self.sparse_embeddings.embed_documents(texts)

        points = []
        for idx, (text, dense_vec, sparse_vec, metadata) in enumerate(
            zip(texts, dense_embeddings, sparse_embeddings, metadatas)
        ):
            # Skip if sparse vector is empty
            if not sparse_vec["indices"]:
                continue

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

        # Batch upload with smaller batch size and explicit timeout for Cloud stability
        if points:
            batch_size = 32  # ✅ Reduced from 100 to prevent network timeouts
            total_points = len(points)

            for i in range(0, total_points, batch_size):
                batch = points[i : i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        timeout=60,  # ✅ Explicit 60-second timeout for cloud uploads
                    )
                    print(
                        f"   💾 Uploaded batch {i//batch_size + 1} ({len(batch)} points)"
                    )
                except Exception as e:
                    print(
                        f"   ⚠️ Warning: Failed to upload batch {i//batch_size + 1}. Retrying with smaller batch..."
                    )
                    # Fallback: try uploading one by one if the batch fails
                    for point in batch:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=[point],
                            timeout=60,
                        )

        return len(points)

    def hybrid_search(
        self,
        query_text: str,
        dense_query_embedding: List[float],
        fetch_k: int = 20,
        return_k: int = 4,
    ) -> List[RetrievalResult]:
        """Perform Hybrid Search using Qdrant's Reciprocal Rank Fusion (RRF)."""
        sparse_query = self.sparse_embeddings.embed_query(query_text)

        # If sparse query is empty, fall back to dense-only search
        if not sparse_query["indices"]:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query_embedding,
                using="dense",
                limit=return_k,
                with_payload=True,
            )
        else:
            # Hybrid search with both dense and sparse
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query_embedding, using="dense", limit=fetch_k
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query["indices"],
                            values=sparse_query["values"],
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
                RetrievalResult(content=content, score=point.score, metadata=payload)
            )

        return retrieval_results

    def count(self) -> int:
        """Get total number of points in the collection."""
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
        except Exception as e:
            print(f"⚠️  Warning: Could not clear collection: {e}")
