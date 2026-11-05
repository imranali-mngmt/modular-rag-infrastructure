from typing import List, Dict, Any
from fastembed import SparseTextEmbedding
from app.domain.interfaces.sparse_embeddings_interface import BaseSparseEmbeddings


class FastEmbedSparseEmbeddings(BaseSparseEmbeddings):
    """Uses FastEmbed's Splade model for local, high-quality sparse embeddings."""

    def __init__(self):
        # Downloads the model on first run (~100MB), then runs locally
        self.model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        sparse_embeddings = list(self.model.embed(texts))
        return [
            {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
            for emb in sparse_embeddings
        ]

    def embed_query(self, text: str) -> Dict[str, Any]:
        # Query embedding requires query=True for Splade
        emb = list(self.model.query_embed(text))[0]
        return {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
