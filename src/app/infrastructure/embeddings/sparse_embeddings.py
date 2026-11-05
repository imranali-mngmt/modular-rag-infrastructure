from typing import List, Dict, Any
from collections import Counter
import re
from app.domain.interfaces.sparse_embeddings_interface import BaseSparseEmbeddings


class SimpleSparseEmbeddings(BaseSparseEmbeddings):
    """
    A lightweight, NO-DOWNLOAD sparse embedding based on term frequency.
    Replaces FastEmbed to avoid downloading the ~100MB Splade model.
    """

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            tokens = self._tokenize(text)
            counts = Counter(tokens)
            unique_tokens = list(counts.keys())

            # Create hash-based indices and sort them (Qdrant requires sorted indices)
            token_indices = [
                (abs(hash(t)) % 50000, float(counts[t])) for t in unique_tokens
            ]
            token_indices.sort(key=lambda x: x[0])

            indices = [item[0] for item in token_indices]
            values = [item[1] for item in token_indices]
            results.append({"indices": indices, "values": values})
        return results

    def embed_query(self, text: str) -> Dict[str, Any]:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        unique_tokens = list(counts.keys())

        token_indices = [
            (abs(hash(t)) % 50000, float(counts[t])) for t in unique_tokens
        ]
        token_indices.sort(key=lambda x: x[0])

        indices = [item[0] for item in token_indices]
        values = [item[1] for item in token_indices]
        return {"indices": indices, "values": values}
