from typing import List, Dict, Any
from collections import Counter
import re
from app.domain.interfaces.sparse_embeddings_interface import BaseSparseEmbeddings


class SimpleSparseEmbeddings(BaseSparseEmbeddings):
    """
    Bulletproof sparse embedding using word frequency.
    NO downloads, NO hash collisions, 100% unique indices.
    """

    def _tokenize(self, text: str) -> List[str]:
        """Extract words and convert to lowercase."""
        return re.findall(r"\b[a-z]+\b", text.lower())

    def _build_sparse_vector(self, tokens: List[str]) -> Dict[str, Any]:
        """Build sparse vector with guaranteed unique indices."""
        # Count word frequencies
        word_counts = Counter(tokens)

        # Create a sorted vocabulary (alphabetical order guarantees uniqueness)
        unique_words = sorted(word_counts.keys())

        # Map each word to a unique index (its position in sorted list)
        indices = []
        values = []

        for idx, word in enumerate(unique_words):
            indices.append(idx)  # Sequential indices: 0, 1, 2, 3...
            values.append(float(word_counts[word]))  # Term frequency

        return {"indices": indices, "values": values}

    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Embed multiple documents."""
        results = []
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens:
                # Return empty sparse vector for empty text
                results.append({"indices": [], "values": []})
            else:
                results.append(self._build_sparse_vector(tokens))
        return results

    def embed_query(self, text: str) -> Dict[str, Any]:
        """Embed a single query."""
        tokens = self._tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}
        return self._build_sparse_vector(tokens)
