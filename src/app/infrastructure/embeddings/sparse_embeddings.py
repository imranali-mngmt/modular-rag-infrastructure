from typing import List, Dict, Any
from collections import Counter
import re
from app.domain.interfaces.sparse_embeddings_interface import BaseSparseEmbeddings


class SimpleSparseEmbeddings(BaseSparseEmbeddings):

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def embed_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            tokens = self._tokenize(text)
            counts = Counter(tokens)

            # Aggregate values to guarantee unique indices even if hash collisions occur
            index_map = {}
            for token, count in counts.items():
                # Qdrant requires uint32 indices. 2**31 - 1 is the max positive 32-bit signed int.
                idx = abs(hash(token)) % 2147483647
                index_map[idx] = index_map.get(idx, 0.0) + float(count)

            # Sort by index (Qdrant strict requirement)
            sorted_items = sorted(index_map.items())
            indices = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

            results.append({"indices": indices, "values": values})
        return results

    def embed_query(self, text: str) -> Dict[str, Any]:
        tokens = self._tokenize(text)
        counts = Counter(tokens)

        index_map = {}
        for token, count in counts.items():
            idx = abs(hash(token)) % 2147483647
            index_map[idx] = index_map.get(idx, 0.0) + float(count)

        sorted_items = sorted(index_map.items())
        indices = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        return {"indices": indices, "values": values}
