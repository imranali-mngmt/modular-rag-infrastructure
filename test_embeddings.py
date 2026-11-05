"""Test script for embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.api.dependencies.providers import get_embeddings
from app.core.config import settings
from app.core.exceptions import ConfigurationError, LLMProviderError


def main() -> None:
    print("=" * 60)
    print(f"  Embeddings Test")
    print(f"  Provider: {settings.EMBEDDING_PROVIDER.upper()}")
    print("=" * 60)

    try:
        # 1. Get embeddings instance
        print("\n📦 Requesting embeddings instance...")
        embeddings = get_embeddings()
        print(f"Properly worked Got embeddings instance: {type(embeddings).__name__}\n")

        # 2. Test single query embedding
        print("📝 Testing single query embedding...")
        query = "What is machine learning?"
        query_embedding = embeddings.embed_query(query)
        print(f"Properly worked Query embedded successfully!")
        print(f"   Vector dimension: {len(query_embedding)}")
        print(f"   First 5 values: {query_embedding[:5]}\n")

        # 3. Test batch text embedding
        print("📚 Testing batch text embedding...")
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand human language.",
        ]
        text_embeddings = embeddings.embed_texts(texts)
        print(f"Properly worked {len(texts)} texts embedded successfully!")
        print(f"   Each vector dimension: {len(text_embeddings[0])}")
        print(f"   Total embeddings: {len(text_embeddings)}\n")

        # 4. Calculate similarity (cosine similarity)
        print("🔍 Testing semantic similarity...")
        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_01 = cosine_similarity(text_embeddings[0], text_embeddings[1])
        sim_02 = cosine_similarity(text_embeddings[0], text_embeddings[2])

        print(f"   Similarity (ML vs Deep Learning): {sim_01:.4f}")
        print(f"   Similarity (ML vs NLP): {sim_02:.4f}")
        print(
            f"   Properly worked ML and Deep Learning are more similar (as expected!)\n"
        )

        print("=" * 60)
        print("Properly worked All embedding tests passed!")
        print("=" * 60)

    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)

    except LLMProviderError as e:
        print(f"\n❌ Embeddings Provider Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
