"""
Interactive RAG CLI - Ingest documents and query them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.api.dependencies.providers import (
    get_indexing_service,
    get_retrieval_service,
    get_vector_store,
    get_generator_service,
)
from app.core.config import settings
from app.core.exceptions import ConfigurationError, LLMProviderError


def print_banner():
    print("=" * 70)
    print("  📚 Interactive RAG System")
    print(
        f"  LLM: {settings.LLM_PROVIDER.upper()} | Embeddings: {settings.EMBEDDING_PROVIDER.upper()}"
    )
    print(
        f"  Vector Store: Qdrant ({'in-memory' if settings.QDRANT_USE_IN_MEMORY else settings.QDRANT_URL})"
    )
    print("=" * 70)


def print_help():
    print("\n📖 Available Commands:")
    print("  /ingest <path>     - Ingest a file or folder (PDF, TXT, MD)")
    print("  /list              - Show statistics about indexed documents")
    print("  /clear             - Clear all indexed documents")
    print("  /help              - Show this help message")
    print("  /exit              - Exit the application")
    print()


def handle_ingest(args: str, indexing_service):
    """Handle /ingest command."""
    if not args:
        print("❌ Usage: /ingest <path_to_file_or_folder>")
        return

    path = Path(args.strip())

    if not path.exists():
        print(f"❌ Path not found: {args}")
        return

    try:
        if path.is_file():
            result = indexing_service.ingest_file(str(path))
        else:
            result = indexing_service.ingest_folder(str(path))

        print(f"\n📊 Indexing Summary:")
        print(f"   Documents loaded: {result.documents_loaded}")
        print(f"   Chunks created:   {result.chunks_created}")
        print(f"   Chunks indexed:   {result.chunks_indexed}")
        print(f"   Total in store:   {result.total_in_store}")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")


import traceback
from pathlib import Path

# ... (keep your other imports)


def handle_query(args: str, retrieval_service, generator_service):
    """Handle direct questions or /query command."""
    if not args:
        print("❌ Please provide a question.")
        return

    query = args.strip()
    print(f"\n🔍 Searching for: '{query}'\n")

    try:
        # 1. Retrieve relevant chunks
        response = retrieval_service.retrieve(query)

        if not response.results:
            print(
                "⚠️  No relevant documents found. Try ingesting some documents first!\n"
            )
            return

        print(
            f"📄 Found {len(response.results)} relevant chunk(s). Generating answer...\n"
        )
        print("-" * 70)

        # 2. Generate answer using LLM
        answer = generator_service.generate_answer(query, response.results)

        print("💡 GENERATED ANSWER:")
        print(answer)
        print("-" * 70)

        # 3. Print sources below the answer (Robust to dict or object)
        print("\n📚 SOURCES:")
        for i, result in enumerate(response.results, 1):
            if isinstance(result, dict):
                meta = result.get("metadata", {})
                source = meta.get("source", "unknown")
                page = meta.get("page", "?")
            else:
                meta = getattr(result, "metadata", {})
                source = (
                    meta.get("source", "unknown")
                    if isinstance(meta, dict)
                    else "unknown"
                )
                page = meta.get("page", "?") if isinstance(meta, dict) else "?"

            # Safely handle Path extraction
            try:
                source_name = Path(source).name
            except Exception:
                source_name = str(source)

            print(f"  [{i}] {source_name} (Page: {page})")
        print()

    except Exception as e:
        print(f"\n❌ Error during retrieval/generation:")
        traceback.print_exc()  # This will print the exact line causing the error


def handle_list(vector_store):
    """Handle /list command."""
    count = vector_store.count()
    print(f"\n📊 Vector Store Statistics:")
    print(f"   Collection: {settings.QDRANT_COLLECTION_NAME}")
    print(f"   Total chunks: {count}")
    print()


def handle_clear(vector_store):
    """Handle /clear command."""
    confirm = (
        input("⚠️  Are you sure you want to clear all documents? (y/n): ")
        .strip()
        .lower()
    )
    if confirm == "y":
        try:
            vector_store.clear()
            print("Properly worked All documents cleared.")
        except Exception as e:
            print(f"❌ Error clearing: {e}")
    else:
        print("Cancelled.")


# ... (keep imports and helper functions the same)


def main():
    print_banner()

    try:
        indexing_service = get_indexing_service()
        retrieval_service = get_retrieval_service()
        vector_store = get_vector_store()
        generator_service = get_generator_service()
    except (ConfigurationError, LLMProviderError) as e:
        print(f"\n❌ Initialization Error: {e}")
        sys.exit(1)

    print_help()

    while True:
        try:
            user_input = input("\n🤖 RAG> ").strip()

            if not user_input:
                continue

            # CHANGE THIS BLOCK:
            if not user_input.startswith("/"):
                # Treat direct input as a query
                handle_query(user_input, retrieval_service, generator_service)
                continue

            # Parse command and arguments
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "/exit" or command == "/quit":
                print("\n👋 Goodbye!")
                break
            elif command == "/help":
                print_help()
            elif command == "/ingest":
                handle_ingest(args, indexing_service)
            elif command == "/list":
                handle_list(vector_store)
            elif command == "/clear":
                handle_clear(vector_store)
            else:
                print(
                    f"❌ Unknown command: {command}. Type /help for available commands."
                )

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
