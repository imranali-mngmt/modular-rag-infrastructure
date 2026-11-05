from typing import List
from dataclasses import dataclass
from app.domain.interfaces.document_loader_interface import Document
from app.domain.interfaces.vectorstore_interface import BaseVectorStore
from app.domain.interfaces.embeddings_interface import BaseEmbeddings
from app.infrastructure.document_processing.loader_factory import load_file, load_folder
from app.infrastructure.document_processing.text_splitter import DocumentSplitter
from app.core.config import Settings


@dataclass
class IndexingResult:
    """Result of an indexing operation."""

    documents_loaded: int
    chunks_created: int
    chunks_indexed: int
    total_in_store: int


class IndexingService:
    """Service for indexing documents into the vector store."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        settings: Settings,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.splitter = DocumentSplitter(settings)

    def ingest_file(self, file_path: str) -> IndexingResult:
        """Ingest a single file."""
        print(f"\n📄 Loading file: {file_path}")
        documents = load_file(file_path)
        return self._process_documents(documents)

    def ingest_folder(self, folder_path: str) -> IndexingResult:
        """Ingest all files in a folder."""
        print(f"\n📁 Loading folder: {folder_path}")
        documents = load_folder(folder_path)
        return self._process_documents(documents)

    def _process_documents(self, documents: List[Document]) -> IndexingResult:
        """Process documents: split, embed, and store."""
        if not documents:
            print("⚠️  No documents to process")
            return IndexingResult(0, 0, 0, self.vector_store.count())

        print(f"📚 Loaded {len(documents)} document(s)")

        # Split into chunks
        chunks = self.splitter.split_documents(documents)
        print(f"🔪 Split into {len(chunks)} chunks")

        if not chunks:
            return IndexingResult(len(documents), 0, 0, self.vector_store.count())

        # Extract texts and metadata
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Embed in batches
        print(f"🧠 Embedding {len(texts)} chunks...")
        embeddings = self.embeddings.embed_texts(texts)

        # Store in vector store
        print(f"💾 Storing in vector store...")
        indexed = self.vector_store.add_documents(texts, embeddings, metadatas)

        total = self.vector_store.count()
        print(f"✅ Indexed {indexed} chunks (total in store: {total})")

        return IndexingResult(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            chunks_indexed=indexed,
            total_in_store=total,
        )
