from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import Settings
from app.domain.interfaces.document_loader_interface import Document


class DocumentSplitter:
    """Splits documents into smaller chunks for embedding."""

    def __init__(self, settings: Settings):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.settings = settings

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks, preserving metadata."""
        all_chunks = []

        for doc in documents:
            chunks = self.splitter.split_text(doc.content)

            for i, chunk in enumerate(chunks):
                # Create new metadata with chunk info
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)

                all_chunks.append(Document(content=chunk, metadata=chunk_metadata))

        return all_chunks
