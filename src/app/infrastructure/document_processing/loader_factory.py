from typing import List
from pathlib import Path
from app.domain.interfaces.document_loader_interface import BaseDocumentLoader, Document
from app.infrastructure.document_processing.pdf_loader import PDFLoader
from app.infrastructure.document_processing.text_loader import TextLoader

# Registry of all available loaders
LOADER_REGISTRY: List[BaseDocumentLoader] = [
    PDFLoader(),
    TextLoader(),
]


def load_file(file_path: str) -> List[Document]:
    """Load a single file using the appropriate loader."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.is_dir():
        raise ValueError(
            f"Expected a file, got directory: {file_path}. Use load_folder() instead."
        )

    for loader in LOADER_REGISTRY:
        if loader.can_load(str(path)):
            return loader.load(str(path))

    raise ValueError(
        f"No loader available for file type: {path.suffix}. "
        f"Supported types: pdf, txt, md"
    )


def load_folder(folder_path: str) -> List[Document]:
    """Load all supported files from a folder recursively."""
    path = Path(folder_path)

    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not path.is_dir():
        raise ValueError(f"Expected a directory, got file: {folder_path}")

    all_documents = []
    supported_extensions = set()

    for loader in LOADER_REGISTRY:
        supported_extensions.update(loader.SUPPORTED_EXTENSIONS)

    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                docs = load_file(str(file_path))
                all_documents.extend(docs)
                print(f"  ✅ Loaded: {file_path.name} ({len(docs)} pages/sections)")
            except Exception as e:
                print(f"  ⚠️  Skipped {file_path.name}: {e}")

    return all_documents
