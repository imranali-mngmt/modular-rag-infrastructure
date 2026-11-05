from typing import List
from pathlib import Path
from app.domain.interfaces.document_loader_interface import BaseDocumentLoader, Document


class TextLoader(BaseDocumentLoader):
    """Loads text files (.txt, .md, etc.)."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}

    def can_load(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                return []

            return [
                Document(
                    content=content.strip(),
                    metadata={"source": str(path), "page": 1, "file_type": "text"},
                )
            ]
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode text file {file_path}: {str(e)}") from e
