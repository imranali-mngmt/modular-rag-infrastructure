from typing import List
from pathlib import Path
from pypdf import PdfReader
from app.domain.interfaces.document_loader_interface import BaseDocumentLoader, Document


class PDFLoader(BaseDocumentLoader):
    """Loads PDF files using pypdf."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def can_load(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents = []
        try:
            reader = PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            content=text.strip(),
                            metadata={
                                "source": str(path),
                                "page": page_num,
                                "total_pages": len(reader.pages),
                                "file_type": "pdf",
                            },
                        )
                    )
        except Exception as e:
            raise ValueError(f"Failed to load PDF {file_path}: {str(e)}") from e

        return documents
