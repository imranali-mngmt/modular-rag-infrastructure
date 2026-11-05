from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """A document with content and metadata."""

    content: str
    metadata: dict

    def __repr__(self) -> str:
        source = self.metadata.get("source", "unknown")
        return f"Document(source={source}, length={len(self.content)})"


class BaseDocumentLoader(ABC):
    """Abstract interface for document loaders."""

    @abstractmethod
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the given file."""
        pass

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """Load documents from a file."""
        pass
