from typing import List
from langchain_openai import OpenAIEmbeddings
from app.core.config import Settings
from app.domain.interfaces.embeddings_interface import BaseEmbeddings


class OpenAIEmbeddingsModel(BaseEmbeddings):
    """OpenAI embedding model (text-embedding-3-small or text-embedding-3-large)."""

    def __init__(self, settings: Settings):
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
