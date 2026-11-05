from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import Settings
from app.domain.interfaces.embeddings_interface import BaseEmbeddings


class GoogleEmbeddingsModel(BaseEmbeddings):
    """Google Gemini embedding model."""

    def __init__(self, settings: Settings):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.GOOGLE_EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
