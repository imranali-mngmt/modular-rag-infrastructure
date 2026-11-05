from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings


class GeneratorService:
    """Service for generating answers using the LLM."""

    def __init__(self):
        model_name = getattr(settings, "GOOGLE_MODEL", "gemini-1.5-flash")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )

    def generate_answer(self, query: str, context_results: List[Any]) -> str:
        if not context_results:
            return "I couldn't find any relevant information in the documents to answer your question."

        # Prepare context
        contents = [
            str(
                r.get("content", "")
                if isinstance(r, dict)
                else getattr(r, "content", "")
            )
            for r in context_results
        ]
        context_text = "\n\n---\n\n".join(contents)

        prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately.
If the answer is not explicitly stated in the context, say "I don't have enough information in the provided documents to answer that."

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            return f"Error generating answer: {e}"
