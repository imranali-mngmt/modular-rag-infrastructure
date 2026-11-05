from typing import List, Any
from app.domain.interfaces.llm_interface import BaseLLM


class GeneratorService:
    """Service for generating answers using an LLM based on retrieved context."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_answer(self, query: str, context_results: List[Any]) -> str:
        if not context_results:
            return "I couldn't find any relevant information in the documents to answer your question."

        # Safely extract content whether it's an object or a dictionary
        contents = []
        for r in context_results:
            if isinstance(r, dict):
                contents.append(str(r.get("content", "")))
            else:
                contents.append(str(getattr(r, "content", "")))

        context_text = "\n\n---\n\n".join(contents)

        prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately.
If the answer is not explicitly stated in the context, say "I don't have enough information in the provided documents to answer that."

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            # Handle LangChain AIMessage object or similar
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            return f"Error generating answer: {e}"
