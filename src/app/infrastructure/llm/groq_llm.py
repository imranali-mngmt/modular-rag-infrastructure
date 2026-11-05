from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import Settings
from app.domain.interfaces.llm_interface import BaseLLM, Message


class GroqLLM(BaseLLM):
    """Groq provides ultra-fast inference for Llama and Mixtral models."""

    def __init__(self, settings: Settings):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.1
        )

    def generate(self, messages: list[Message]) -> str:
        lc_messages = []
        for msg in messages:
            role = msg["role"].lower()
            content = msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role in ["human", "user"]:
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append({"role": role, "content": content})

        response = self.llm.invoke(lc_messages)
        return str(response.content)
