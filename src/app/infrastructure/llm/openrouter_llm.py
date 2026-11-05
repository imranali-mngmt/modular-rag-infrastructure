from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import Settings
from app.domain.interfaces.llm_interface import BaseLLM, Message


class OpenRouterLLM(BaseLLM):
    """OpenRouter provides access to multiple LLM models through a unified API."""

    def __init__(self, settings: Settings):
        # OpenRouter uses OpenAI-compatible API with custom base URL
        self.llm = ChatOpenAI(
            model=settings.OPENROUTER_MODEL,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
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
