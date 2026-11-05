from typing import List
from app.domain.interfaces.llm_interface import Message


def build_test_assistant_prompt(user_input: str) -> List[Message]:
    """
    Builds a structured prompt for testing the LLM connection.
    Separates system behavior from user input.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful, concise, and professional AI assistant. "
                "Always follow the user's formatting instructions exactly. "
                "Do not add unnecessary pleasantries or explanations."
            ),
        },
        {"role": "human", "content": user_input},
    ]
