from abc import ABC, abstractmethod
from typing import List, Dict

# Define a standard message structure
Message = Dict[str, str]


class BaseLLM(ABC):
    """Abstract interface for any LLM provider."""

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate a text response from a list of structured messages."""
        pass
