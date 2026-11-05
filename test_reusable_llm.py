"""
Main entry point for the application.
Demonstrates the clean architecture: wiring dependencies and executing workflows.
"""

import sys
from pathlib import Path

# Ensure the src folder is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.api.dependencies.providers import get_llm
from app.core.config import settings
from app.core.exceptions import ConfigurationError, LLMProviderError
from app.langchain.prompts.builders import build_test_assistant_prompt


def main() -> None:
    print("=" * 60)
    print(f"  LLM Test")
    print(f"  Provider: {settings.LLM_PROVIDER.upper()}")  # <-- SHOW PROVIDER
    print(f"  Mocks: {settings.USE_MOCKS}")
    print("=" * 60)

    try:
        # 1. Get the LLM instance (Dependency Injection)
        llm = get_llm()

        # 2. Build the structured prompt (Separation of Concerns)
        user_question = "Hello! This is a test of the reusable architecture. Reply with exactly: 'Success! The reusable LLM is working.'"
        messages = build_test_assistant_prompt(user_input=user_question)

        # 3. Execute and print
        print("\n📤 Sending structured messages to LLM...\n")
        response = llm.generate(messages)

        print("🤖 LLM Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print("\n✅ Test completed successfully!")

    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print(
            "\n💡 Please check your .env file and ensure all required settings are present."
        )
        sys.exit(1)

    except LLMProviderError as e:
        print(f"\n❌ LLM Provider Error: {e}")
        print(
            "\n💡 This is likely an issue with the LLM initialization or API connection."
        )
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print("\n💡 An unexpected error occurred. Please check the logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
