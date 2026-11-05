from guardrails import Guard
from guardrails.hub import DetectPromptInjection
from guardrails.validators import IsProfanity, DetectPII


class GuardrailService:
    """Professional-grade guardrails using Guardrails AI Hub."""

    def __init__(self):
        # 🛡️ 1. INPUT GUARD: Detect Prompt Injection & Jailbreaks
        self.input_guard = Guard(
            name="input_guard",
            validators=[DetectPromptInjection(on_fail="exception", threshold=0.5)],
        )

        # 🛡️ 2. OUTPUT GUARD: Detect PII and Toxic Language
        self.output_guard = Guard(
            name="output_guard",
            validators=[
                DetectPII(
                    on_fail="fix", pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"]
                ),
                IsProfanity(on_fail="fix"),
            ],
        )

    def validate_input(self, query: str) -> tuple[bool, str]:
        """Validates the user's input before sending it to the LLM."""
        try:
            self.input_guard.validate(query)
            return True, "Input is safe."
        except Exception as e:
            error_msg = str(e).lower()
            if "injection" in error_msg or "jailbreak" in error_msg:
                return (
                    False,
                    "🚫 Security Block: Prompt injection or jailbreak attempt detected.",
                )
            return False, f"🚫 Security Block: {str(e)}"

    def validate_output(self, text: str) -> tuple[bool, str]:
        """Validates the LLM's output before showing it to the user."""
        try:
            validated_result = self.output_guard.validate(text)
            return True, validated_result.validated_output
        except Exception as e:
            return False, f"🚫 Output blocked for safety: {str(e)}"
