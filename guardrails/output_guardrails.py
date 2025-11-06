"""
Output guardrails for validating and filtering agent responses before returning to users.

These guardrails are typically used as final validation nodes before END in your graph
to ensure responses meet quality and safety standards.
"""

from typing import List, Dict, Callable
from guardrails.base import BaseGuardrail, GuardrailResult, GuardrailAction
from guardrails.utils import (
    extract_json_from_text,
    sanitize_html,
)


class OutputFormatGuardrail(BaseGuardrail):
    """
    Validates that output matches expected format (JSON, plain text, etc.).

    Example:
        # Ensure output is valid JSON
        guardrail = OutputFormatGuardrail(
            expected_format="json",
            required_fields=["answer", "confidence"]
        )
    """

    def __init__(
        self,
        expected_format: str = "text",  # "json", "text", "markdown"
        required_fields: List[str] = None,
        name: str = None,
    ):
        super().__init__(name=name or "OutputFormat")
        self.expected_format = expected_format
        self.required_fields = required_fields or []

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate output format"""
        if self.expected_format == "json":
            json_obj = extract_json_from_text(content)

            if json_obj is None:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Output is not valid JSON",
                )

            # Check required fields
            missing_fields = [f for f in self.required_fields if f not in json_obj]
            if missing_fields:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Missing required fields: {', '.join(missing_fields)}",
                    metadata={"missing_fields": missing_fields},
                )

            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="Output is valid JSON with all required fields",
                metadata={"parsed_json": json_obj},
            )

        elif self.expected_format == "text":
            # Simple validation - ensure it's not empty
            if not content or not content.strip():
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Output is empty",
                )

            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="Output is valid text",
            )

        elif self.expected_format == "markdown":
            # Basic markdown validation
            if not content or not content.strip():
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Output is empty",
                )

            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="Output is valid markdown",
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason=f"Format validation not implemented for {self.expected_format}",
        )

    def get_content_to_validate(self, state: dict) -> str:
        """Extract output content from state"""
        # For output guardrails, we typically validate the last message
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "")
            return str(last_message)
        return state.get("output", state.get("content", ""))


class OutputLengthGuardrail(BaseGuardrail):
    """
    Ensures output is within acceptable length bounds.

    Example:
        guardrail = OutputLengthGuardrail(
            min_length=10,
            max_length=500,
            truncate=True
        )
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = None,
        truncate: bool = False,
        count_by: str = "characters",
        name: str = None,
    ):
        super().__init__(name=name or "OutputLength")
        self.min_length = min_length
        self.max_length = max_length
        self.truncate = truncate
        self.count_by = count_by

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate output length"""
        if self.count_by == "tokens":
            from guardrails.utils import count_tokens_approximate

            length = count_tokens_approximate(content)
        else:
            length = len(content)

        # Check minimum
        if length < self.min_length:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Output too short (minimum {self.min_length} {self.count_by})",
                metadata={"length": length},
            )

        # Check maximum
        if self.max_length and length > self.max_length:
            if self.truncate:
                from guardrails.utils import truncate_to_length

                truncated = truncate_to_length(content, self.max_length)
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    reason=f"Output truncated to {self.max_length} {self.count_by}",
                    modified_content=truncated,
                    metadata={"original_length": length},
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Output too long (maximum {self.max_length} {self.count_by})",
                    metadata={"length": length},
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="Output length within bounds",
            metadata={"length": length},
        )

    def get_content_to_validate(self, state: dict) -> str:
        """Extract output content from state"""
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "")
            return str(last_message)
        return state.get("output", state.get("content", ""))
