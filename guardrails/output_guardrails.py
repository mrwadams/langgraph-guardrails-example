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
    calculate_text_similarity,
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


class FactualityGuardrail(BaseGuardrail):
    """
    Validates output against known facts or reference data.

    Example:
        guardrail = FactualityGuardrail(
            reference_data={"capital_of_france": "Paris"},
            check_function=custom_fact_checker
        )
    """

    def __init__(
        self,
        reference_data: Dict = None,
        check_function: Callable = None,
        similarity_threshold: float = 0.8,
        name: str = None,
    ):
        super().__init__(name=name or "Factuality")
        self.reference_data = reference_data or {}
        self.check_function = check_function
        self.similarity_threshold = similarity_threshold

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate factual accuracy"""
        # If custom check function provided, use it
        if self.check_function:
            try:
                result = self.check_function(content, metadata)
                if isinstance(result, bool):
                    if result:
                        return GuardrailResult(
                            action=GuardrailAction.ALLOW,
                            reason="Factuality check passed",
                        )
                    else:
                        return GuardrailResult(
                            action=GuardrailAction.BLOCK,
                            reason="Factuality check failed",
                        )
                # If function returns GuardrailResult, use it directly
                elif isinstance(result, GuardrailResult):
                    return result
            except Exception as e:
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    reason=f"Factuality check error: {str(e)}",
                )

        # Simple reference data checking
        if self.reference_data and metadata and "query" in metadata:
            query = metadata["query"].lower()

            for key, expected_value in self.reference_data.items():
                if key.lower() in query:
                    # Check if expected value is in the output
                    similarity = calculate_text_similarity(content, expected_value)

                    if similarity < self.similarity_threshold:
                        return GuardrailResult(
                            action=GuardrailAction.BLOCK,
                            reason=f"Output does not match expected fact for '{key}'",
                            metadata={
                                "expected": expected_value,
                                "similarity": similarity,
                            },
                        )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="No factuality issues detected",
        )

    def get_content_to_validate(self, state: dict) -> str:
        """Extract output content from state"""
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


class NoHallucinationGuardrail(BaseGuardrail):
    """
    Detects potential hallucinations in output.

    This is a simplified implementation. For production, consider using:
    - LLM-based verification
    - External knowledge base checking
    - Citation verification

    Example:
        guardrail = NoHallucinationGuardrail(
            confidence_threshold=0.7,
            require_citations=True
        )
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        require_citations: bool = False,
        name: str = None,
    ):
        super().__init__(name=name or "NoHallucination")
        self.confidence_threshold = confidence_threshold
        self.require_citations = require_citations

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check for potential hallucinations"""
        # Check if confidence score is provided in metadata
        confidence = (metadata or {}).get("confidence", 1.0)

        if confidence < self.confidence_threshold:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Low confidence output (confidence: {confidence})",
                metadata={"confidence": confidence},
            )

        # Check for citation requirements
        if self.require_citations:
            has_citations = (
                "[" in content and "]" in content
            ) or "source:" in content.lower()

            if not has_citations:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Output requires citations but none were provided",
                )

        # Check for uncertainty markers (hedge words)
        uncertainty_markers = ["maybe", "perhaps", "might", "possibly", "i think"]
        has_uncertainty = any(marker in content.lower() for marker in uncertainty_markers)

        if has_uncertainty:
            return GuardrailResult(
                action=GuardrailAction.WARN,
                reason="Output contains uncertainty markers",
                metadata={"has_uncertainty": True},
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="No hallucination indicators detected",
        )

    def get_content_to_validate(self, state: dict) -> str:
        """Extract output content from state"""
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "")
            return str(last_message)
        return state.get("output", state.get("content", ""))


class SensitiveContentFilterGuardrail(BaseGuardrail):
    """
    Filters sensitive content from outputs.

    Example:
        guardrail = SensitiveContentFilterGuardrail(
            blocked_patterns=["password", "api_key", "secret"],
            redact=True
        )
    """

    def __init__(
        self,
        blocked_patterns: List[str] = None,
        redact: bool = True,
        name: str = None,
    ):
        super().__init__(name=name or "SensitiveContentFilter")
        self.blocked_patterns = blocked_patterns or []
        self.redact = redact

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check for sensitive content"""
        import re

        found_patterns = []

        for pattern in self.blocked_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_patterns.append(pattern)

        if found_patterns:
            if self.redact:
                # Redact the sensitive content
                redacted = content
                for pattern in found_patterns:
                    redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    reason=f"Sensitive content redacted: {', '.join(found_patterns)}",
                    modified_content=redacted,
                    metadata={"redacted_patterns": found_patterns},
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Output contains sensitive content: {', '.join(found_patterns)}",
                    metadata={"found_patterns": found_patterns},
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="No sensitive content detected",
        )

    def get_content_to_validate(self, state: dict) -> str:
        """Extract output content from state"""
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "")
            return str(last_message)
        return state.get("output", state.get("content", ""))
