"""
Base classes and abstractions for LangGraph guardrails.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Callable
from enum import Enum


class GuardrailAction(str, Enum):
    """Actions a guardrail can take"""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    WARN = "warn"


@dataclass
class GuardrailResult:
    """Result from a guardrail check"""
    action: GuardrailAction
    reason: str = ""
    modified_content: Any = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def passed(self) -> bool:
        """Whether the guardrail check passed"""
        return self.action == GuardrailAction.ALLOW

    @property
    def blocked(self) -> bool:
        """Whether the guardrail blocked the content"""
        return self.action == GuardrailAction.BLOCK


class BaseGuardrail(ABC):
    """
    Abstract base class for all guardrails.

    Guardrails in LangGraph are implemented as nodes that validate/transform state.
    This base class provides a common interface for:
    - Checking content (check method)
    - Routing based on result (route method)
    - Creating response nodes (rejection_node method)
    """

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.metrics = {
            "total_checks": 0,
            "blocked": 0,
            "allowed": 0,
            "modified": 0,
        }

    @abstractmethod
    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """
        Validate content and return a GuardrailResult.

        Args:
            content: The content to validate
            metadata: Optional metadata about the content/context

        Returns:
            GuardrailResult indicating the action to take
        """
        pass

    def check(self, state: dict) -> dict:
        """
        Node function that checks content and updates state.

        This is the primary method used as a LangGraph node.

        Args:
            state: The LangGraph state dict

        Returns:
            Updated state with guardrail results
        """
        self.metrics["total_checks"] += 1

        # Extract content to validate (customizable via get_content_to_validate)
        content = self.get_content_to_validate(state)
        metadata = self.get_metadata(state)

        # Run validation
        result = self.validate(content, metadata)

        # Update metrics
        if result.blocked:
            self.metrics["blocked"] += 1
        elif result.action == GuardrailAction.MODIFY:
            self.metrics["modified"] += 1
        else:
            self.metrics["allowed"] += 1

        # Update state with results
        return self.update_state_with_result(state, result)

    def get_content_to_validate(self, state: dict) -> str:
        """
        Extract content to validate from state.
        Override this for custom content extraction logic.

        Args:
            state: The LangGraph state dict

        Returns:
            Content string to validate
        """
        # Default: look for 'input', 'messages', or 'content' keys
        if "input" in state:
            return str(state["input"])
        elif "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict):
                return last_message.get("content", "")
            return str(last_message)
        elif "content" in state:
            return str(state["content"])
        return ""

    def get_metadata(self, state: dict) -> dict:
        """
        Extract metadata from state for validation context.
        Override this for custom metadata extraction.

        Args:
            state: The LangGraph state dict

        Returns:
            Metadata dict
        """
        return state.get("metadata", {})

    def update_state_with_result(self, state: dict, result: GuardrailResult) -> dict:
        """
        Update state with guardrail result.
        Override this for custom state update logic.

        Args:
            state: The current state
            result: The guardrail result

        Returns:
            Updated state dict
        """
        updates = {
            f"{self.name}_result": result.action.value,
            f"{self.name}_reason": result.reason,
            f"{self.name}_metadata": result.metadata,
        }

        # If content was modified, update the appropriate field
        if result.action == GuardrailAction.MODIFY and result.modified_content is not None:
            if "input" in state:
                updates["input"] = result.modified_content
            elif "content" in state:
                updates["content"] = result.modified_content

        return {**state, **updates}

    def route(self, state: dict) -> str:
        """
        Routing function for conditional edges.

        Returns the next node based on the guardrail result.
        Override this for custom routing logic.

        Args:
            state: The current state

        Returns:
            Name of next node to route to
        """
        result_key = f"{self.name}_result"
        result = state.get(result_key)

        if result == GuardrailAction.BLOCK.value:
            return "reject"
        elif result == GuardrailAction.ALLOW.value:
            return "continue"
        elif result == GuardrailAction.MODIFY.value:
            return "continue"  # Continue with modified content
        else:
            return "continue"  # Default to continue

    def rejection_node(self, state: dict) -> dict:
        """
        Node that handles rejected content.
        Returns a user-friendly rejection message.

        Args:
            state: The current state

        Returns:
            Updated state with rejection message
        """
        reason = state.get(f"{self.name}_reason", "Content did not pass validation")

        return {
            **state,
            "messages": [
                {
                    "role": "assistant",
                    "content": f"I cannot process this request. Reason: {reason}"
                }
            ],
            "rejected": True,
        }

    def get_metrics(self) -> dict:
        """Get guardrail metrics for monitoring"""
        return {
            **self.metrics,
            "block_rate": self.metrics["blocked"] / max(self.metrics["total_checks"], 1),
            "pass_rate": self.metrics["allowed"] / max(self.metrics["total_checks"], 1),
        }


class GuardrailChain:
    """
    Chain multiple guardrails together.

    Executes guardrails in sequence, stopping at the first one that blocks.
    Useful for layering multiple validation checks.

    Example:
        chain = GuardrailChain([
            InputLengthGuardrail(max_length=1000),
            RateLimitGuardrail(max_requests_per_hour=100),
            PIIDetectionGuardrail(redact=True),
        ])
    """

    def __init__(self, guardrails: list[BaseGuardrail]):
        self.guardrails = guardrails
        self.name = "GuardrailChain"

    def check(self, state: dict) -> dict:
        """
        Run all guardrails in sequence.

        Args:
            state: The LangGraph state dict

        Returns:
            Updated state after all guardrails
        """
        for guardrail in self.guardrails:
            state = guardrail.check(state)

            # Stop if any guardrail blocks
            result_key = f"{guardrail.name}_result"
            if state.get(result_key) == GuardrailAction.BLOCK.value:
                break

        return state

    def route(self, state: dict) -> str:
        """
        Route based on combined guardrail results.
        Blocks if any guardrail blocked.

        Args:
            state: The current state

        Returns:
            Next node name
        """
        # Check if any guardrail blocked
        for guardrail in self.guardrails:
            result_key = f"{guardrail.name}_result"
            if state.get(result_key) == GuardrailAction.BLOCK.value:
                return "reject"

        return "continue"

    def rejection_node(self, state: dict) -> dict:
        """
        Create rejection message from the first guardrail that blocked.

        Args:
            state: The current state

        Returns:
            State with rejection message
        """
        for guardrail in self.guardrails:
            result_key = f"{guardrail.name}_result"
            if state.get(result_key) == GuardrailAction.BLOCK.value:
                return guardrail.rejection_node(state)

        # Fallback
        return {
            **state,
            "messages": [{"role": "assistant", "content": "Request blocked by guardrails"}],
            "rejected": True,
        }


class ConditionalGuardrail(BaseGuardrail):
    """
    A guardrail that only runs under certain conditions.

    Example:
        # Only check for PII for non-premium users
        conditional = ConditionalGuardrail(
            condition=lambda state: state.get("user_type") != "premium",
            guardrail=PIIDetectionGuardrail(),
        )
    """

    def __init__(
        self,
        condition: Callable[[dict], bool],
        guardrail: BaseGuardrail,
        name: str = None,
    ):
        super().__init__(name=name or f"Conditional_{guardrail.name}")
        self.condition = condition
        self.guardrail = guardrail

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """This method is not used; check() is overridden instead"""
        return GuardrailResult(action=GuardrailAction.ALLOW)

    def check(self, state: dict) -> dict:
        """Only run the guardrail if condition is met"""
        if self.condition(state):
            return self.guardrail.check(state)
        else:
            # Condition not met, automatically allow
            return {
                **state,
                f"{self.name}_result": GuardrailAction.ALLOW.value,
                f"{self.name}_reason": "Condition not met, skipped validation",
            }
