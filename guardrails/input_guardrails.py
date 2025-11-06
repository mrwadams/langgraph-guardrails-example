"""
Input guardrails for validating and filtering user inputs before processing.

These guardrails are typically used as entry points or early nodes in your graph
to validate inputs before they reach your agent.
"""

from typing import List, Set
from guardrails.base import BaseGuardrail, GuardrailResult, GuardrailAction
from guardrails.utils import (
    contains_profanity,
    check_topic_match,
    count_tokens_approximate,
    truncate_to_length,
)


class TopicValidationGuardrail(BaseGuardrail):
    """
    Validates that input is about allowed topics.

    Example use case: Restrict a chatbot to only answer questions about specific domains.

    Example:
        guardrail = TopicValidationGuardrail(
            allowed_topics=["weather", "climate", "temperature"],
            fuzzy_match=True
        )
    """

    def __init__(
        self,
        allowed_topics: List[str],
        fuzzy_match: bool = True,
        custom_rejection_message: str = None,
        name: str = None,
    ):
        super().__init__(name=name or "TopicValidation")
        self.allowed_topics = allowed_topics
        self.fuzzy_match = fuzzy_match
        self.custom_rejection_message = custom_rejection_message

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check if content matches allowed topics"""
        is_match, matched_topic = check_topic_match(
            content, self.allowed_topics, fuzzy=self.fuzzy_match
        )

        if is_match:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason=f"Topic matches: {matched_topic}",
                metadata={"matched_topic": matched_topic},
            )
        else:
            message = self.custom_rejection_message or (
                f"I can only answer questions about: {', '.join(self.allowed_topics)}"
            )
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=message,
                metadata={"allowed_topics": self.allowed_topics},
            )


class ProfanityFilterGuardrail(BaseGuardrail):
    """
    Filters out inputs containing profanity or inappropriate language.

    Example:
        guardrail = ProfanityFilterGuardrail(
            custom_words={"custom_bad_word", "another_bad_word"}
        )
    """

    def __init__(
        self,
        custom_words: Set[str] = None,
        strict_mode: bool = True,
        name: str = None,
    ):
        super().__init__(name=name or "ProfanityFilter")
        self.custom_words = custom_words
        self.strict_mode = strict_mode

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check for profanity in content"""
        has_profanity, found_words = contains_profanity(content, self.custom_words)

        if has_profanity:
            if self.strict_mode:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Input contains inappropriate language",
                    metadata={"found_words_count": len(found_words)},
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    reason="Input may contain inappropriate language",
                    metadata={"found_words_count": len(found_words)},
                )
        else:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="No profanity detected",
            )


class InputLengthGuardrail(BaseGuardrail):
    """
    Enforces minimum and maximum input length constraints.

    Can either block or automatically truncate inputs that are too long.

    Example:
        # Block long inputs
        guardrail = InputLengthGuardrail(max_length=1000, truncate=False)

        # Auto-truncate long inputs
        guardrail = InputLengthGuardrail(max_length=1000, truncate=True)
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 10000,
        truncate: bool = False,
        count_by: str = "characters",  # "characters" or "tokens"
        name: str = None,
    ):
        super().__init__(name=name or "InputLength")
        self.min_length = min_length
        self.max_length = max_length
        self.truncate = truncate
        self.count_by = count_by

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate input length"""
        if self.count_by == "tokens":
            length = count_tokens_approximate(content)
        else:
            length = len(content)

        # Check minimum length
        if length < self.min_length:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Input too short (minimum {self.min_length} {self.count_by})",
                metadata={"length": length, "min_length": self.min_length},
            )

        # Check maximum length
        if length > self.max_length:
            if self.truncate:
                # Truncate and continue
                truncated = truncate_to_length(content, self.max_length)
                return GuardrailResult(
                    action=GuardrailAction.MODIFY,
                    reason=f"Input truncated to {self.max_length} {self.count_by}",
                    modified_content=truncated,
                    metadata={"original_length": length, "truncated_length": self.max_length},
                )
            else:
                # Block
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Input too long (maximum {self.max_length} {self.count_by})",
                    metadata={"length": length, "max_length": self.max_length},
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="Input length within limits",
            metadata={"length": length},
        )


class RateLimitGuardrail(BaseGuardrail):
    """
    Rate limits requests per user/session.

    Example:
        guardrail = RateLimitGuardrail(
            max_requests_per_minute=10,
            max_requests_per_hour=100
        )
    """

    def __init__(
        self,
        max_requests_per_minute: int = None,
        max_requests_per_hour: int = None,
        name: str = None,
    ):
        super().__init__(name=name or "RateLimit")
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour

        # Simple in-memory storage (for production, use Redis or similar)
        self.request_history = {}

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check rate limits"""
        import time
        from collections import deque

        # Get user identifier from metadata
        user_id = (metadata or {}).get("user_id", "anonymous")

        if user_id not in self.request_history:
            self.request_history[user_id] = deque()

        current_time = time.time()
        history = self.request_history[user_id]

        # Clean old entries
        while history and current_time - history[0] > 3600:  # 1 hour
            history.popleft()

        # Check hourly limit
        if self.max_requests_per_hour and len(history) >= self.max_requests_per_hour:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Rate limit exceeded: {self.max_requests_per_hour} requests per hour",
                metadata={"user_id": user_id, "requests_count": len(history)},
            )

        # Check per-minute limit
        if self.max_requests_per_minute:
            recent_requests = sum(1 for t in history if current_time - t < 60)
            if recent_requests >= self.max_requests_per_minute:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Rate limit exceeded: {self.max_requests_per_minute} requests per minute",
                    metadata={"user_id": user_id, "requests_last_minute": recent_requests},
                )

        # Add current request
        history.append(current_time)

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="Rate limit check passed",
            metadata={"user_id": user_id},
        )


class InputFormatGuardrail(BaseGuardrail):
    """
    Validates input format using regex patterns.

    Example:
        # Ensure input is a valid email
        guardrail = InputFormatGuardrail(
            pattern=r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',
            format_name="email"
        )
    """

    def __init__(
        self,
        pattern: str,
        format_name: str = "input",
        invert: bool = False,
        name: str = None,
    ):
        super().__init__(name=name or "InputFormat")
        self.pattern = pattern
        self.format_name = format_name
        self.invert = invert

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate input format against pattern"""
        import re

        matches = bool(re.match(self.pattern, content, re.DOTALL))

        # Invert logic if needed (useful for blocking patterns)
        if self.invert:
            matches = not matches

        if matches:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason=f"Input matches expected {self.format_name} format",
            )
        else:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Input does not match expected {self.format_name} format",
                metadata={"expected_format": self.format_name},
            )


class LanguageDetectionGuardrail(BaseGuardrail):
    """
    Validates that input is in an allowed language.

    Note: Requires langdetect library for production use.
    This is a simplified implementation.

    Example:
        guardrail = LanguageDetectionGuardrail(
            allowed_languages=["en", "es", "fr"]
        )
    """

    def __init__(
        self,
        allowed_languages: List[str],
        name: str = None,
    ):
        super().__init__(name=name or "LanguageDetection")
        self.allowed_languages = allowed_languages

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Detect and validate language"""
        try:
            # Try to import langdetect (optional dependency)
            from langdetect import detect

            detected_lang = detect(content)

            if detected_lang in self.allowed_languages:
                return GuardrailResult(
                    action=GuardrailAction.ALLOW,
                    reason=f"Detected language: {detected_lang}",
                    metadata={"detected_language": detected_lang},
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Language {detected_lang} not supported. Supported: {', '.join(self.allowed_languages)}",
                    metadata={"detected_language": detected_lang},
                )
        except ImportError:
            # If langdetect not installed, allow by default
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="Language detection not available (langdetect not installed)",
            )
        except Exception as e:
            # On any error, allow by default (fail open)
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason=f"Language detection failed: {str(e)}",
            )
