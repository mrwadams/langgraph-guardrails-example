"""
Safety guardrails for content moderation, PII detection, and security checks.

These guardrails can be used for both input and output validation to ensure
content meets safety and privacy standards.
"""

from typing import List, Set, Callable
from guardrails.base import BaseGuardrail, GuardrailResult, GuardrailAction
from guardrails.utils import detect_pii_patterns, redact_pii


class PIIDetectionGuardrail(BaseGuardrail):
    """
    Detects and optionally redacts Personally Identifiable Information (PII).

    Detects:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses

    Example:
        # Block content with PII
        guardrail = PIIDetectionGuardrail(redact=False, strict=True)

        # Auto-redact PII and continue
        guardrail = PIIDetectionGuardrail(redact=True, strict=False)
    """

    def __init__(
        self,
        redact: bool = True,
        strict: bool = False,
        allowed_pii_types: Set[str] = None,
        name: str = None,
    ):
        super().__init__(name=name or "PIIDetection")
        self.redact = redact
        self.strict = strict
        self.allowed_pii_types = allowed_pii_types or set()

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Detect and handle PII in content"""
        if self.redact:
            redacted_content, pii_found = redact_pii(content)

            if pii_found:
                # Filter out allowed PII types
                blocked_pii_types = set(pii_found.keys()) - self.allowed_pii_types

                if blocked_pii_types:
                    return GuardrailResult(
                        action=GuardrailAction.MODIFY,
                        reason=f"PII detected and redacted: {', '.join(blocked_pii_types)}",
                        modified_content=redacted_content,
                        metadata={"pii_types": list(blocked_pii_types), "pii_count": sum(len(v) for k, v in pii_found.items())},
                    )

            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="No PII detected or all PII types allowed",
            )
        else:
            # Detection only mode
            pii_found = detect_pii_patterns(content)

            if pii_found:
                blocked_pii_types = set(pii_found.keys()) - self.allowed_pii_types

                if blocked_pii_types:
                    if self.strict:
                        return GuardrailResult(
                            action=GuardrailAction.BLOCK,
                            reason=f"PII detected: {', '.join(blocked_pii_types)}",
                            metadata={"pii_types": list(blocked_pii_types), "pii_count": sum(len(v) for k, v in pii_found.items())},
                        )
                    else:
                        return GuardrailResult(
                            action=GuardrailAction.WARN,
                            reason=f"PII detected: {', '.join(blocked_pii_types)}",
                            metadata={"pii_types": list(blocked_pii_types)},
                        )

            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="No PII detected",
            )


class ContentSafetyGuardrail(BaseGuardrail):
    """
    LLM-based content safety checking.

    Uses an LLM to classify content as safe/unsafe across multiple dimensions:
    - Violence
    - Hate speech
    - Sexual content
    - Self-harm
    - Illegal activities

    Example:
        from langchain_openai import ChatOpenAI

        guardrail = ContentSafetyGuardrail(
            llm=ChatOpenAI(model="gpt-4"),
            safety_categories=["violence", "hate", "sexual"],
            threshold=0.8
        )
    """

    def __init__(
        self,
        llm=None,
        safety_categories: List[str] = None,
        threshold: float = 0.7,
        custom_safety_prompt: str = None,
        name: str = None,
    ):
        super().__init__(name=name or "ContentSafety")
        self.llm = llm
        self.safety_categories = safety_categories or [
            "violence",
            "hate",
            "sexual",
            "self-harm",
            "illegal",
        ]
        self.threshold = threshold
        self.custom_safety_prompt = custom_safety_prompt

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check content safety using LLM"""
        if not self.llm:
            # If no LLM provided, do basic keyword checking
            return self._keyword_safety_check(content)

        # Build safety check prompt
        if self.custom_safety_prompt:
            prompt = self.custom_safety_prompt.format(content=content)
        else:
            prompt = self._build_safety_prompt(content)

        try:
            # Call LLM for safety assessment
            response = self.llm.invoke(prompt)

            # Parse response (expect JSON with safety scores)
            import json

            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            # Try to extract JSON from response
            from guardrails.utils import extract_json_from_text

            safety_result = extract_json_from_text(response_text)

            if safety_result:
                is_safe = safety_result.get("is_safe", True)
                violations = safety_result.get("violations", [])
                confidence = safety_result.get("confidence", 1.0)

                if not is_safe and confidence >= self.threshold:
                    return GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        reason=f"Content safety violation: {', '.join(violations)}",
                        metadata={
                            "violations": violations,
                            "confidence": confidence,
                            "safety_result": safety_result,
                        },
                    )
                elif not is_safe:
                    return GuardrailResult(
                        action=GuardrailAction.WARN,
                        reason=f"Potential safety concerns: {', '.join(violations)}",
                        metadata={"violations": violations, "confidence": confidence},
                    )

            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                reason="Content is safe",
                metadata={"safety_result": safety_result if safety_result else {}},
            )

        except Exception as e:
            # On error, fail open (allow) but log warning
            return GuardrailResult(
                action=GuardrailAction.WARN,
                reason=f"Safety check failed: {str(e)}",
                metadata={"error": str(e)},
            )

    def _build_safety_prompt(self, content: str) -> str:
        """Build the safety check prompt"""
        categories_str = ", ".join(self.safety_categories)

        return f"""You are a content safety classifier. Analyze the following content for safety issues.

Content to analyze:
{content}

Safety categories to check: {categories_str}

Provide your assessment as JSON with the following structure:
{{
    "is_safe": true/false,
    "violations": ["category1", "category2"],
    "confidence": 0.0-1.0,
    "explanation": "brief explanation"
}}

Only mark as unsafe if there are clear violations. Be conservative and favor allowing content unless it's clearly problematic."""

    def _keyword_safety_check(self, content: str) -> GuardrailResult:
        """Fallback keyword-based safety check when no LLM is provided"""
        # Simple keyword lists (very basic)
        unsafe_patterns = {
            "violence": ["kill", "murder", "attack", "harm", "weapon"],
            "hate": ["hate", "racist", "discriminate"],
            "illegal": ["illegal", "crime", "fraud"],
        }

        found_violations = []
        content_lower = content.lower()

        for category, keywords in unsafe_patterns.items():
            if category in self.safety_categories:
                for keyword in keywords:
                    if keyword in content_lower:
                        found_violations.append(category)
                        break

        if found_violations:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Potential safety issues detected: {', '.join(found_violations)}",
                metadata={"violations": found_violations, "method": "keyword"},
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="No safety issues detected (keyword check)",
        )


class ToxicityGuardrail(BaseGuardrail):
    """
    Detects toxic, rude, or disrespectful content.

    For production, consider using specialized models like:
    - Perspective API
    - Detoxify
    - Custom fine-tuned models

    This is a simplified implementation.

    Example:
        guardrail = ToxicityGuardrail(threshold=0.7)
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_model: bool = False,
        model_name: str = None,
        name: str = None,
    ):
        super().__init__(name=name or "Toxicity")
        self.threshold = threshold
        self.use_model = use_model
        self.model_name = model_name

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check content toxicity"""
        if self.use_model:
            try:
                # Try to use detoxify library if available
                from detoxify import Detoxify

                model = Detoxify(self.model_name or "original")
                results = model.predict(content)

                # Check toxicity score
                toxicity_score = results.get("toxicity", 0)

                if toxicity_score >= self.threshold:
                    return GuardrailResult(
                        action=GuardrailAction.BLOCK,
                        reason=f"High toxicity detected (score: {toxicity_score:.2f})",
                        metadata={"toxicity_score": toxicity_score, "all_scores": results},
                    )

                return GuardrailResult(
                    action=GuardrailAction.ALLOW,
                    reason="Toxicity within acceptable range",
                    metadata={"toxicity_score": toxicity_score},
                )

            except ImportError:
                # Fall back to keyword check
                return self._keyword_toxicity_check(content)
            except Exception as e:
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    reason=f"Toxicity check failed: {str(e)}",
                )
        else:
            return self._keyword_toxicity_check(content)

    def _keyword_toxicity_check(self, content: str) -> GuardrailResult:
        """Simple keyword-based toxicity check"""
        toxic_keywords = [
            "stupid",
            "idiot",
            "dumb",
            "moron",
            "shut up",
            # Add more as needed
        ]

        content_lower = content.lower()
        found_toxic = [kw for kw in toxic_keywords if kw in content_lower]

        if found_toxic:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Potentially toxic language detected",
                metadata={"found_keywords": found_toxic},
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="No toxic language detected",
        )


class PromptInjectionGuardrail(BaseGuardrail):
    """
    Detects potential prompt injection attacks.

    Looks for patterns that might indicate attempts to override system instructions.

    Example:
        guardrail = PromptInjectionGuardrail(strict=True)
    """

    def __init__(
        self,
        strict: bool = True,
        custom_patterns: List[str] = None,
        name: str = None,
    ):
        super().__init__(name=name or "PromptInjection")
        self.strict = strict
        self.custom_patterns = custom_patterns or []

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check for prompt injection attempts"""
        # Common prompt injection patterns
        injection_patterns = [
            r"ignore (previous|above|all|the|your) (instructions?|prompts?|rules?)",
            r"disregard (previous|above|all|the|your) (instructions?|prompts?|rules?)",
            r"forget (everything|all|previous|above)",
            r"new instructions?:",
            r"system prompt:",
            r"you are now",
            r"act as",
            r"pretend (you are|to be)",
            r"roleplay as",
        ] + self.custom_patterns

        import re

        content_lower = content.lower()
        detected_patterns = []

        for pattern in injection_patterns:
            if re.search(pattern, content_lower):
                detected_patterns.append(pattern)

        if detected_patterns:
            if self.strict:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Potential prompt injection detected",
                    metadata={"detected_patterns": detected_patterns},
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    reason="Possible prompt injection attempt",
                    metadata={"detected_patterns": detected_patterns},
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="No prompt injection detected",
        )


class CodeExecutionGuardrail(BaseGuardrail):
    """
    Detects and prevents potentially malicious code execution attempts.

    Useful when your agent might generate or execute code.

    Example:
        guardrail = CodeExecutionGuardrail(
            allowed_imports=["os", "sys", "json"],
            block_dangerous_functions=True
        )
    """

    def __init__(
        self,
        allowed_imports: List[str] = None,
        block_dangerous_functions: bool = True,
        name: str = None,
    ):
        super().__init__(name=name or "CodeExecution")
        self.allowed_imports = set(allowed_imports) if allowed_imports else None
        self.block_dangerous_functions = block_dangerous_functions

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Check for dangerous code patterns"""
        import re

        issues = []

        # Check for dangerous imports
        if self.allowed_imports is not None:
            import_pattern = r"import\s+([a-zA-Z0-9_\.]+)"
            imports = re.findall(import_pattern, content)

            dangerous_imports = [imp for imp in imports if imp not in self.allowed_imports]

            if dangerous_imports:
                issues.append(f"Unauthorized imports: {', '.join(dangerous_imports)}")

        # Check for dangerous functions
        if self.block_dangerous_functions:
            dangerous_functions = [
                r"eval\s*\(",
                r"exec\s*\(",
                r"__import__\s*\(",
                r"compile\s*\(",
                r"open\s*\(",  # File operations
                r"system\s*\(",
                r"subprocess\.",
            ]

            for pattern in dangerous_functions:
                if re.search(pattern, content):
                    issues.append(f"Dangerous function detected: {pattern}")

        if issues:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Potentially unsafe code detected",
                metadata={"issues": issues},
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="Code passes safety checks",
        )

    def get_content_to_validate(self, state: dict) -> str:
        """Extract code content from state"""
        # Look for code in various possible fields
        if "code" in state:
            return state["code"]
        elif "generated_code" in state:
            return state["generated_code"]
        else:
            return super().get_content_to_validate(state)
