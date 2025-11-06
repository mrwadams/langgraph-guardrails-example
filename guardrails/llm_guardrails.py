"""
LLM-based guardrails using Claude 4.5 Haiku or other models.

These guardrails use language models for semantic validation, providing
more nuanced and context-aware checking than pattern matching.
"""

from typing import Optional, Callable, Dict, Any
from guardrails.base import BaseGuardrail, GuardrailResult, GuardrailAction
from guardrails.utils import extract_json_from_text


class LLMGuardrail(BaseGuardrail):
    """
    Base LLM-powered guardrail with customizable prompts.

    This is a flexible guardrail that uses an LLM to validate content based on
    custom instructions. You provide the prompt/instructions, and the LLM
    evaluates whether content passes or fails.

    The LLM should respond with JSON in this format:
    {
        "is_safe": true/false,
        "reason": "explanation",
        "confidence": 0.0-1.0,
        "metadata": {...}  // optional
    }

    Example:
        # Create a custom brand safety guardrail
        brand_guardrail = LLMGuardrail(
            llm=create_anthropic_llm(),
            system_prompt="You are a brand safety checker for ACME Corp.",
            instruction_template=\"\"\"
            Check if this content aligns with our brand values:
            - Professional and respectful tone
            - No controversial political statements
            - Family-friendly language

            Content: {content}

            Respond with JSON: {{"is_safe": bool, "reason": str, "confidence": float}}
            \"\"\",
            threshold=0.8,
        )
    """

    def __init__(
        self,
        llm=None,
        system_prompt: str = None,
        instruction_template: str = None,
        threshold: float = 0.7,
        fallback_fn: Callable[[str], GuardrailResult] = None,
        parse_response_fn: Callable[[str], Dict[str, Any]] = None,
        name: str = None,
    ):
        """
        Initialize LLM guardrail.

        Args:
            llm: LangChain LLM instance (e.g., ChatAnthropic, ChatOpenAI)
            system_prompt: System prompt to set context for the LLM
            instruction_template: Template for the validation instruction.
                Must include {content} placeholder. Can include other placeholders
                that will be filled from metadata.
            threshold: Confidence threshold for blocking (0.0-1.0)
            fallback_fn: Function to call if LLM fails or is not provided
            parse_response_fn: Custom function to parse LLM response into result dict
            name: Guardrail name for metrics
        """
        super().__init__(name=name or "LLMGuardrail")
        self.llm = llm
        self.system_prompt = system_prompt or "You are a content validation assistant."
        self.instruction_template = instruction_template or self._default_instruction_template()
        self.threshold = threshold
        self.fallback_fn = fallback_fn
        self.parse_response_fn = parse_response_fn or self._default_parse_response

    def _default_instruction_template(self) -> str:
        """Default instruction template if none provided"""
        return """Analyze the following content and determine if it's appropriate.

Content: {content}

Respond with JSON:
{{
    "is_safe": true or false,
    "reason": "brief explanation",
    "confidence": 0.0 to 1.0
}}"""

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate content using LLM"""
        if not self.llm:
            # No LLM provided, use fallback if available
            if self.fallback_fn:
                return self.fallback_fn(content)
            else:
                # No fallback, allow by default (fail open)
                return GuardrailResult(
                    action=GuardrailAction.ALLOW,
                    reason="No LLM configured and no fallback provided",
                )

        try:
            # Build the prompt
            prompt_vars = {"content": content}
            if metadata:
                prompt_vars.update(metadata)

            instruction = self.instruction_template.format(**prompt_vars)

            # Call LLM
            if self.system_prompt:
                # Use system prompt if LLM supports it
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": instruction},
                ]
                try:
                    response = self.llm.invoke(messages)
                except:
                    # Fallback to simple string prompt if messages not supported
                    full_prompt = f"{self.system_prompt}\n\n{instruction}"
                    response = self.llm.invoke(full_prompt)
            else:
                response = self.llm.invoke(instruction)

            # Extract response text
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse response
            result_dict = self.parse_response_fn(response_text)

            # Convert to GuardrailResult
            is_safe = result_dict.get("is_safe", True)
            reason = result_dict.get("reason", "No reason provided")
            confidence = result_dict.get("confidence", 1.0)
            result_metadata = result_dict.get("metadata", {})
            result_metadata["llm_response"] = response_text
            result_metadata["confidence"] = confidence

            if not is_safe and confidence >= self.threshold:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=reason,
                    metadata=result_metadata,
                )
            elif not is_safe:
                return GuardrailResult(
                    action=GuardrailAction.WARN,
                    reason=f"Low confidence concern: {reason}",
                    metadata=result_metadata,
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.ALLOW,
                    reason=reason,
                    metadata=result_metadata,
                )

        except Exception as e:
            # LLM call failed
            error_msg = f"LLM validation failed: {str(e)}"

            # Try fallback if available
            if self.fallback_fn:
                return self.fallback_fn(content)

            # No fallback, fail open with warning
            return GuardrailResult(
                action=GuardrailAction.WARN,
                reason=error_msg,
                metadata={"error": str(e)},
            )

    def _default_parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Default response parser - expects JSON.

        Override this if your LLM returns a different format.
        """
        # Try to extract JSON
        result = extract_json_from_text(response_text)

        if result and "is_safe" in result:
            return result

        # Couldn't parse JSON, try to infer from text
        response_lower = response_text.lower()

        # Look for clear safe/unsafe indicators
        if "is_safe: false" in response_lower or '"is_safe": false' in response_lower:
            is_safe = False
        elif "is_safe: true" in response_lower or '"is_safe": true' in response_lower:
            is_safe = True
        elif any(word in response_lower for word in ["unsafe", "inappropriate", "violation", "block"]):
            is_safe = False
        elif any(word in response_lower for word in ["safe", "appropriate", "allow", "okay"]):
            is_safe = True
        else:
            # Can't determine, default to safe (fail open)
            is_safe = True

        return {
            "is_safe": is_safe,
            "reason": response_text[:200],  # Truncate if too long
            "confidence": 0.5,  # Low confidence since we couldn't parse properly
        }


class BrandSafetyGuardrail(LLMGuardrail):
    """
    Validates content against brand guidelines and values.

    Example:
        guardrail = BrandSafetyGuardrail(
            llm=create_anthropic_llm(),
            brand_values=[
                "Professional and respectful",
                "Family-friendly",
                "Inclusive and diverse",
            ],
            prohibited_topics=["politics", "religion"],
        )
    """

    def __init__(
        self,
        llm,
        brand_values: list[str] = None,
        prohibited_topics: list[str] = None,
        threshold: float = 0.7,
        name: str = None,
    ):
        brand_values = brand_values or ["Professional", "Respectful", "Appropriate"]
        prohibited_topics = prohibited_topics or []

        system_prompt = "You are a brand safety expert ensuring content aligns with company values."

        instruction_template = f"""Evaluate if this content aligns with our brand values and guidelines.

Brand Values:
{chr(10).join(f'- {v}' for v in brand_values)}

{f"Prohibited Topics: {', '.join(prohibited_topics)}" if prohibited_topics else ""}

Content to evaluate:
{{content}}

Respond with JSON:
{{{{
    "is_safe": true/false,
    "reason": "explanation of why content passes or fails",
    "confidence": 0.0-1.0,
    "violations": ["list of specific violations if any"]
}}}}

Be thoughtful and consider context. Content should only fail if it clearly violates brand values or discusses prohibited topics."""

        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            instruction_template=instruction_template,
            threshold=threshold,
            name=name or "BrandSafety",
        )


class ToneGuardrail(LLMGuardrail):
    """
    Validates that content matches desired tone/style.

    Example:
        guardrail = ToneGuardrail(
            llm=create_anthropic_llm(),
            desired_tone="friendly and helpful, not too formal",
            disallowed_tones=["aggressive", "condescending", "sarcastic"],
        )
    """

    def __init__(
        self,
        llm,
        desired_tone: str,
        disallowed_tones: list[str] = None,
        threshold: float = 0.7,
        name: str = None,
    ):
        disallowed_tones = disallowed_tones or []

        system_prompt = "You are a tone analysis expert."

        instruction_template = f"""Analyze the tone of this content.

Desired tone: {desired_tone}
{f"Disallowed tones: {', '.join(disallowed_tones)}" if disallowed_tones else ""}

Content:
{{content}}

Respond with JSON:
{{{{
    "is_safe": true/false (true if tone matches desired, false if it doesn't or uses disallowed tones),
    "reason": "explanation of the tone",
    "confidence": 0.0-1.0,
    "detected_tone": "description of actual tone"
}}}}"""

        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            instruction_template=instruction_template,
            threshold=threshold,
            name=name or "ToneValidation",
        )


class FactualAccuracyGuardrail(LLMGuardrail):
    """
    Validates factual claims against known information.

    Example:
        guardrail = FactualAccuracyGuardrail(
            llm=create_anthropic_llm(),
            knowledge_base="Paris is the capital of France. The Eiffel Tower is in Paris.",
        )
    """

    def __init__(
        self,
        llm,
        knowledge_base: str = None,
        require_citations: bool = False,
        threshold: float = 0.8,
        name: str = None,
    ):
        system_prompt = "You are a fact-checking expert."

        kb_section = f"\n\nKnowledge Base (verified facts):\n{knowledge_base}" if knowledge_base else ""
        citation_req = "\nContent MUST include citations or sources." if require_citations else ""

        instruction_template = f"""Verify the factual accuracy of this content.{kb_section}{citation_req}

Content to verify:
{{content}}

Respond with JSON:
{{{{
    "is_safe": true/false (true if accurate, false if contains misinformation),
    "reason": "explanation",
    "confidence": 0.0-1.0,
    "issues": ["list of factual errors if any"]
}}}}"""

        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            instruction_template=instruction_template,
            threshold=threshold,
            name=name or "FactualAccuracy",
        )
