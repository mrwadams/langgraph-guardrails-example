"""
LangGraph Guardrails - A comprehensive guardrails implementation for LangGraph workflows.

This package provides reusable guardrail components for:
- Input validation and filtering
- Output validation and filtering
- Content safety checks
- PII detection and redaction
- LLM-based semantic validation (Claude 3.5 Haiku)
"""

from guardrails.base import (
    BaseGuardrail,
    GuardrailResult,
    GuardrailChain,
)
from guardrails.input_guardrails import (
    TopicValidationGuardrail,
    ProfanityFilterGuardrail,
    InputLengthGuardrail,
)
from guardrails.output_guardrails import (
    OutputFormatGuardrail,
    FactualityGuardrail,
)
from guardrails.safety_guardrails import (
    PIIDetectionGuardrail,
    ContentSafetyGuardrail,
)
from guardrails.llm_guardrails import (
    LLMGuardrail,
    BrandSafetyGuardrail,
    ToneGuardrail,
    FactualAccuracyGuardrail,
)
from guardrails.config import (
    create_anthropic_llm,
    create_openai_llm,
    get_anthropic_api_key,
)

__all__ = [
    # Base classes
    "BaseGuardrail",
    "GuardrailResult",
    "GuardrailChain",
    # Input guardrails
    "TopicValidationGuardrail",
    "ProfanityFilterGuardrail",
    "InputLengthGuardrail",
    # Output guardrails
    "OutputFormatGuardrail",
    "FactualityGuardrail",
    # Safety guardrails
    "PIIDetectionGuardrail",
    "ContentSafetyGuardrail",
    # LLM-based guardrails
    "LLMGuardrail",
    "BrandSafetyGuardrail",
    "ToneGuardrail",
    "FactualAccuracyGuardrail",
    # Configuration
    "create_anthropic_llm",
    "create_openai_llm",
    "get_anthropic_api_key",
]

__version__ = "0.1.0"
