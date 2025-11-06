"""
LangGraph Guardrails - A comprehensive guardrails implementation for LangGraph workflows.

This package provides reusable guardrail components for:
- Input validation and filtering
- Output validation and filtering
- Content safety checks
- PII detection and redaction
- LLM-based semantic validation (Claude 4.5 Haiku)
- Embedding-based semantic similarity validation
"""

from guardrails.base import (
    BaseGuardrail,
    GuardrailResult,
    GuardrailAction,
    GuardrailChain,
    ConditionalGuardrail,
)
from guardrails.input_guardrails import (
    InputLengthGuardrail,
    RateLimitGuardrail,
)
from guardrails.output_guardrails import (
    OutputFormatGuardrail,
    OutputLengthGuardrail,
)
from guardrails.safety_guardrails import (
    PIIDetectionGuardrail,
    ContentSafetyGuardrail,
    PromptInjectionGuardrail,
    CodeExecutionGuardrail,
)
from guardrails.llm_guardrails import (
    LLMGuardrail,
    BrandSafetyGuardrail,
    ToneGuardrail,
    FactualAccuracyGuardrail,
)
from guardrails.embedding_guardrails import (
    SemanticSimilarityGuardrail,
    MultiModalSemanticGuardrail,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
)
from guardrails.config import (
    create_anthropic_llm,
    create_openai_llm,
    create_embedding_provider,
    get_anthropic_api_key,
)

__all__ = [
    # Base classes
    "BaseGuardrail",
    "GuardrailResult",
    "GuardrailAction",
    "GuardrailChain",
    "ConditionalGuardrail",
    # Input guardrails
    "InputLengthGuardrail",
    "RateLimitGuardrail",
    # Output guardrails
    "OutputFormatGuardrail",
    "OutputLengthGuardrail",
    # Safety guardrails
    "PIIDetectionGuardrail",
    "ContentSafetyGuardrail",
    "PromptInjectionGuardrail",
    "CodeExecutionGuardrail",
    # LLM-based guardrails
    "LLMGuardrail",
    "BrandSafetyGuardrail",
    "ToneGuardrail",
    "FactualAccuracyGuardrail",
    # Embedding-based guardrails
    "SemanticSimilarityGuardrail",
    "MultiModalSemanticGuardrail",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    # Configuration
    "create_anthropic_llm",
    "create_openai_llm",
    "create_embedding_provider",
    "get_anthropic_api_key",
]

__version__ = "0.1.0"
