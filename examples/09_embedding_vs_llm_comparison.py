"""
Example 9: Embedding vs LLM Topic Validation Comparison
=========================================================

This example compares two approaches for semantic topic validation:

1. **Embedding-based** (SemanticSimilarityGuardrail):
   - Speed: ~10-50ms per validation
   - Cost: Very low (after initial embedding)
   - Accuracy: 85-95%
   - Best for: High-volume, clear topic categories

2. **LLM-based** (LLMGuardrail with Claude 4.5 Haiku):
   - Speed: ~1-2s per validation
   - Cost: Low-medium ($1/M tokens)
   - Accuracy: 95-99%
   - Best for: Complex reasoning, context understanding

This helps you choose the right approach for your use case.
"""

import time
from typing import TypedDict, List
from guardrails.embedding_guardrails import (
    SemanticSimilarityGuardrail,
    SentenceTransformerProvider,
    OpenAIEmbeddingProvider,
)
from guardrails.llm_guardrails import LLMGuardrail
from guardrails.config import create_anthropic_llm, get_anthropic_api_key, get_openai_api_key


def test_embedding_guardrail(test_cases: List[dict], use_openai: bool = False):
    """Test embedding-based topic validation."""
    print("\n" + "="*70)
    print("EMBEDDING-BASED TOPIC VALIDATION")
    print("="*70)

    # Choose embedding provider
    if use_openai and get_openai_api_key():
        print("Using OpenAI embeddings (text-embedding-3-small)")
        provider = OpenAIEmbeddingProvider()
    else:
        print("Using local embeddings (all-MiniLM-L6-v2, no API key required)")
        provider = SentenceTransformerProvider()

    # Create guardrail
    guardrail = SemanticSimilarityGuardrail(
        topic_descriptions=[
            "weather forecasts, climate, temperature, and atmospheric conditions",
            "meteorology and weather patterns"
        ],
        embedding_provider=provider,
        similarity_threshold=0.65,  # Lower threshold for embeddings
    )

    results = []
    total_time = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. \"{test['input']}\"")

        start = time.time()
        result = guardrail.validate(test["input"])
        elapsed = time.time() - start
        total_time += elapsed

        similarity = result.metadata.get("similarity_score", 0)
        is_correct = (result.action.value == "allow") == test["should_allow"]

        print(f"   Result: {result.action.value.upper()}")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   Time: {elapsed*1000:.1f}ms")
        print(f"   Correct: {'✓' if is_correct else '✗ WRONG'}")

        results.append(is_correct)

    accuracy = (sum(results) / len(results)) * 100
    avg_time = (total_time / len(results)) * 1000

    print(f"\n{'='*70}")
    print(f"EMBEDDING RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.1f}% ({sum(results)}/{len(results)} correct)")
    print(f"Average time: {avg_time:.1f}ms")
    print(f"Total time: {total_time:.2f}s")

    return accuracy, avg_time


def test_llm_guardrail(test_cases: List[dict]):
    """Test LLM-based topic validation."""
    print("\n" + "="*70)
    print("LLM-BASED TOPIC VALIDATION (Claude 4.5 Haiku)")
    print("="*70)

    if not get_anthropic_api_key():
        print("⚠ ANTHROPIC_API_KEY not set - skipping LLM test")
        return None, None

    # Create LLM guardrail
    llm = create_anthropic_llm()

    guardrail = LLMGuardrail(
        llm=llm,
        system_prompt="You validate if user queries are about weather and climate.",
        instruction_template="""Is this query about weather, climate, forecasts, or atmospheric conditions?

Query: {content}

Respond with JSON:
{{
    "is_safe": true/false,  // true if about weather, false if not
    "reason": "brief explanation",
    "confidence": 0.0-1.0
}}

Be understanding of related topics. Weather includes forecasts, climate, temperature, etc.""",
        threshold=0.7,
    )

    results = []
    total_time = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. \"{test['input']}\"")

        start = time.time()
        result = guardrail.validate(test["input"])
        elapsed = time.time() - start
        total_time += elapsed

        confidence = result.metadata.get("confidence", 0)
        is_correct = (result.action.value == "allow") == test["should_allow"]

        print(f"   Result: {result.action.value.upper()}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Reason: {result.reason[:60]}...")
        print(f"   Time: {elapsed*1000:.1f}ms")
        print(f"   Correct: {'✓' if is_correct else '✗ WRONG'}")

        results.append(is_correct)

    accuracy = (sum(results) / len(results)) * 100
    avg_time = (total_time / len(results)) * 1000

    print(f"\n{'='*70}")
    print(f"LLM RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.1f}% ({sum(results)}/{len(results)} correct)")
    print(f"Average time: {avg_time:.1f}ms")
    print(f"Total time: {total_time:.2f}s")

    return accuracy, avg_time


if __name__ == "__main__":
    # Test cases covering various scenarios
    test_cases = [
        # Clear weather questions (should allow)
        {"input": "What's the weather like today?", "should_allow": True},
        {"input": "Will it rain tomorrow?", "should_allow": True},
        {"input": "What's the forecast for this weekend?", "should_allow": True},
        {"input": "Is it going to be hot next week?", "should_allow": True},

        # Semantic weather questions (should allow)
        {"input": "Should I bring an umbrella?", "should_allow": True},
        {"input": "Do I need a jacket for tomorrow?", "should_allow": True},
        {"input": "What's the temperature going to be?", "should_allow": True},

        # Weather-related words but off-topic (tricky cases)
        {"input": "I'm feeling under the weather, can you help?", "should_allow": False},
        {"input": "Whether or not I go depends on you", "should_allow": False},

        # Clearly off-topic (should block)
        {"input": "What's the capital of France?", "should_allow": False},
        {"input": "Tell me a joke", "should_allow": False},
        {"input": "How do I make pasta?", "should_allow": False},

        # Edge cases
        {"input": "Climate change predictions", "should_allow": True},
        {"input": "Atmospheric pressure", "should_allow": True},
        {"input": "Best Italian restaurant nearby", "should_allow": False},
    ]

    print("="*70)
    print("TOPIC VALIDATION APPROACH COMPARISON")
    print("="*70)
    print(f"Testing {len(test_cases)} queries for weather topic validation\n")

    # Test embedding approach
    use_openai_embeddings = get_openai_api_key() is not None
    embedding_accuracy, embedding_time = test_embedding_guardrail(
        test_cases,
        use_openai=use_openai_embeddings
    )

    # Test LLM approach
    llm_accuracy, llm_time = test_llm_guardrail(test_cases)

    # Final comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print("\n📊 Accuracy:")
    print(f"  Embedding-based: {embedding_accuracy:.1f}%")
    if llm_accuracy:
        print(f"  LLM-based:       {llm_accuracy:.1f}%")
        print(f"  Difference:      {llm_accuracy - embedding_accuracy:+.1f}%")

    print("\n⚡ Speed:")
    print(f"  Embedding-based: {embedding_time:.1f}ms avg")
    if llm_time:
        print(f"  LLM-based:       {llm_time:.1f}ms avg")
        speedup = llm_time / embedding_time
        print(f"  Speedup:         {speedup:.1f}x faster with embeddings")

    print("\n💰 Cost considerations:")
    print("  Embedding-based:")
    print("    - OpenAI: ~$0.00002 per request (text-embedding-3-small)")
    print("    - Local: FREE (after model download)")
    if llm_accuracy:
        print("  LLM-based:")
        print("    - Claude 4.5 Haiku: ~$0.001 per request")
        print("    - ~50-100x more expensive than OpenAI embeddings")
        print("    - ~∞ more expensive than local embeddings")

    print("\n📋 Recommendations:")
    print("\n  Use EMBEDDING-based when:")
    print("    ✓ High volume (1000+ requests/min)")
    print("    ✓ Clear, well-defined topics")
    print("    ✓ 85-95% accuracy acceptable")
    print("    ✓ Cost is a constraint")
    print("    ✓ Need <50ms response time")

    print("\n  Use LLM-based when:")
    print("    ✓ Complex, nuanced validation needed")
    print("    ✓ Need to understand context and intent")
    print("    ✓ 95-99% accuracy required")
    print("    ✓ Edge cases and tricky phrasing common")
    print("    ✓ Can tolerate 1-2s response time")

    print("\n  💡 BEST APPROACH:")
    print("    Layer both! Use embedding guardrail first (fast),")
    print("    then LLM for borderline cases (similarity 0.6-0.8)")
    print("="*70)
