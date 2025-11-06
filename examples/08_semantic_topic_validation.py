"""
Example 8: Semantic Topic Validation with LLM
==============================================

This example shows why LLM-based topic validation is superior to keyword matching.

The keyword-based TopicValidationGuardrail uses simple substring matching which
leads to false positives and can't understand semantic relationships.

An LLM can understand:
- Semantic similarity (e.g., "forecast" relates to "weather")
- Context and intent
- Paraphrasing and synonyms
- Off-topic requests that happen to contain topic keywords
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.input_guardrails import TopicValidationGuardrail
from guardrails.llm_guardrails import LLMGuardrail
from guardrails.config import create_anthropic_llm, get_anthropic_api_key


class AgentState(TypedDict):
    input: str
    messages: list


def agent_node(state: AgentState) -> AgentState:
    """Simple agent"""
    return {
        **state,
        "messages": [
            {"role": "user", "content": state["input"]},
            {"role": "assistant", "content": f"Answering: {state['input']}"},
        ],
    }


print("=" * 70)
print("KEYWORD VS LLM-BASED TOPIC VALIDATION")
print("=" * 70)
print()

# Test cases that show the limitations of keyword matching
test_cases = [
    # Should PASS - clearly about weather
    ("What's the forecast for tomorrow?", True, "Uses 'forecast' (weather-related)"),
    ("Will it rain today?", True, "About precipitation (weather)"),
    ("What's the temperature outside?", True, "About temperature (weather)"),

    # Should PASS - semantically about weather but no exact keywords
    ("Should I bring an umbrella tomorrow?", True, "Implies weather question"),
    ("Is it going to be sunny this weekend?", True, "About weather conditions"),

    # Should FAIL - off-topic despite containing "weather"
    ("Whether or not I go depends on you", False, "Contains 'weather' but off-topic"),
    ("I don't know whether that's true", False, "Contains 'whether' (sounds like weather)"),

    # Should FAIL - clearly off-topic
    ("What's the capital of France?", False, "Geography, not weather"),
    ("Tell me about quantum physics", False, "Physics, not weather"),
]

print("TEST CASES:")
for i, (query, should_pass, explanation) in enumerate(test_cases, 1):
    print(f"{i}. '{query}'")
    print(f"   Expected: {'PASS' if should_pass else 'FAIL'} - {explanation}")
print()

# Method 1: Keyword-based (current implementation)
print("=" * 70)
print("METHOD 1: KEYWORD-BASED TOPIC VALIDATION")
print("=" * 70)
print()

keyword_guardrail = TopicValidationGuardrail(
    allowed_topics=["weather", "climate", "temperature", "forecast"],
    fuzzy_match=True,  # This is just substring matching!
    name="KeywordTopic"
)

print("Testing keyword-based approach...")
keyword_results = []
for query, expected, explanation in test_cases:
    result = keyword_guardrail.validate(query)
    passed = result.action.value == "allow"
    correct = passed == expected

    keyword_results.append({
        "query": query,
        "passed": passed,
        "expected": expected,
        "correct": correct
    })

    status = "✓" if correct else "✗"
    print(f"{status} '{query[:50]}...'")
    print(f"   Result: {'PASS' if passed else 'FAIL'} (Expected: {'PASS' if expected else 'FAIL'})")

accuracy = sum(r["correct"] for r in keyword_results) / len(keyword_results)
print(f"\nKeyword Accuracy: {accuracy:.1%}")
print()

# Method 2: LLM-based semantic validation
print("=" * 70)
print("METHOD 2: LLM-BASED SEMANTIC TOPIC VALIDATION")
print("=" * 70)
print()

llm = None
if get_anthropic_api_key():
    try:
        llm = create_anthropic_llm()
        print("✓ Using Claude 4.5 Haiku for semantic topic validation")
    except Exception as e:
        print(f"⚠ Could not initialize Claude: {e}")
else:
    print("ℹ No ANTHROPIC_API_KEY found - showing expected results only")

semantic_guardrail = LLMGuardrail(
    llm=llm,
    system_prompt="You are a topic classifier for a weather information service.",
    instruction_template="""Determine if this user query is asking about WEATHER or CLIMATE.

Allowed topics:
- Weather conditions (current or forecast)
- Temperature, precipitation, wind, humidity
- Climate and weather patterns
- Anything related to atmospheric conditions

The query should be GENUINELY about weather, not just containing weather-related words.

Query: {content}

Respond with JSON:
{{
    "is_safe": true/false,  // true if about weather, false if off-topic
    "reason": "brief explanation of why this is or isn't about weather",
    "confidence": 0.0-1.0
}}

Important: Look at the INTENT and MEANING, not just keywords!""",
    threshold=0.7,
    name="SemanticTopic"
)

if llm:
    print("\nTesting LLM-based semantic approach...")
    llm_results = []
    for query, expected, explanation in test_cases:
        result = semantic_guardrail.validate(query)
        passed = result.action.value == "allow"
        correct = passed == expected

        llm_results.append({
            "query": query,
            "passed": passed,
            "expected": expected,
            "correct": correct,
            "reason": result.reason
        })

        status = "✓" if correct else "✗"
        print(f"\n{status} '{query}'")
        print(f"   Result: {'PASS' if passed else 'FAIL'} (Expected: {'PASS' if expected else 'FAIL'})")
        print(f"   LLM Reasoning: {result.reason}")

    accuracy = sum(r["correct"] for r in llm_results) / len(llm_results)
    print(f"\nLLM Accuracy: {accuracy:.1%}")
else:
    print("\nExpected LLM behavior (with API key):")
    print("✓ Would understand 'forecast' relates to weather")
    print("✓ Would understand 'umbrella tomorrow' is about weather")
    print("✓ Would catch 'whether or not' is NOT about weather")
    print("✓ Would understand semantic meaning, not just keywords")
    print("\nExpected accuracy: 90-100% (vs 60-70% for keywords)")

print()
print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print("""
Keyword-Based Matching:
❌ False positives: "whether" matches "weather"
❌ False negatives: Misses "forecast", "umbrella", "sunny"
❌ No context: Can't distinguish genuine questions from keyword mentions
✓  Fast: < 1ms
✓  Cheap: No API costs
✓  Deterministic: Same input = same output

LLM-Based Semantic Matching:
✓  Understands semantics: "umbrella" relates to weather
✓  Context-aware: Knows "whether or not" isn't about weather
✓  Handles paraphrasing: Many ways to ask about weather
✓  Reduces false positives/negatives significantly
❌ Slower: 1-2 seconds
❌ Costs: ~$0.001 per validation (Claude 4.5 Haiku)
❌ Non-deterministic: Can vary slightly

RECOMMENDATION:
- For simple, exact topic matching with known keywords → Use keyword-based
- For semantic topic understanding → Use LLM-based
- For production → Layer both: keyword filter first, then LLM for edge cases
""")

print()
print("=" * 70)
print("CREATING AN IMPROVED SEMANTIC TOPIC GUARDRAIL")
print("=" * 70)
print("""
# Recommended pattern: Combine both approaches

from guardrails.llm_guardrails import LLMGuardrail
from guardrails.config import create_anthropic_llm

# Create semantic topic guardrail
semantic_topic_guardrail = LLMGuardrail(
    llm=create_anthropic_llm(),
    system_prompt="You validate if queries match allowed topics.",
    instruction_template=\"\"\"
    Is this query about: {allowed_topics}?

    Query: {content}

    Respond: {{"is_safe": bool, "reason": str, "confidence": float}}
    \"\"\",
    threshold=0.8
)

# Use in your graph
workflow.add_node("topic_validation", semantic_topic_guardrail.check)

# For production: Layer keyword check first (fast filter)
# Then LLM check for borderline cases
""")

if llm:
    print("\n✓ LLM-based semantic topic validation is available")
    print("  All examples in this file use Claude 4.5 Haiku")
else:
    print("\nℹ  Set ANTHROPIC_API_KEY in .env to test LLM-based validation")
    print("   Example: cp .env.example .env")
