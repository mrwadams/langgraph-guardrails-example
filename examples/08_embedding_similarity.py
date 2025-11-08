"""
Example 8: Semantic Similarity Guardrails with Embeddings
==========================================================

This example demonstrates the core technique of using embedding-based semantic similarity
for fast, accurate topic validation. This approach provides LLM-like semantic understanding
but is 10-100x faster and can run entirely locally.

KEY CONCEPT: Semantic Similarity via Embeddings
-----------------------------------------------
Instead of keyword matching or expensive LLM calls, we:
1. Convert text to vectors (embeddings) that capture semantic meaning
2. Measure cosine similarity between input and allowed topics
3. Accept/reject based on similarity threshold

Benefits:
- Understands semantic meaning (not just keywords)
- 10-50ms latency vs 1-2s for LLMs
- 85-95% accuracy for topic validation
- Can run completely offline/private
- No per-request costs when using local models

Three Embedding Provider Options (easily switchable):
-----------------------------------------------------

1. **Local (SentenceTransformer)** - Pure Python, no external setup
   provider = create_embedding_provider("local")
   Best for: Privacy, simplicity, offline operation

2. **LM Studio** - GUI for experimenting with different models
   provider = create_embedding_provider("lm-studio", base_url="http://localhost:1234/v1")
   Best for: Development, testing different embedding models

3. **OpenAI** - Highest quality, cloud-based
   provider = create_embedding_provider("openai")
   Best for: Production scale, highest accuracy

This example defaults to the simplest option (local) but shows how to switch providers.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.embedding_guardrails import SemanticSimilarityGuardrail
from guardrails.config import create_embedding_provider


class AgentState(TypedDict):
    input: str
    messages: list


def agent_node(state: AgentState) -> AgentState:
    """Simple weather agent"""
    return {
        **state,
        "messages": [
            {"role": "user", "content": state["input"]},
            {"role": "assistant", "content": f"Weather info for: {state['input']}"},
        ],
    }


print("=" * 70)
print("SEMANTIC SIMILARITY GUARDRAILS DEMO")
print("=" * 70)
print()

# ============================================================================
# PROVIDER CONFIGURATION - Switch between these options:
# ============================================================================

# Option 1: Local SentenceTransformer (default - simplest, no setup needed)
PROVIDER = "local"
embedding_provider = create_embedding_provider(PROVIDER)

# Option 2: LM Studio (requires LM Studio running with embedding model loaded)
# PROVIDER = "lm-studio"
# embedding_provider = create_embedding_provider(
#     PROVIDER,
#     base_url="http://localhost:1234/v1",
#     model="default"
# )

# Option 3: OpenAI (requires OPENAI_API_KEY environment variable)
# PROVIDER = "openai"
# embedding_provider = create_embedding_provider(PROVIDER)

print(f"Using embedding provider: {PROVIDER}")
print("This provider will generate embeddings (vectors) for semantic comparison")
print()

# Define allowed topics as natural language descriptions
print("Setting up semantic topic guardrail...")
print()
print("How it works:")
print("  1. Topic descriptions are converted to embeddings (vectors)")
print("  2. Each user input is converted to an embedding")
print("  3. Cosine similarity is calculated between input and topics")
print("  4. If similarity > threshold, input is allowed through")
print()

weather_topics = [
    "weather forecasts and predictions",
    "temperature and climate conditions",
    "precipitation like rain and snow",
    "atmospheric conditions and storms",
]

print(f"Allowed topic descriptions ({len(weather_topics)}):")
for topic in weather_topics:
    print(f"  • {topic}")
print()

topic_guardrail = SemanticSimilarityGuardrail(
    topic_descriptions=weather_topics,
    embedding_provider=embedding_provider,
    similarity_threshold=0.65,  # Inputs with similarity < 0.65 are blocked
    name="WeatherTopicGuardrail"
)
print("✓ Guardrail configured")
print(f"  Similarity threshold: 0.65 (adjust higher for stricter filtering)")
print()

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("topic_check", topic_guardrail.check)
workflow.add_node("weather_agent", agent_node)
workflow.add_node("rejection", topic_guardrail.rejection_node)

workflow.set_entry_point("topic_check")
workflow.add_conditional_edges("topic_check", topic_guardrail.route, {
    "continue": "weather_agent",
    "reject": "rejection"
})
workflow.add_edge("weather_agent", END)
workflow.add_edge("rejection", END)

app = workflow.compile()

print("=" * 70)
print("TESTING SEMANTIC UNDERSTANDING")
print("=" * 70)
print()
print("Key insight: These tests show semantic understanding, not keyword matching!")
print("Notice how the guardrail can:")
print("  • Understand implied weather questions without weather keywords")
print("  • Block off-topic questions even if they contain related words")
print()

# Test cases demonstrating semantic understanding
test_cases = [
    # Should PASS - clearly weather-related
    ("What's the forecast for tomorrow?", True, "Direct weather query"),
    ("Will it rain today?", True, "Precipitation question"),
    ("What's the temperature outside?", True, "Temperature question"),

    # Should PASS - semantically related (no exact keyword match!)
    ("Should I bring an umbrella tomorrow?", True, "Implies weather/precipitation"),
    ("Is it going to be sunny this weekend?", True, "Weather condition query"),
    ("Do I need a coat today?", True, "Implies temperature concern"),
    ("What's it like outside?", True, "General condition inquiry"),

    # Should FAIL - semantically different topics
    ("What's the capital of France?", False, "Geography question"),
    ("Tell me about quantum physics", False, "Physics question"),
    ("How do I cook pasta?", False, "Cooking question"),
    ("What's 2 + 2?", False, "Math question"),
]

print(f"Running {len(test_cases)} test cases...\n")

results = {"passed": 0, "failed": 0}

for i, (query, should_pass, description) in enumerate(test_cases, 1):
    print(f"Test {i}: {query}")
    print(f"  Expected: {'PASS' if should_pass else 'FAIL'} ({description})")

    try:
        result = app.invoke({"input": query, "messages": []})
        did_pass = not result.get("rejected", False)

        # Check if result matches expectation
        if did_pass == should_pass:
            print(f"  Result: ✓ {'PASSED' if did_pass else 'BLOCKED'} (correct)")
            results["passed"] += 1
        else:
            print(f"  Result: ✗ {'PASSED' if did_pass else 'BLOCKED'} (incorrect - expected {'PASS' if should_pass else 'BLOCK'})")
            results["failed"] += 1
    except Exception as e:
        print(f"  Result: ✗ Error: {e}")
        results["failed"] += 1

    print()

print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Total tests: {len(test_cases)}")
print(f"Passed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Accuracy: {results['passed']/len(test_cases)*100:.1f}%")
print()

# Show metrics
metrics = topic_guardrail.get_metrics()
print("Guardrail Metrics:")
print(f"  Total checks: {metrics['total_checks']}")
print(f"  Blocked: {metrics['blocked']}")
print(f"  Allowed: {metrics['allowed']}")
print(f"  Block rate: {metrics['block_rate']:.2%}")
print()

print("=" * 70)
print("UNDERSTANDING THE EMBEDDING SIMILARITY TECHNIQUE")
print("=" * 70)
print()
print("What are embeddings?")
print("  Embeddings are vectors (arrays of numbers) that capture semantic meaning.")
print("  Similar concepts have similar vectors, even with different wording.")
print()
print("Example:")
print("  'What's the weather?' → [0.23, -0.45, 0.67, ...]  (768 dimensions)")
print("  'Will it rain?'      → [0.21, -0.43, 0.69, ...]  (similar vector!)")
print("  'What is 2+2?'       → [-0.89, 0.12, -0.34, ...] (very different)")
print()
print("We calculate cosine similarity (0.0 to 1.0) between vectors:")
print("  • 1.0 = identical meaning")
print("  • 0.8-0.9 = very similar")
print("  • 0.6-0.7 = somewhat related")
print("  • <0.5 = different topics")
print()

print("=" * 70)
print("CHOOSING AN EMBEDDING PROVIDER")
print("=" * 70)
print()
print("All providers use the same semantic similarity technique.")
print("The only difference is where/how the embeddings are generated.")
print()
print("┌─────────────────────┬────────────────────────────────────────┐")
print("│ Provider            │ Best For                               │")
print("├─────────────────────┼────────────────────────────────────────┤")
print("│ Local (default)     │ • Getting started quickly              │")
print("│ SentenceTransformer │ • Privacy-critical applications        │")
print("│                     │ • Offline/air-gapped environments      │")
print("│                     │ • Cost-sensitive + moderate volume     │")
print("├─────────────────────┼────────────────────────────────────────┤")
print("│ LM Studio           │ • Experimenting with different models  │")
print("│                     │ • Development/testing                  │")
print("│                     │ • Visual model management              │")
print("├─────────────────────┼────────────────────────────────────────┤")
print("│ OpenAI              │ • Production at scale (1000+ req/min)  │")
print("│                     │ • Highest accuracy requirements        │")
print("│                     │ • No compute overhead needed           │")
print("└─────────────────────┴────────────────────────────────────────┘")
print()
print("💡 Start with 'local' (SentenceTransformer) - it just works!")
print("   Switch to OpenAI only if you need higher scale/accuracy.")
print()

print("=" * 70)
print("SWITCHING PROVIDERS (CODE EXAMPLES)")
print("=" * 70)
print()
print("# Option 1: Local SentenceTransformer (recommended starting point)")
print("from guardrails.config import create_embedding_provider")
print("provider = create_embedding_provider('local')")
print()
print("# Option 2: LM Studio (requires LM Studio running)")
print("provider = create_embedding_provider(")
print("    'lm-studio',")
print("    base_url='http://localhost:1234/v1',")
print("    model='default'  # or specific model name")
print(")")
print()
print("# Option 3: OpenAI (requires OPENAI_API_KEY env var)")
print("provider = create_embedding_provider('openai')")
print("# or with specific model:")
print("# provider = create_embedding_provider('openai', model='text-embedding-3-large')")
print()
print("# Then use with any guardrail:")
print("guardrail = SemanticSimilarityGuardrail(")
print("    topic_descriptions=['topic1', 'topic2'],")
print("    embedding_provider=provider,")
print("    similarity_threshold=0.65")
print(")")
print()

print("=" * 70)
print("TUNING FOR YOUR USE CASE")
print("=" * 70)
print()
print("If too many valid inputs are blocked:")
print("  • Lower the similarity_threshold (e.g., 0.60 → 0.55)")
print("  • Add more diverse topic_descriptions")
print("  • Make topic descriptions more general")
print()
print("If too many off-topic inputs are allowed:")
print("  • Raise the similarity_threshold (e.g., 0.65 → 0.75)")
print("  • Make topic descriptions more specific")
print("  • Use fewer, more focused topic descriptions")
print()
print("Typical threshold ranges:")
print("  • 0.50-0.60: Very permissive (broad matching)")
print("  • 0.60-0.70: Balanced (recommended starting point)")
print("  • 0.70-0.80: Strict (high precision)")
print("  • 0.80+: Very strict (may reject valid inputs)")
print()
print("Provider-specific notes:")
print("  • Local: First run downloads model (~80MB), then cached")
print("  • LM Studio: Requires app running with embedding model loaded")
print("  • OpenAI: Requires OPENAI_API_KEY environment variable")
print()
print("For more examples:")
print("  • examples/07_embedding_vs_llm_comparison.py - Compare with LLM approach")
print("  • examples/06_semantic_topic_validation.py - Multi-topic validation")
print("  • guardrails/README.md - Full documentation")
print()
