"""
Example 8: Using LM Studio for Local Embeddings
================================================

This example demonstrates how to use LM Studio with the semantic similarity guardrails.
LM Studio provides a user-friendly GUI for running embedding models locally with an
OpenAI-compatible API endpoint.

Benefits of LM Studio:
- Free and runs locally (no API costs)
- User-friendly GUI for model management
- Run various embedding models (BERT, MiniLM, E5, nomic-embed, etc.)
- OpenAI-compatible API (easy integration)
- Switch models without changing code
- Monitor performance in real-time

Setup Instructions:
1. Download LM Studio from https://lmstudio.ai/
2. In LM Studio, go to the "Search" tab
3. Search for and download an embedding model (e.g., "nomic-embed-text", "all-MiniLM-L6-v2")
4. Go to the "Local Server" tab
5. Load your embedding model
6. Click "Start Server" (default: http://localhost:1234)
7. Run this example!
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
print("LM STUDIO EMBEDDING GUARDRAILS DEMO")
print("=" * 70)
print()

# Configuration
LM_STUDIO_ENDPOINT = "http://localhost:1234/v1"  # Default LM Studio endpoint
LM_STUDIO_MODEL = "default"  # Or specify the model name you loaded

print("Configuration:")
print(f"  LM Studio Endpoint: {LM_STUDIO_ENDPOINT}")
print(f"  Model: {LM_STUDIO_MODEL}")
print()

# Create embedding provider using LM Studio
print("Creating LM Studio embedding provider...")
try:
    embedding_provider = create_embedding_provider(
        "lm-studio",
        base_url=LM_STUDIO_ENDPOINT,
        model=LM_STUDIO_MODEL
    )
    print("✓ LM Studio provider created successfully")
    print()
except Exception as e:
    print(f"✗ Error creating provider: {e}")
    print()
    print("Make sure:")
    print("  1. LM Studio is running")
    print("  2. You have loaded an embedding model")
    print("  3. The local server is started")
    print("  4. The endpoint URL is correct")
    print()
    exit(1)

# Create a weather-focused topic guardrail
print("Setting up weather topic guardrail...")
weather_topics = [
    "weather forecasts and predictions",
    "temperature and climate conditions",
    "precipitation like rain and snow",
    "atmospheric conditions and storms",
]

topic_guardrail = SemanticSimilarityGuardrail(
    topic_descriptions=weather_topics,
    embedding_provider=embedding_provider,
    similarity_threshold=0.65,  # Adjust based on your model
    name="WeatherTopicGuardrail"
)
print(f"✓ Guardrail configured with {len(weather_topics)} topic descriptions")
print(f"  Similarity threshold: 0.65")
print()

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("topic_check", topic_guardrail.check)
workflow.add_node("weather_agent", agent_node)
workflow.add_node("rejection", topic_guardrail.rejection_node)

workflow.set_entry_point("topic_check")
workflow.add_conditional_edges("topic_check", topic_guardrail.route, {
    "allow": "weather_agent",
    "block": "rejection"
})
workflow.add_edge("weather_agent", END)
workflow.add_edge("rejection", END)

app = workflow.compile()

print("=" * 70)
print("TESTING SEMANTIC UNDERSTANDING")
print("=" * 70)
print()

# Test cases demonstrating semantic understanding
test_cases = [
    # Should PASS - clearly weather-related
    ("What's the forecast for tomorrow?", True, "Direct weather query"),
    ("Will it rain today?", True, "Precipitation question"),
    ("What's the temperature outside?", True, "Temperature question"),

    # Should PASS - semantically related (no exact keyword match)
    ("Should I bring an umbrella tomorrow?", True, "Implies weather concern"),
    ("Is it going to be sunny this weekend?", True, "Weather condition query"),
    ("Do I need a coat today?", True, "Implies temperature/weather"),
    ("What's it like outside?", True, "General weather inquiry"),

    # Should FAIL - off-topic
    ("What's the capital of France?", False, "Geography question"),
    ("Tell me about quantum physics", False, "Science question"),
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
        did_pass = "rejection" not in str(result.get("messages", []))

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
print("COMPARISON: LM Studio vs Other Providers")
print("=" * 70)
print()
print("1. LM Studio (this example):")
print("   ✓ Free and runs locally")
print("   ✓ User-friendly GUI")
print("   ✓ Easy model switching")
print("   ✓ OpenAI-compatible API")
print("   ✓ No API costs")
print("   ✓ Privacy (data stays local)")
print()
print("2. SentenceTransformer (create_embedding_provider('local')):")
print("   ✓ Free and runs locally")
print("   ✓ Python-only (no GUI)")
print("   ✓ Good for scripts/automation")
print("   - Manual model management")
print()
print("3. OpenAI (create_embedding_provider('openai')):")
print("   ✓ High quality embeddings")
print("   ✓ No local setup required")
print("   - Costs money (~$0.00002/request)")
print("   - Data sent to OpenAI")
print("   - Requires API key")
print()

print("=" * 70)
print("ADVANCED USAGE")
print("=" * 70)
print()
print("# Using environment variables (recommended for production)")
print("# Add to your .env file:")
print("# LM_STUDIO_BASE_URL=http://localhost:1234/v1")
print("# LM_STUDIO_EMBEDDING_MODEL=your-model-name")
print()
print("import os")
print("provider = create_embedding_provider(")
print("    'lm-studio',")
print("    base_url=os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1'),")
print("    model=os.getenv('LM_STUDIO_EMBEDDING_MODEL', 'default')")
print(")")
print()
print("# Or use OpenAIEmbeddingProvider directly for more control")
print("from guardrails.embedding_guardrails import OpenAIEmbeddingProvider")
print()
print("provider = OpenAIEmbeddingProvider(")
print("    api_key='not-needed',  # LM Studio doesn't validate this")
print("    model='your-model-name',")
print("    base_url='http://localhost:1234/v1'")
print(")")
print()

print("=" * 70)
print("TROUBLESHOOTING")
print("=" * 70)
print()
print("If you get connection errors:")
print("  1. Make sure LM Studio is running")
print("  2. Check that the local server is started (green 'Server Running' indicator)")
print("  3. Verify the port (default: 1234) matches your configuration")
print("  4. Try accessing http://localhost:1234/v1/models in your browser")
print()
print("If accuracy is low:")
print("  1. Try different embedding models in LM Studio")
print("  2. Adjust the similarity_threshold (lower = more permissive)")
print("  3. Add more topic descriptions to improve matching")
print("  4. Consider using more specific topic descriptions")
print()
print("For more examples, see:")
print("  - examples/07_embedding_vs_llm_comparison.py - Compare approaches")
print("  - guardrails/README.md - Full documentation")
print()
