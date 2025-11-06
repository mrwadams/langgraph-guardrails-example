"""
Example 5: LLM-Based Safety Guardrail
======================================

This example demonstrates using an LLM to perform semantic content safety checks.
Unlike rule-based guardrails, this can understand context and nuance.

This example uses Claude 4.5 Haiku - fast, cost-effective, and accurate for guardrails.

Setup:
1. Copy .env.example to .env
2. Add your Anthropic API key to .env
3. Run this example

If no API key is set, it falls back to keyword-based checking.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.safety_guardrails import ContentSafetyGuardrail
from guardrails.config import create_anthropic_llm, get_anthropic_api_key


# Define the state
class AgentState(TypedDict):
    input: str
    messages: list
    ContentSafety_result: str
    ContentSafety_reason: str
    ContentSafety_metadata: dict


def create_safety_guardrail():
    """
    Create an LLM-based content safety guardrail using Claude 4.5 Haiku.

    Falls back to keyword-based checking if no API key is set.
    """
    llm = None

    # Try to create LLM if API key is available
    if get_anthropic_api_key():
        try:
            llm = create_anthropic_llm()
            print("✓ Using Claude 4.5 Haiku for LLM-based safety checking\n")
        except Exception as e:
            print(f"⚠ Could not initialize Claude: {e}")
            print("Falling back to keyword-based checking\n")
    else:
        print("ℹ No ANTHROPIC_API_KEY found in environment")
        print("Using keyword-based fallback. For better results:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Anthropic API key")
        print("  3. Run this example again\n")

    return ContentSafetyGuardrail(
        llm=llm,
        safety_categories=["violence", "hate", "sexual", "self-harm", "illegal"],
        threshold=0.7,
    )


safety_guardrail = create_safety_guardrail()


# Agent node
def agent_node(state: AgentState) -> AgentState:
    """Process safe input"""
    user_input = state["input"]
    response = f"Thank you for your appropriate question: '{user_input}'. Let me help you with that."

    return {
        **state,
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ],
    }


def build_safety_checked_agent():
    """Build an agent with LLM-based safety checking"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("safety_check", safety_guardrail.check)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reject", safety_guardrail.rejection_node)

    # Set entry point
    workflow.set_entry_point("safety_check")

    # Conditional routing from safety check
    workflow.add_conditional_edges(
        "safety_check",
        safety_guardrail.route,
        {
            "continue": "agent",
            "reject": "reject",
        },
    )

    # Add edges to END
    workflow.add_edge("agent", END)
    workflow.add_edge("reject", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_safety_checked_agent()

    print("=== Agent with LLM-Based Safety Checking ===")
    print("Note: This example uses keyword-based fallback without an LLM API key.\n")

    # Test 1: Safe input
    print("Test 1: Safe, appropriate input")
    result = graph.invoke({"input": "How do I bake a cake?"})
    print(f"Input: {result['input']}")
    print(f"Safety result: {result.get('ContentSafety_result')}")
    print(f"Reason: {result.get('ContentSafety_reason')}")
    print(f"Rejected: {result.get('rejected', False)}")
    print()

    # Test 2: Potentially unsafe input (violence keywords)
    print("Test 2: Input with violence keywords")
    result = graph.invoke({"input": "How to attack someone?"})
    print(f"Input: {result['input']}")
    print(f"Safety result: {result.get('ContentSafety_result')}")
    print(f"Reason: {result.get('ContentSafety_reason')}")
    print(f"Rejected: {result.get('rejected', False)}")
    if result.get('rejected'):
        print(f"Rejection message: {result['messages'][-1]['content']}")
    print()

    # Test 3: Borderline content
    print("Test 3: Borderline content (video game violence)")
    result = graph.invoke({
        "input": "What are the best weapons in Call of Duty?"
    })
    print(f"Input: {result['input']}")
    print(f"Safety result: {result.get('ContentSafety_result')}")
    print(f"Reason: {result.get('ContentSafety_reason')}")
    print(
        "Note: An LLM-based check would understand this is about a video game "
        "and allow it. Keyword-based check might flag it."
    )
    print()

    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS:")
    print("=" * 60)
    print("""
To enable LLM-based safety checking with Claude 4.5 Haiku:

1. Copy the example environment file:
   cp .env.example .env

2. Get your Anthropic API key:
   https://console.anthropic.com/

3. Add your API key to .env:
   ANTHROPIC_API_KEY=your_api_key_here

4. Run this example again:
   python examples/05_llm_based_safety.py

Why Claude 4.5 Haiku for guardrails?
- Fastest: Sub-2 second latency for real-time validation
- Cost-effective: $1 per million input tokens
- Powerful: Matches Sonnet 4 performance on coding/agent tasks
- Accurate: Excellent at classification and safety tasks
- Reliable: High uptime and consistent performance

Benefits of LLM-based safety over keyword matching:
- Understands context (e.g., "kill" in "kill the process" vs real violence)
- Handles nuanced cases better than keywords
- Can adapt to new safety patterns
- Significantly reduces false positives
- Better at detecting subtle manipulation attempts

Alternative: You can also use OpenAI models by setting OPENAI_API_KEY
and using create_openai_llm() from guardrails.config
    """)

    # Check metrics
    print("\nSafety Guardrail Metrics:")
    print(safety_guardrail.get_metrics())
