"""
Example 5: LLM-Based Safety Guardrail
======================================

This example demonstrates using an LLM to perform semantic content safety checks.
Unlike rule-based guardrails, this can understand context and nuance.

Note: This example requires an LLM provider (OpenAI, Anthropic, etc.)
Set your API key in the environment before running.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.safety_guardrails import ContentSafetyGuardrail

# Uncomment one of these based on your provider:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic


# Define the state
class AgentState(TypedDict):
    input: str
    messages: list
    ContentSafety_result: str
    ContentSafety_reason: str
    ContentSafety_metadata: dict


def create_safety_guardrail():
    """
    Create an LLM-based content safety guardrail.

    Uncomment and configure based on your LLM provider:
    """
    # Option 1: OpenAI
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Option 2: Anthropic
    # llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

    # For demo purposes without API key, we'll use None (falls back to keyword check)
    llm = None

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
    print("PRODUCTION USAGE:")
    print("=" * 60)
    print("""
To use with a real LLM for better safety checking:

1. Install your LLM provider:
   pip install langchain-openai  # for OpenAI
   # OR
   pip install langchain-anthropic  # for Anthropic

2. Set your API key:
   export OPENAI_API_KEY='your-key'
   # OR
   export ANTHROPIC_API_KEY='your-key'

3. Update the create_safety_guardrail() function:
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   return ContentSafetyGuardrail(
       llm=llm,
       safety_categories=["violence", "hate", "sexual", "self-harm", "illegal"],
       threshold=0.7,
   )

Benefits of LLM-based safety:
- Understands context (e.g., "kill" in "kill the process" vs real violence)
- Handles nuanced cases better than keywords
- Can adapt to new safety patterns
- Reduces false positives
    """)

    # Check metrics
    print("\nSafety Guardrail Metrics:")
    print(safety_guardrail.get_metrics())
