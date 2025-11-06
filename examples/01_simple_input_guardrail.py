"""
Example 1: Simple Input Guardrail
==================================

This example shows how to implement a basic topic validation guardrail
that only allows questions about specific topics (weather, in this case).

This is similar to the official LangGraph example but with better structure
and reusable components.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from guardrails.input_guardrails import TopicValidationGuardrail


# Define the state
class AgentState(TypedDict):
    input: str
    messages: list
    TopicValidation_result: str
    TopicValidation_reason: str


# Create the guardrail
weather_guardrail = TopicValidationGuardrail(
    allowed_topics=["weather", "climate", "temperature", "forecast"],
    fuzzy_match=True,
    custom_rejection_message="I can only answer questions about weather and climate.",
)


# Simple agent node (placeholder - would call your actual LLM)
def agent_node(state: AgentState) -> AgentState:
    """Simple agent that responds to weather questions"""
    user_input = state["input"]

    # In a real implementation, this would call your LLM
    response = f"Let me help you with that weather question: '{user_input}'"

    return {
        **state,
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ],
    }


# Build the graph
def build_weather_agent():
    """Build a weather agent with topic validation guardrail"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("guardrail_check", weather_guardrail.check)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reject", weather_guardrail.rejection_node)

    # Set entry point
    workflow.set_entry_point("guardrail_check")

    # Add conditional edge from guardrail
    workflow.add_conditional_edges(
        "guardrail_check",
        weather_guardrail.route,
        {
            "continue": "agent",
            "reject": "reject",
        },
    )

    # Add edges to END
    workflow.add_edge("agent", END)
    workflow.add_edge("reject", END)

    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_weather_agent()

    print("=== Weather Agent with Topic Guardrail ===\n")

    # Test 1: Valid weather question
    print("Test 1: Valid weather question")
    result = graph.invoke({"input": "What's the weather like today?"})
    print(f"Input: {result['input']}")
    print(f"Messages: {result['messages']}")
    print(f"Guardrail result: {result.get('TopicValidation_result')}")
    print()

    # Test 2: Invalid (off-topic) question
    print("Test 2: Off-topic question")
    result = graph.invoke({"input": "What's the capital of France?"})
    print(f"Input: {result['input']}")
    print(f"Messages: {result['messages']}")
    print(f"Guardrail result: {result.get('TopicValidation_result')}")
    print(f"Rejection reason: {result.get('TopicValidation_reason')}")
    print()

    # Test 3: Valid question with different phrasing
    print("Test 3: Valid question (climate related)")
    result = graph.invoke({"input": "Tell me about climate change and temperature rises"})
    print(f"Input: {result['input']}")
    print(f"Messages: {result['messages']}")
    print(f"Guardrail result: {result.get('TopicValidation_result')}")
    print()

    # Check metrics
    print("Guardrail Metrics:")
    print(weather_guardrail.get_metrics())
