"""
Example 3: Output Validation Guardrail
=======================================

This example shows how to validate agent outputs before returning them to users.
Useful for ensuring responses meet quality standards.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.output_guardrails import OutputLengthGuardrail, OutputFormatGuardrail


# Define the state
class AgentState(TypedDict):
    input: str
    messages: list
    output: str
    OutputLength_result: str
    OutputLength_reason: str


# Create output guardrails
length_guardrail = OutputLengthGuardrail(
    min_length=10,  # At least 10 characters
    max_length=200,  # At most 200 characters
    truncate=True,  # Auto-truncate if too long
    count_by="characters",
)


# Agent node that generates responses
def agent_node(state: AgentState) -> AgentState:
    """Generate a response based on input"""
    user_input = state["input"]

    # Simulate different response lengths based on input
    if "short" in user_input.lower():
        response = "OK"  # Too short
    elif "long" in user_input.lower():
        response = (
            "This is a very long response that goes on and on. "
            * 20  # Very long response
        )
    else:
        response = f"Here's a helpful response to your question: {user_input}"

    return {
        **state,
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ],
    }


def retry_generation_node(state: AgentState) -> AgentState:
    """Generate a better response if the first one failed validation"""
    # In a real system, you might modify the prompt or use a different strategy
    return {
        **state,
        "messages": [
            state["messages"][0],  # Keep original user message
            {
                "role": "assistant",
                "content": "I apologize, but I couldn't generate an appropriate response. Please try rephrasing your question.",
            },
        ],
    }


def route_output_validation(state: AgentState) -> str:
    """Route based on output validation result"""
    result = state.get("OutputLength_result")

    if result == "block":
        # Output failed validation, try to regenerate
        return "retry"
    else:
        # Output passed (allow or modify), continue
        return "end"


def build_output_validated_agent():
    """Build an agent with output validation guardrail"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("output_validation", length_guardrail.check)
    workflow.add_node("retry", retry_generation_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Agent -> Output Validation
    workflow.add_edge("agent", "output_validation")

    # Conditional routing from validation
    workflow.add_conditional_edges(
        "output_validation",
        route_output_validation,
        {
            "end": END,
            "retry": "retry",
        },
    )

    # Retry -> END (you could also loop back to agent)
    workflow.add_edge("retry", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_output_validated_agent()

    print("=== Agent with Output Validation ===\n")

    # Test 1: Normal response (should pass)
    print("Test 1: Normal response")
    result = graph.invoke({"input": "What's the weather?"})
    print(f"Input: {result['input']}")
    print(f"Response: {result['messages'][-1]['content']}")
    print(f"Validation result: {result.get('OutputLength_result')}")
    print(f"Reason: {result.get('OutputLength_reason')}")
    print()

    # Test 2: Too short response (should be blocked and retried)
    print("Test 2: Too short response")
    result = graph.invoke({"input": "Give me a short answer"})
    print(f"Input: {result['input']}")
    print(f"Response: {result['messages'][-1]['content']}")
    print(f"Validation result: {result.get('OutputLength_result')}")
    print(f"Reason: {result.get('OutputLength_reason')}")
    print()

    # Test 3: Too long response (should be truncated)
    print("Test 3: Too long response (will be truncated)")
    result = graph.invoke({"input": "Give me a long answer"})
    print(f"Input: {result['input']}")
    response_content = result['messages'][-1]['content']
    print(f"Response length: {len(response_content)} chars")
    print(f"Response (truncated): {response_content[:100]}...")
    print(f"Validation result: {result.get('OutputLength_result')}")
    print(f"Reason: {result.get('OutputLength_reason')}")
    print()

    # Check metrics
    print("Output Length Guardrail Metrics:")
    print(length_guardrail.get_metrics())
