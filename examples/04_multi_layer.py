"""
Example 4: Multi-Layer Guardrails
==================================

This example demonstrates how to combine multiple guardrails in layers
for comprehensive input validation.

Layers:
1. Input length check (cheap, fast)
2. Profanity filter (regex-based, fast)
3. Topic validation (keyword-based, fast)
4. PII detection (pattern-based, moderate)

This layered approach optimizes performance by running cheap checks first.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.base import GuardrailChain
from guardrails.input_guardrails import (
    InputLengthGuardrail,
    ProfanityFilterGuardrail,
    TopicValidationGuardrail,
)
from guardrails.safety_guardrails import PIIDetectionGuardrail


# Define the state
class AgentState(TypedDict):
    input: str
    messages: list
    # Guardrail result fields
    InputLength_result: str
    ProfanityFilter_result: str
    TopicValidation_result: str
    PIIDetection_result: str


# Create individual guardrails
length_guardrail = InputLengthGuardrail(
    min_length=5,
    max_length=1000,
    truncate=False,  # Block if too long
)

profanity_guardrail = ProfanityFilterGuardrail(
    strict_mode=True,
)

topic_guardrail = TopicValidationGuardrail(
    allowed_topics=["technology", "programming", "software", "code", "computer"],
    fuzzy_match=True,
    custom_rejection_message="I can only discuss technology and programming topics.",
)

pii_guardrail = PIIDetectionGuardrail(
    redact=True,
    strict=False,
)

# Create a guardrail chain
guardrail_chain = GuardrailChain([
    length_guardrail,
    profanity_guardrail,
    topic_guardrail,
    pii_guardrail,
])


# Agent node
def agent_node(state: AgentState) -> AgentState:
    """Process the validated (and possibly modified) input"""
    user_input = state["input"]

    response = f"Great question about technology! Regarding '{user_input}', here's what I know..."

    return {
        **state,
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ],
    }


def build_multi_layer_agent():
    """Build an agent with multiple layered guardrails"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("guardrail_chain", guardrail_chain.check)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reject", guardrail_chain.rejection_node)

    # Set entry point
    workflow.set_entry_point("guardrail_chain")

    # Conditional routing from guardrail chain
    workflow.add_conditional_edges(
        "guardrail_chain",
        guardrail_chain.route,
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
    graph = build_multi_layer_agent()

    print("=== Agent with Multi-Layer Guardrails ===\n")

    # Test 1: Valid input (passes all guardrails)
    print("Test 1: Valid input")
    result = graph.invoke({"input": "How do I learn Python programming?"})
    print(f"Input: {result['input']}")
    print(f"Length check: {result.get('InputLength_result')}")
    print(f"Profanity check: {result.get('ProfanityFilter_result')}")
    print(f"Topic check: {result.get('TopicValidation_result')}")
    print(f"PII check: {result.get('PIIDetection_result')}")
    print(f"Response: {result['messages'][-1]['content'][:100]}...")
    print()

    # Test 2: Too short input (fails first guardrail)
    print("Test 2: Too short input")
    result = graph.invoke({"input": "Hi"})
    print(f"Input: {result['input']}")
    print(f"Length check: {result.get('InputLength_result')}")
    print(f"Rejected: {result.get('rejected', False)}")
    if result.get('rejected'):
        print(f"Rejection message: {result['messages'][-1]['content']}")
    print()

    # Test 3: Off-topic input (fails topic guardrail)
    print("Test 3: Off-topic input")
    result = graph.invoke({"input": "What's the best recipe for chocolate cake?"})
    print(f"Input: {result['input']}")
    print(f"Length check: {result.get('InputLength_result')}")
    print(f"Topic check: {result.get('TopicValidation_result')}")
    print(f"Rejected: {result.get('rejected', False)}")
    if result.get('rejected'):
        print(f"Rejection message: {result['messages'][-1]['content']}")
    print()

    # Test 4: Input with PII (passes but gets modified)
    print("Test 4: Input with PII")
    original_input = "How do I send an email to user@example.com using Python?"
    result = graph.invoke({"input": original_input})
    print(f"Original input: '{original_input}'")
    print(f"Modified input: '{result['input']}'")
    print(f"PII check: {result.get('PIIDetection_result')}")
    print(f"PII reason: {result.get('PIIDetection_reason')}")
    print()

    # Print metrics for each guardrail
    print("\n=== Guardrail Metrics ===")
    print("\n1. Length Guardrail:")
    print(length_guardrail.get_metrics())
    print("\n2. Profanity Guardrail:")
    print(profanity_guardrail.get_metrics())
    print("\n3. Topic Guardrail:")
    print(topic_guardrail.get_metrics())
    print("\n4. PII Guardrail:")
    print(pii_guardrail.get_metrics())
