"""
Example 2: PII Redaction Guardrail
===================================

This example demonstrates automatic PII detection and redaction.
The guardrail detects sensitive information (emails, phone numbers, SSNs, etc.)
and automatically redacts it before processing.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.safety_guardrails import PIIDetectionGuardrail


# Define the state
class AgentState(TypedDict):
    input: str
    messages: list
    PIIDetection_result: str
    PIIDetection_reason: str
    PIIDetection_metadata: dict


# Create PII guardrail (with auto-redaction)
pii_guardrail = PIIDetectionGuardrail(
    redact=True,  # Automatically redact PII
    strict=False,  # Don't block, just redact
)


# Simple agent node
def agent_node(state: AgentState) -> AgentState:
    """Process the input (PII will already be redacted)"""
    user_input = state["input"]

    # The input here will have PII redacted if any was found
    response = f"I received your message: '{user_input}'. Your privacy is protected."

    return {
        **state,
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ],
    }


def build_pii_protected_agent():
    """Build an agent with PII redaction guardrail"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("pii_check", pii_guardrail.check)
    workflow.add_node("agent", agent_node)

    # Set entry point
    workflow.set_entry_point("pii_check")

    # PII guardrail always continues (with redacted content)
    # In MODIFY mode, it updates the state and continues
    workflow.add_edge("pii_check", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_pii_protected_agent()

    print("=== Agent with PII Redaction ===\n")

    # Test 1: Input with email
    print("Test 1: Input with email")
    result = graph.invoke({
        "input": "Please contact me at john.doe@example.com for more information."
    })
    print(f"Original input: 'Please contact me at john.doe@example.com for more information.'")
    print(f"Processed input: '{result['input']}'")
    print(f"Guardrail action: {result.get('PIIDetection_result')}")
    print(f"Reason: {result.get('PIIDetection_reason')}")
    print(f"Messages: {result['messages']}")
    print()

    # Test 2: Input with phone number
    print("Test 2: Input with phone number")
    result = graph.invoke({
        "input": "Call me at 555-123-4567 tomorrow."
    })
    print(f"Original input: 'Call me at 555-123-4567 tomorrow.'")
    print(f"Processed input: '{result['input']}'")
    print(f"Guardrail action: {result.get('PIIDetection_result')}")
    print(f"PII found: {result.get('PIIDetection_metadata', {}).get('pii_types')}")
    print()

    # Test 3: Input with multiple PII types
    print("Test 3: Multiple PII types")
    result = graph.invoke({
        "input": "My SSN is 123-45-6789 and my email is user@test.com, call me at 555-999-8888."
    })
    print(f"Original input: 'My SSN is 123-45-6789 and my email is user@test.com, call me at 555-999-8888.'")
    print(f"Processed input: '{result['input']}'")
    print(f"PII types found: {result.get('PIIDetection_metadata', {}).get('pii_types')}")
    print(f"PII count: {result.get('PIIDetection_metadata', {}).get('pii_count')}")
    print()

    # Test 4: Clean input (no PII)
    print("Test 4: Clean input (no PII)")
    result = graph.invoke({
        "input": "What's the weather like today?"
    })
    print(f"Input: '{result['input']}'")
    print(f"Guardrail action: {result.get('PIIDetection_result')}")
    print(f"Reason: {result.get('PIIDetection_reason')}")
    print()

    # Check metrics
    print("PII Guardrail Metrics:")
    print(pii_guardrail.get_metrics())
