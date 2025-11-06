"""
Generate Mermaid diagrams for examples.
Run from repository root: python generate_diagrams.py
"""

import sys
from pathlib import Path

# Import modules
from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.visualization import visualize_graph
from guardrails.input_guardrails import InputLengthGuardrail, RateLimitGuardrail
from guardrails.safety_guardrails import PIIDetectionGuardrail
from guardrails.output_guardrails import OutputLengthGuardrail
from guardrails.base import GuardrailChain

# Create diagrams directory
Path("docs/diagrams").mkdir(parents=True, exist_ok=True)

print("Generating Mermaid diagrams for example workflows...")
print()

# Example 1: Simple Input Guardrail
print("1. Simple Input Guardrail (Length Check)")

class State1(TypedDict):
    input: str
    messages: list

length_guardrail = InputLengthGuardrail(
    min_length=5,
    max_length=1000,
)

workflow1 = StateGraph(State1)
workflow1.add_node("guardrail_check", lambda s: s)
workflow1.add_node("agent", lambda s: s)
workflow1.add_node("reject", lambda s: s)
workflow1.set_entry_point("guardrail_check")
workflow1.add_conditional_edges(
    "guardrail_check",
    lambda s: "continue",
    {"continue": "agent", "reject": "reject"},
)
workflow1.add_edge("agent", END)
workflow1.add_edge("reject", END)
graph1 = workflow1.compile()

visualize_graph(graph1, "docs/diagrams/01_simple_input_guardrail.mermaid", format="mermaid")
print()

# Example 2: PII Redaction
print("2. PII Detection and Redaction")

class State2(TypedDict):
    input: str
    messages: list

workflow2 = StateGraph(State2)
workflow2.add_node("pii_check", lambda s: s)
workflow2.add_node("agent", lambda s: s)
workflow2.set_entry_point("pii_check")
workflow2.add_edge("pii_check", "agent")
workflow2.add_edge("agent", END)
graph2 = workflow2.compile()

visualize_graph(graph2, "docs/diagrams/02_pii_redaction.mermaid", format="mermaid")
print()

# Example 3: Output Validation
print("3. Output Validation with Retry")

class State3(TypedDict):
    input: str
    messages: list
    output: str

workflow3 = StateGraph(State3)
workflow3.add_node("agent", lambda s: s)
workflow3.add_node("output_validation", lambda s: s)
workflow3.add_node("retry", lambda s: s)
workflow3.set_entry_point("agent")
workflow3.add_edge("agent", "output_validation")
workflow3.add_conditional_edges(
    "output_validation",
    lambda s: "end",
    {"end": END, "retry": "retry"},
)
workflow3.add_edge("retry", END)
graph3 = workflow3.compile()

visualize_graph(graph3, "docs/diagrams/03_output_validation.mermaid", format="mermaid")
print()

# Example 4: Multi-Layer Guardrails
print("4. Multi-Layer Guardrail Chain")

class State4(TypedDict):
    input: str
    messages: list

workflow4 = StateGraph(State4)
workflow4.add_node("guardrail_chain", lambda s: s)
workflow4.add_node("agent", lambda s: s)
workflow4.add_node("reject", lambda s: s)
workflow4.set_entry_point("guardrail_chain")
workflow4.add_conditional_edges(
    "guardrail_chain",
    lambda s: "continue",
    {"continue": "agent", "reject": "reject"},
)
workflow4.add_edge("agent", END)
workflow4.add_edge("reject", END)
graph4 = workflow4.compile()

visualize_graph(graph4, "docs/diagrams/04_multi_layer.mermaid", format="mermaid")
print()

# Example 5: Production Setup
print("5. Production Multi-Guardrail System")

class State5(TypedDict):
    input: str
    messages: list

workflow5 = StateGraph(State5)
workflow5.add_node("input_validation", lambda s: s)
workflow5.add_node("input_rejection", lambda s: s)
workflow5.add_node("agent", lambda s: s)
workflow5.add_node("output_validation", lambda s: s)
workflow5.add_node("error_handler", lambda s: s)
workflow5.set_entry_point("input_validation")
workflow5.add_conditional_edges(
    "input_validation",
    lambda s: "continue",
    {"continue": "agent", "reject": "input_rejection"},
)
workflow5.add_edge("input_rejection", END)
workflow5.add_edge("agent", "output_validation")
workflow5.add_conditional_edges(
    "output_validation",
    lambda s: "success",
    {"success": END, "error": "error_handler"},
)
workflow5.add_edge("error_handler", END)
graph5 = workflow5.compile()

visualize_graph(graph5, "docs/diagrams/06_production.mermaid", format="mermaid")
print()

print("=" * 60)
print("Done! Mermaid diagrams saved to docs/diagrams/")
print()
print("You can:")
print("1. View .mermaid files in VS Code with Mermaid extension")
print("2. Paste content into https://mermaid.live for rendering")
print("3. Use Mermaid in Markdown documentation")
print("=" * 60)
