"""
Example 7: Custom LLM-Based Guardrails
=======================================

This example demonstrates using flexible LLM-based guardrails with custom prompts.
Instead of hardcoded safety categories, you can define your own validation logic
using natural language instructions.

This is more flexible than keyword matching and can handle nuanced, context-aware
validation that rule-based systems struggle with.

Setup: Add ANTHROPIC_API_KEY to .env file (or it will show examples without calling LLM)
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from guardrails.llm_guardrails import (
    LLMGuardrail,
    BrandSafetyGuardrail,
    ToneGuardrail,
)
from guardrails.config import create_anthropic_llm, get_anthropic_api_key


# Define the state
class AgentState(TypedDict):
    input: str
    messages: list


def agent_node(state: AgentState) -> AgentState:
    """Simple agent that echoes input"""
    return {
        **state,
        "messages": [
            {"role": "user", "content": state["input"]},
            {"role": "assistant", "content": f"Processing: {state['input']}"},
        ],
    }


print("=" * 70)
print("CUSTOM LLM-BASED GUARDRAILS EXAMPLES")
print("=" * 70)
print()

# Check if we have an API key
llm = None
if get_anthropic_api_key():
    try:
        llm = create_anthropic_llm()
        print("✓ Using Claude 3.5 Haiku for custom guardrails")
        print()
    except Exception as e:
        print(f"⚠ Could not initialize Claude: {e}")
        print("Showing examples without running them")
        print()
else:
    print("ℹ No ANTHROPIC_API_KEY found")
    print("Showing example configurations without running them")
    print("To run these examples:")
    print("  1. Copy .env.example to .env")
    print("  2. Add your Anthropic API key")
    print("  3. Run this example again")
    print()


# Example 1: Custom Policy Guardrail
print("-" * 70)
print("Example 1: Custom Company Policy Guardrail")
print("-" * 70)
print()

custom_policy_guardrail = LLMGuardrail(
    llm=llm,
    system_prompt="You are a compliance checker for ACME Corporation.",
    instruction_template="""Check if this content violates our company policies:

1. No discussion of competitors' products
2. No sharing of internal financial information
3. No promises about future features or timelines
4. Must maintain professional, respectful tone

Content to check:
{content}

Respond with JSON:
{{
    "is_safe": true/false,
    "reason": "explanation of any policy violations",
    "confidence": 0.0-1.0
}}""",
    threshold=0.7,
    name="CompanyPolicy",
)

print("Configuration:")
print("- Checks against specific company policies")
print("- Uses custom prompt tailored to business needs")
print("- No hardcoded keywords - LLM understands context")
print()

test_inputs_1 = [
    "Our new feature will launch next month!",  # Policy violation: future promises
    "How can I help you today?",  # OK
    "Compare us to Competitor X",  # Policy violation: competitors
]

if llm:
    workflow = StateGraph(AgentState)
    workflow.add_node("guardrail", custom_policy_guardrail.check)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reject", custom_policy_guardrail.rejection_node)
    workflow.set_entry_point("guardrail")
    workflow.add_conditional_edges(
        "guardrail",
        custom_policy_guardrail.route,
        {"continue": "agent", "reject": "reject"},
    )
    workflow.add_edge("agent", END)
    workflow.add_edge("reject", END)
    graph = workflow.compile()

    for test_input in test_inputs_1:
        print(f"Testing: '{test_input}'")
        result = graph.invoke({"input": test_input})
        status = "BLOCKED" if result.get("rejected") else "ALLOWED"
        print(f"  Result: {status}")
        if result.get("CompanyPolicy_reason"):
            print(f"  Reason: {result['CompanyPolicy_reason']}")
        print()
else:
    print("Sample test inputs:")
    for inp in test_inputs_1:
        print(f"  - {inp}")
    print()


# Example 2: Brand Safety Guardrail
print("-" * 70)
print("Example 2: Brand Safety Guardrail")
print("-" * 70)
print()

brand_guardrail = BrandSafetyGuardrail(
    llm=llm,
    brand_values=[
        "Inclusive and welcoming to all",
        "Educational and informative",
        "Encouraging and positive",
    ],
    prohibited_topics=["politics", "religion", "personal medical advice"],
    threshold=0.8,
)

print("Configuration:")
print("- Validates against brand values")
print("- Blocks prohibited topics")
print("- Context-aware (e.g., medical info vs. medical advice)")
print()

test_inputs_2 = [
    "I support political party X",  # Prohibited topic
    "Great question! Here's how to learn Python...",  # OK - educational
    "You should definitely take this medication",  # Prohibited - medical advice
    "Many people find exercise helpful for mood",  # OK - general health info
]

if llm:
    workflow = StateGraph(AgentState)
    workflow.add_node("guardrail", brand_guardrail.check)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reject", brand_guardrail.rejection_node)
    workflow.set_entry_point("guardrail")
    workflow.add_conditional_edges(
        "guardrail",
        brand_guardrail.route,
        {"continue": "agent", "reject": "reject"},
    )
    workflow.add_edge("agent", END)
    workflow.add_edge("reject", END)
    graph = workflow.compile()

    for test_input in test_inputs_2:
        print(f"Testing: '{test_input[:50]}...'")
        result = graph.invoke({"input": test_input})
        status = "BLOCKED" if result.get("rejected") else "ALLOWED"
        print(f"  Result: {status}")
        print()
else:
    print("Sample test inputs:")
    for inp in test_inputs_2:
        print(f"  - {inp[:60]}...")
    print()


# Example 3: Tone Validation Guardrail
print("-" * 70)
print("Example 3: Tone Validation Guardrail")
print("-" * 70)
print()

tone_guardrail = ToneGuardrail(
    llm=llm,
    desired_tone="friendly, helpful, and professional",
    disallowed_tones=["sarcastic", "condescending", "aggressive", "dismissive"],
    threshold=0.7,
)

print("Configuration:")
print("- Ensures responses match desired tone")
print("- Detects and blocks inappropriate tones")
print("- More nuanced than sentiment analysis")
print()

test_inputs_3 = [
    "Obviously, you should know this already.",  # Condescending
    "I'd be happy to help you with that!",  # Good tone
    "Why don't you just Google it?",  # Dismissive
    "Let me explain how this works...",  # Good tone
]

if llm:
    workflow = StateGraph(AgentState)
    workflow.add_node("guardrail", tone_guardrail.check)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reject", tone_guardrail.rejection_node)
    workflow.set_entry_point("guardrail")
    workflow.add_conditional_edges(
        "guardrail",
        tone_guardrail.route,
        {"continue": "agent", "reject": "reject"},
    )
    workflow.add_edge("agent", END)
    workflow.add_edge("reject", END)
    graph = workflow.compile()

    for test_input in test_inputs_3:
        print(f"Testing: '{test_input}'")
        result = graph.invoke({"input": test_input})
        status = "BLOCKED" if result.get("rejected") else "ALLOWED"
        print(f"  Result: {status}")
        print()
else:
    print("Sample test inputs:")
    for inp in test_inputs_3:
        print(f"  - {inp}")
    print()


# Example 4: Completely Custom Validation
print("-" * 70)
print("Example 4: Domain-Specific Custom Validation")
print("-" * 70)
print()

# Example: Code review guardrail
code_review_guardrail = LLMGuardrail(
    llm=llm,
    system_prompt="You are a senior software engineer reviewing code suggestions.",
    instruction_template="""Review this code-related content for quality and safety:

Requirements:
- No suggestions that could introduce security vulnerabilities
- No recommendations to disable security features
- Must follow Python best practices if code is included
- Should be helpful and educational

Content:
{content}

Respond with JSON:
{{
    "is_safe": true/false,
    "reason": "explanation",
    "confidence": 0.0-1.0,
    "concerns": ["list any specific concerns"]
}}""",
    threshold=0.8,
    name="CodeReview",
)

print("Configuration:")
print("- Custom domain-specific validation (code review)")
print("- Checks for security issues")
print("- Validates best practices")
print()

test_inputs_4 = [
    "Just disable SSL certificate verification",  # Security issue
    "Here's how to implement input validation...",  # Good
    "Run this command with sudo chmod 777",  # Security issue
    "Use parameterized queries to prevent SQL injection",  # Good
]

if llm:
    print("Testing code review guardrail:")
    for test_input in test_inputs_4:
        print(f"\nInput: '{test_input}'")
        result = code_review_guardrail.validate(test_input)
        print(f"  Action: {result.action.value}")
        print(f"  Reason: {result.reason}")
else:
    print("Sample test inputs:")
    for inp in test_inputs_4:
        print(f"  - {inp}")
    print()


# Summary
print()
print("=" * 70)
print("KEY BENEFITS OF CUSTOM LLM GUARDRAILS")
print("=" * 70)
print("""
✓ Flexible: Define validation logic in natural language
✓ Context-aware: Understands nuance better than keywords
✓ Maintainable: Update rules by changing prompts, not code
✓ Powerful: Can handle complex, domain-specific validation
✓ Fast: Claude 3.5 Haiku provides sub-2-second validation

When to use LLM guardrails vs. keyword matching:

Use LLM guardrails when:
- You need context-aware validation (e.g., distinguishing medical info from advice)
- Rules are complex or nuanced
- You want to validate tone, style, or intent
- Domain expertise is needed (e.g., code review, legal compliance)

Use keyword/pattern matching when:
- Rules are simple and explicit (e.g., profanity filtering)
- Speed is critical (< 100ms required)
- You want deterministic, explainable results
- Cost is a major constraint

Best practice: Layer both! Use fast keyword checks first, then LLM checks for
cases that pass the initial filter. See example 06 for production patterns.
""")

print("\nGuardrail Metrics:")
if llm:
    print(f"Custom Policy: {custom_policy_guardrail.get_metrics()}")
    print(f"Brand Safety: {brand_guardrail.get_metrics()}")
    print(f"Tone Validation: {tone_guardrail.get_metrics()}")
else:
    print("(No metrics - API key not configured)")
