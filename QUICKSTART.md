# Quick Start Guide

Get up and running with LangGraph guardrails in 5 minutes.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd langgraph-guardrails-example

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install with optional dependencies
pip install -e ".[all]"  # Install all optional dependencies
pip install -e ".[openai]"  # Just OpenAI
pip install -e ".[dev]"  # Development dependencies
```

## Your First Guardrail in 3 Steps

### Step 1: Import and Create a Guardrail

```python
from guardrails.input_guardrails import TopicValidationGuardrail

# Create a guardrail that only allows questions about cooking
cooking_guardrail = TopicValidationGuardrail(
    allowed_topics=["cooking", "recipe", "food", "baking"],
    fuzzy_match=True,
)
```

### Step 2: Build Your Graph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    input: str
    messages: list
    TopicValidation_result: str

workflow = StateGraph(State)

# Add nodes
workflow.add_node("guardrail", cooking_guardrail.check)
workflow.add_node("reject", cooking_guardrail.rejection_node)
workflow.add_node("agent", your_agent_function)  # Your actual agent

# Set routing
workflow.set_entry_point("guardrail")
workflow.add_conditional_edges(
    "guardrail",
    cooking_guardrail.route,
    {"continue": "agent", "reject": "reject"}
)
workflow.add_edge("agent", END)
workflow.add_edge("reject", END)

graph = workflow.compile()
```

### Step 3: Use It

```python
# Valid input - will pass
result = graph.invoke({"input": "How do I bake a cake?"})
print(result["messages"])

# Invalid input - will be rejected
result = graph.invoke({"input": "What's the weather?"})
print(result["messages"])
```

## Run the Examples

Try the included examples to see different patterns:

```bash
# Simple topic validation
python examples/01_simple_input_guardrail.py

# PII detection and redaction
python examples/02_pii_redaction.py

# Output validation
python examples/03_output_validation.py

# Multi-layer guardrails
python examples/04_multi_layer.py

# LLM-based safety (requires API key)
python examples/05_llm_based_safety.py

# Production-ready example
python examples/06_production_example.py
```

## Available Guardrails

### Input Guardrails (Pre-processing)
- `TopicValidationGuardrail` - Validate input topics
- `ProfanityFilterGuardrail` - Filter profanity
- `InputLengthGuardrail` - Enforce length limits
- `RateLimitGuardrail` - Rate limiting per user
- `InputFormatGuardrail` - Regex pattern validation
- `LanguageDetectionGuardrail` - Language validation

### Output Guardrails (Post-processing)
- `OutputFormatGuardrail` - Validate response format
- `OutputLengthGuardrail` - Enforce output length
- `FactualityGuardrail` - Check factual accuracy
- `SensitiveContentFilterGuardrail` - Filter sensitive data

### Safety Guardrails (Both)
- `PIIDetectionGuardrail` - Detect/redact PII
- `ContentSafetyGuardrail` - LLM-based safety check
- `ToxicityGuardrail` - Toxicity detection
- `PromptInjectionGuardrail` - Prevent prompt injection
- `CodeExecutionGuardrail` - Validate code safety

## Common Patterns

### Pattern 1: Chain Multiple Guardrails

```python
from guardrails.base import GuardrailChain

chain = GuardrailChain([
    InputLengthGuardrail(max_length=1000),
    ProfanityFilterGuardrail(),
    TopicValidationGuardrail(allowed_topics=["tech"]),
])

workflow.add_node("validation", chain.check)
```

### Pattern 2: Conditional Guardrails

```python
from guardrails.base import ConditionalGuardrail

# Only check for PII for non-premium users
conditional = ConditionalGuardrail(
    condition=lambda state: state.get("user_tier") != "premium",
    guardrail=PIIDetectionGuardrail(),
)
```

### Pattern 3: Custom Guardrail

```python
from guardrails.base import BaseGuardrail, GuardrailResult, GuardrailAction

class MyCustomGuardrail(BaseGuardrail):
    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        if "forbidden_word" in content.lower():
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Contains forbidden word"
            )
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            reason="Validation passed"
        )
```

## Using with LLMs

For LLM-based guardrails (like `ContentSafetyGuardrail`):

```python
from langchain_openai import ChatOpenAI
from guardrails.safety_guardrails import ContentSafetyGuardrail

# Set your API key
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Create LLM-based guardrail
safety_guardrail = ContentSafetyGuardrail(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    safety_categories=["violence", "hate", "sexual"],
    threshold=0.7
)
```

## Monitoring and Metrics

Track guardrail performance:

```python
# Get metrics from any guardrail
metrics = guardrail.get_metrics()
print(f"Total checks: {metrics['total_checks']}")
print(f"Block rate: {metrics['block_rate']:.2%}")
print(f"Pass rate: {metrics['pass_rate']:.2%}")
```

## Next Steps

1. **Explore the Examples** - See `/examples` directory for complete working examples
2. **Read the Full README** - Comprehensive guide with best practices
3. **Customize Guardrails** - Extend `BaseGuardrail` for custom validation logic
4. **Deploy to Production** - See `examples/06_production_example.py` for production setup

## Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install -e .
```

**Missing dependencies for optional features:**
```bash
pip install langdetect  # For language detection
pip install detoxify    # For toxicity detection
```

**LLM-based guardrails not working:**
- Ensure you've set your API key environment variable
- Install the appropriate LLM provider package (`langchain-openai`, `langchain-anthropic`)

## Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Report bugs or request features via GitHub Issues
- **Examples**: Check the `/examples` directory for working code

Happy building! 🛡️
