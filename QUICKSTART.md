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

# Optional: Set up for LLM-based guardrails (Claude 4.5 Haiku)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY from https://console.anthropic.com/

# Or install with optional dependencies
pip install -e ".[all]"  # Install all optional dependencies
pip install -e ".[openai]"  # Just OpenAI
pip install -e ".[dev]"  # Development dependencies
```

## Your First Guardrail in 3 Steps

### Step 1: Import and Create a Guardrail

```python
from guardrails.input_guardrails import InputLengthGuardrail

# Create a guardrail that limits input length
length_guardrail = InputLengthGuardrail(
    min_length=5,
    max_length=500,
    name="LengthCheck"
)
```

### Step 2: Build Your Graph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    input: str
    messages: list

workflow = StateGraph(State)

# Add nodes
workflow.add_node("length_check", length_guardrail.check)
workflow.add_node("rejection", length_guardrail.rejection_node)
workflow.add_node("agent", your_agent_function)  # Your actual agent

# Set routing
workflow.set_entry_point("length_check")
workflow.add_conditional_edges(
    "length_check",
    length_guardrail.route,
    {"allow": "agent", "block": "rejection"}
)
workflow.add_edge("agent", END)
workflow.add_edge("rejection", END)

graph = workflow.compile()
```

### Step 3: Use It

```python
# Valid input - will pass
result = graph.invoke({"input": "How do I bake a chocolate cake?"})
print(result["messages"])

# Invalid input - will be rejected (too short)
result = graph.invoke({"input": "Hi"})
print(result["messages"])
```

## Run the Examples

Try the included examples to see different patterns:

```bash
# PII detection and redaction
python examples/01_pii_redaction.py

# Output validation
python examples/02_output_validation.py

# LLM-based safety (requires API key)
python examples/03_llm_based_safety.py

# Production-ready example
python examples/04_production_example.py

# Custom LLM guardrails
python examples/05_custom_llm_guardrails.py

# LLM vs keyword topic validation
python examples/06_semantic_topic_validation.py

# Embedding vs LLM comparison
python examples/07_embedding_vs_llm_comparison.py

# LM Studio for local embeddings
python examples/08_lm_studio_embeddings.py
```

## Available Guardrails

### Input Guardrails (Pre-processing)
- `InputLengthGuardrail` - Enforce length limits
- `RateLimitGuardrail` - Rate limiting per user

### Output Guardrails (Post-processing)
- `OutputFormatGuardrail` - Validate response format
- `OutputLengthGuardrail` - Enforce output length

### Safety Guardrails
- `PIIDetectionGuardrail` - Detect/redact PII
- `ContentSafetyGuardrail` - LLM-based safety check
- `PromptInjectionGuardrail` - Prevent prompt injection
- `CodeExecutionGuardrail` - Validate code safety

### Embedding-Based Guardrails (Fast Semantic Validation)
- `SemanticSimilarityGuardrail` - Topic validation via embeddings
- `MultiModalSemanticGuardrail` - Allow/block lists with embeddings

### LLM-Based Guardrails (Flexible, Context-Aware)
- `LLMGuardrail` - Custom prompt-based validation
- `BrandSafetyGuardrail` - Brand values enforcement
- `ToneGuardrail` - Tone/style validation
- `FactualAccuracyGuardrail` - Fact-checking with knowledge base

## Common Patterns

### Pattern 1: Chain Multiple Guardrails

```python
from guardrails.base import GuardrailChain
from guardrails.input_guardrails import InputLengthGuardrail, RateLimitGuardrail
from guardrails.safety_guardrails import PIIDetectionGuardrail

chain = GuardrailChain([
    InputLengthGuardrail(max_length=1000),
    RateLimitGuardrail(max_requests_per_hour=100),
    PIIDetectionGuardrail(redact=True),
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

For LLM-based guardrails with Claude 4.5 Haiku (recommended):

```python
from guardrails.safety_guardrails import ContentSafetyGuardrail
from guardrails.config import create_anthropic_llm

# Set your ANTHROPIC_API_KEY in .env file, then:
llm = create_anthropic_llm()  # Uses Claude 4.5 Haiku by default

# Create LLM-based guardrail
safety_guardrail = ContentSafetyGuardrail(
    llm=llm,
    safety_categories=["violence", "hate", "sexual"],
    threshold=0.7
)
```

**Why Claude 4.5 Haiku?**
- Fastest: Sub-2 second latency for real-time validation
- Cost-effective: $1 per million input tokens
- Powerful: Matches Sonnet 4 on coding and agent tasks
- Excellent at classification and safety tasks
- Easy setup with .env file

Alternative: Use OpenAI by setting `OPENAI_API_KEY` and calling `create_openai_llm()` from `guardrails.config`

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
4. **Deploy to Production** - See `examples/04_production_example.py` for production setup

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
