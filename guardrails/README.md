# Guardrails Module

This directory contains reusable guardrail components for LangGraph workflows. Each guardrail validates or transforms content at different points in your graph.

## Quick Reference

| Guardrail | File | Type | Use Case |
|-----------|------|------|----------|
| `BaseGuardrail` | base.py | Base Class | Extend to create custom guardrails |
| `GuardrailChain` | base.py | Combiner | Chain multiple guardrails together |
| `ConditionalGuardrail` | base.py | Wrapper | Apply guardrails conditionally |
| `TopicValidationGuardrail` | input_guardrails.py | Input | Validate input topics |
| `ProfanityFilterGuardrail` | input_guardrails.py | Input | Filter profanity |
| `InputLengthGuardrail` | input_guardrails.py | Input | Enforce length limits |
| `RateLimitGuardrail` | input_guardrails.py | Input | Rate limit by user |
| `InputFormatGuardrail` | input_guardrails.py | Input | Regex format validation |
| `LanguageDetectionGuardrail` | input_guardrails.py | Input | Validate language |
| `OutputFormatGuardrail` | output_guardrails.py | Output | Validate response format |
| `OutputLengthGuardrail` | output_guardrails.py | Output | Enforce output length |
| `FactualityGuardrail` | output_guardrails.py | Output | Check factual accuracy |
| `NoHallucinationGuardrail` | output_guardrails.py | Output | Detect hallucinations |
| `SensitiveContentFilterGuardrail` | output_guardrails.py | Output | Filter sensitive data |
| `PIIDetectionGuardrail` | safety_guardrails.py | Safety | Detect/redact PII |
| `ContentSafetyGuardrail` | safety_guardrails.py | Safety | Content moderation (LLM) |
| `ToxicityGuardrail` | safety_guardrails.py | Safety | Toxicity detection |
| `PromptInjectionGuardrail` | safety_guardrails.py | Safety | Prevent prompt injection |
| `CodeExecutionGuardrail` | safety_guardrails.py | Safety | Code safety validation |
| `LLMGuardrail` | llm_guardrails.py | LLM-Based | Custom prompt-based validation |
| `BrandSafetyGuardrail` | llm_guardrails.py | LLM-Based | Brand values enforcement |
| `ToneGuardrail` | llm_guardrails.py | LLM-Based | Tone/style validation |
| `FactualAccuracyGuardrail` | llm_guardrails.py | LLM-Based | Fact-checking with knowledge base |

## Files Overview

### `base.py` - Core Infrastructure

Foundation classes for building guardrails.

**Classes:**
- **`BaseGuardrail`** - Abstract base class for all guardrails
  - Provides common interface: `validate()`, `check()`, `route()`, `rejection_node()`
  - Includes metrics tracking
  - Handles state updates automatically

- **`GuardrailChain`** - Combine multiple guardrails
  - Executes guardrails in sequence
  - Stops at first failure
  - Use for layered validation (cheap checks first, then expensive ones)

- **`ConditionalGuardrail`** - Apply guardrails conditionally
  - Only runs guardrail if condition is met
  - Example: Only check PII for non-premium users

- **`GuardrailResult`** - Standardized result object
- **`GuardrailAction`** - Enum: ALLOW, BLOCK, MODIFY, WARN

**When to use:** Extend `BaseGuardrail` to create custom guardrails with specific validation logic.

---

### `input_guardrails.py` - Input Validation

Pre-process and validate user inputs before they reach your agent.

**Guardrails:**

1. **`TopicValidationGuardrail`** - Restrict to allowed topics
   - **Use case**: Chatbot that only answers questions about specific domains
   - **Example**: Weather bot that blocks non-weather questions
   - **Features**: Fuzzy matching, custom rejection messages

2. **`ProfanityFilterGuardrail`** - Filter inappropriate language
   - **Use case**: Family-friendly applications
   - **Features**: Custom word lists, strict/lenient modes

3. **`InputLengthGuardrail`** - Enforce min/max input length
   - **Use case**: Prevent too-short or too-long inputs
   - **Features**: Character or token counting, auto-truncation option

4. **`RateLimitGuardrail`** - Limit requests per user/session
   - **Use case**: Prevent abuse, manage costs
   - **Features**: Per-minute and per-hour limits, user-based tracking

5. **`InputFormatGuardrail`** - Validate input format with regex
   - **Use case**: Ensure inputs match expected patterns (email, phone, etc.)
   - **Features**: Custom regex patterns, invert matching

6. **`LanguageDetectionGuardrail`** - Validate input language
   - **Use case**: Only process specific languages
   - **Features**: Multiple language support (requires langdetect)

**When to use:** Add as early nodes in your graph to filter/validate before processing.

---

### `output_guardrails.py` - Output Validation

Validate and filter agent responses before returning to users.

**Guardrails:**

1. **`OutputFormatGuardrail`** - Validate response format
   - **Use case**: Ensure outputs are valid JSON, markdown, or plain text
   - **Features**: Required field checking for JSON

2. **`OutputLengthGuardrail`** - Enforce output length constraints
   - **Use case**: Keep responses concise or ensure minimum detail
   - **Features**: Auto-truncation, character or token counting

3. **`FactualityGuardrail`** - Validate factual claims
   - **Use case**: Check responses against known facts
   - **Features**: Reference data checking, custom validation functions

4. **`NoHallucinationGuardrail`** - Detect potential hallucinations
   - **Use case**: Ensure high-confidence, cited responses
   - **Features**: Confidence thresholds, citation requirements

5. **`SensitiveContentFilterGuardrail`** - Filter sensitive information
   - **Use case**: Prevent leaking passwords, API keys, secrets
   - **Features**: Pattern-based blocking, auto-redaction

**When to use:** Add before END node to validate agent outputs before delivery.

---

### `safety_guardrails.py` - Security & Safety

Content moderation, PII protection, and security validation.

**Guardrails:**

1. **`PIIDetectionGuardrail`** - Detect and redact PII
   - **Use case**: Privacy protection, GDPR compliance
   - **Detects**: Emails, phone numbers, SSNs, credit cards, IP addresses
   - **Features**: Auto-redaction or blocking, allowed PII types

2. **`ContentSafetyGuardrail`** - LLM-based content moderation
   - **Use case**: Semantic safety checking for violence, hate, etc.
   - **Features**: Uses Claude 4.5 Haiku for context-aware moderation
   - **Categories**: Violence, hate speech, sexual content, self-harm, illegal

3. **`ToxicityGuardrail`** - Detect toxic/rude content
   - **Use case**: Maintain respectful communication
   - **Features**: Can use Detoxify model or keyword fallback

4. **`PromptInjectionGuardrail`** - Prevent prompt injection attacks
   - **Use case**: Security - prevent instruction override attempts
   - **Features**: Detects common injection patterns

5. **`CodeExecutionGuardrail`** - Validate code safety
   - **Use case**: When agent generates or executes code
   - **Features**: Whitelist imports, block dangerous functions

**When to use:** Apply to both inputs and outputs for comprehensive safety.

---

### `llm_guardrails.py` - Flexible LLM-Based Validation

Use Claude 4.5 Haiku to validate content with custom prompts. More flexible and context-aware than rule-based guardrails.

**Guardrails:**

1. **`LLMGuardrail`** - Base class with fully customizable prompts
   - **Use case**: Any complex, domain-specific validation
   - **Features**:
     - Define validation logic in natural language
     - System prompt + instruction template
     - Custom response parsing
     - Fallback function support
   - **Example**: Brand guidelines, legal compliance, custom policies

2. **`BrandSafetyGuardrail`** - Validate against brand values
   - **Use case**: Ensure content aligns with company values
   - **Features**: Brand values list, prohibited topics
   - **Example**: "Professional, family-friendly, no politics"

3. **`ToneGuardrail`** - Validate tone and style
   - **Use case**: Ensure responses match desired tone
   - **Features**: Desired tone description, disallowed tones
   - **Example**: "Friendly and helpful, not condescending"

4. **`FactualAccuracyGuardrail`** - LLM-based fact checking
   - **Use case**: Verify claims against knowledge base
   - **Features**: Knowledge base support, citation requirements
   - **Example**: Validate medical facts, historical claims

**When to use:**
- Complex, nuanced validation that keyword matching can't handle
- Need to understand context (e.g., "kill the process" vs violence)
- Domain-specific validation (legal, medical, technical)
- Tone, intent, or style checking

**Benefits:**
- Context-aware: Understands nuance
- Maintainable: Update rules by changing prompts, not code
- Powerful: Handles complex validation logic
- Fast: Claude 4.5 Haiku provides sub-2-second validation

---

### `config.py` - Configuration & LLM Setup

Utilities for configuring guardrails and creating LLM instances.

**Functions:**
- `get_anthropic_api_key()` - Load API key from environment
- `get_guardrail_model()` - Get configured model (default: claude-haiku-4-5)
- `create_anthropic_llm()` - Create Claude instance for guardrails
- `create_openai_llm()` - Create OpenAI instance (alternative)
- `print_config_info()` - Debug configuration

**Setup:**
1. Copy `.env.example` to `.env`
2. Add your `ANTHROPIC_API_KEY`
3. Use `create_anthropic_llm()` in your guardrails

---

### `utils.py` - Helper Functions

Common utility functions used across guardrails.

**Functions:**
- `contains_profanity()` - Check for profanity
- `detect_pii_patterns()` - Find PII in text
- `redact_pii()` - Remove PII from text
- `check_topic_match()` - Match topics with fuzzy logic
- `count_tokens_approximate()` - Estimate token count
- `truncate_to_length()` - Truncate text safely
- `extract_json_from_text()` - Parse JSON from LLM responses
- `sanitize_html()` - Remove HTML tags
- `calculate_text_similarity()` - Compare text similarity

---

### `visualization.py` - Graph Visualization

Generate visual representations of your LangGraph workflows.

**Functions:**
- `visualize_graph()` - Generate Mermaid, ASCII, or PNG diagrams
- `print_graph_info()` - Display graph structure
- `create_graph_documentation()` - Generate diagrams for all examples

**Usage:**
```python
from guardrails.visualization import visualize_graph

graph = build_my_graph()
visualize_graph(graph, "workflow.mermaid", format="mermaid")
```

See `generate_diagrams.py` in the root directory for batch generation.

---

## Usage Patterns

### Pattern 1: Single Guardrail
```python
from guardrails.input_guardrails import TopicValidationGuardrail

guardrail = TopicValidationGuardrail(allowed_topics=["weather"])
workflow.add_node("validate", guardrail.check)
workflow.add_conditional_edges("validate", guardrail.route, {
    "continue": "agent",
    "reject": "rejection"
})
```

### Pattern 2: Chained Guardrails
```python
from guardrails.base import GuardrailChain

chain = GuardrailChain([
    InputLengthGuardrail(max_length=1000),
    ProfanityFilterGuardrail(),
    TopicValidationGuardrail(allowed_topics=["tech"]),
])
workflow.add_node("validate", chain.check)
```

### Pattern 3: Custom LLM Guardrail
```python
from guardrails.llm_guardrails import LLMGuardrail
from guardrails.config import create_anthropic_llm

custom = LLMGuardrail(
    llm=create_anthropic_llm(),
    instruction_template="""Check if this follows our policies:
    {content}

    Respond: {{"is_safe": bool, "reason": str}}"""
)
workflow.add_node("custom_check", custom.check)
```

### Pattern 4: Conditional Application
```python
from guardrails.base import ConditionalGuardrail

conditional = ConditionalGuardrail(
    condition=lambda state: state.get("user_type") != "premium",
    guardrail=RateLimitGuardrail(max_requests_per_hour=100)
)
```

## Best Practices

1. **Layer Your Guardrails**: Start with fast, cheap checks before expensive LLM calls
   ```python
   GuardrailChain([
       InputLengthGuardrail(),      # Cheap
       ProfanityFilterGuardrail(),  # Fast regex
       LLMGuardrail(),              # Expensive, but accurate
   ])
   ```

2. **Use Metrics**: Track guardrail performance
   ```python
   metrics = guardrail.get_metrics()
   print(f"Block rate: {metrics['block_rate']:.2%}")
   ```

3. **Fail Gracefully**: LLM guardrails should fail open (allow on error)
   - Include fallback functions
   - Log errors for monitoring
   - Don't block on technical failures

4. **Make Configurable**: Use environment variables for thresholds
   ```python
   threshold = float(os.getenv("SAFETY_THRESHOLD", "0.7"))
   ```

5. **Test Thoroughly**: Create test cases for edge cases
   - Valid inputs that should pass
   - Invalid inputs that should block
   - Borderline cases

## When to Use Which Guardrail

**Use keyword/pattern matching when:**
- Rules are simple and explicit (profanity, PII patterns)
- Speed is critical (< 100ms)
- Want deterministic results
- Cost is a constraint

**Use LLM-based guardrails when:**
- Rules are complex or nuanced
- Need context awareness
- Validating tone, intent, or style
- Domain expertise required

**Layer both for best results:**
```python
GuardrailChain([
    ProfanityFilterGuardrail(),     # Fast keyword check
    PromptInjectionGuardrail(),      # Fast pattern check
    ContentSafetyGuardrail(llm),     # LLM for nuanced cases
])
```

## Examples

See the `examples/` directory for complete working examples:
- `01_simple_input_guardrail.py` - Basic topic validation
- `02_pii_redaction.py` - PII detection and redaction
- `03_output_validation.py` - Output format validation
- `04_multi_layer.py` - Chained guardrails
- `05_llm_based_safety.py` - LLM-powered safety
- `06_production_example.py` - Production setup
- `07_custom_llm_guardrails.py` - Custom LLM validation

## Contributing

To add a new guardrail:
1. Extend `BaseGuardrail`
2. Implement `validate()` method
3. Add to appropriate file (input/output/safety/llm)
4. Export in `__init__.py`
5. Add example usage
6. Update this README

## Support

- Full documentation: See main `README.md`
- Quick start: See `QUICKSTART.md`
- Issues: GitHub Issues
