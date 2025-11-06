# Guardrails Module

This directory contains reusable guardrail components for LangGraph workflows. Each guardrail validates or transforms content at different points in your graph.

## Quick Reference

| Guardrail | File | Type | Use Case |
|-----------|------|------|----------|
| `BaseGuardrail` | base.py | Base Class | Extend to create custom guardrails |
| `GuardrailChain` | base.py | Combiner | Chain multiple guardrails together |
| `ConditionalGuardrail` | base.py | Wrapper | Apply guardrails conditionally |
| `InputLengthGuardrail` | input_guardrails.py | Input | Enforce length limits |
| `RateLimitGuardrail` | input_guardrails.py | Input | Rate limit by user |
| `OutputFormatGuardrail` | output_guardrails.py | Output | Validate response format |
| `OutputLengthGuardrail` | output_guardrails.py | Output | Enforce output length |
| `PIIDetectionGuardrail` | safety_guardrails.py | Safety | Detect/redact PII |
| `ContentSafetyGuardrail` | safety_guardrails.py | Safety | Content moderation (LLM) |
| `PromptInjectionGuardrail` | safety_guardrails.py | Safety | Prevent prompt injection |
| `CodeExecutionGuardrail` | safety_guardrails.py | Safety | Code safety validation |
| `SemanticSimilarityGuardrail` | embedding_guardrails.py | Embedding | Fast topic validation via embeddings |
| `MultiModalSemanticGuardrail` | embedding_guardrails.py | Embedding | Allow/block topics with embeddings |
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

1. **`InputLengthGuardrail`** - Enforce min/max input length
   - **Use case**: Prevent too-short or too-long inputs
   - **Features**: Character or token counting, auto-truncation option

2. **`RateLimitGuardrail`** - Limit requests per user/session
   - **Use case**: Prevent abuse, manage costs
   - **Features**: Per-minute and per-hour limits, user-based tracking

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

**When to use:** Add before END node to validate agent outputs before delivery.

**Note:** For factual validation or hallucination detection, use LLM-based guardrails with custom prompts rather than simple pattern matching.

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

3. **`PromptInjectionGuardrail`** - Prevent prompt injection attacks
   - **Use case**: Security - prevent instruction override attempts
   - **Features**: Detects common injection patterns

4. **`CodeExecutionGuardrail`** - Validate code safety
   - **Use case**: When agent generates or executes code
   - **Features**: Whitelist imports, block dangerous functions

**When to use:** Apply to both inputs and outputs for comprehensive safety.

---

### `embedding_guardrails.py` - Fast Semantic Similarity Validation

Use embedding models to validate topic relevance via semantic similarity. Fast, cost-effective middle ground between keyword matching (unreliable) and LLM validation (slower/costlier).

**Guardrails:**

1. **`SemanticSimilarityGuardrail`** - Topic validation using embeddings
   - **Use case**: High-volume topic filtering with semantic understanding
   - **Features**:
     - Pre-compute and cache topic embeddings
     - Cosine similarity matching
     - Multiple embedding provider support
     - Configurable similarity thresholds
     - Deterministic results
   - **Example**: Weather bot that understands "umbrella" relates to weather
   - **Performance**: 10-50ms, 85-95% accuracy

2. **`MultiModalSemanticGuardrail`** - Allow + block topic lists
   - **Use case**: Topics with both allowed and explicitly blocked categories
   - **Features**: Separate thresholds for allow/block
   - **Example**: Tech support bot (allow: tech questions, block: politics, finance)

**Embedding Providers:**

- **`SentenceTransformerProvider`** - Local embeddings (FREE, no API required)
  - Model: all-MiniLM-L6-v2 (default, fast, 384 dims)
  - Model: all-mpnet-base-v2 (higher quality, 768 dims)
  - Model: paraphrase-multilingual (50+ languages)
  - **Best for**: Python-only projects, quick prototyping

- **LM Studio Provider** - Local embeddings with GUI (FREE, RECOMMENDED)
  - **Download**: [LM Studio](https://lmstudio.ai/) - user-friendly interface for running embedding models locally
  - **Benefits**:
    - Easy model selection and management via GUI
    - Run various embedding models (BERT, MiniLM, E5, etc.)
    - OpenAI-compatible API endpoint
    - Switch models without code changes
    - Monitor performance in real-time
  - **Setup**: Load an embedding model in LM Studio, start the server, use provider type `"lm-studio"`
  - **Default endpoint**: `http://localhost:1234/v1`

- **`OpenAIEmbeddingProvider`** - OpenAI embeddings or compatible APIs
  - Model: text-embedding-3-small (~$0.00002/request)
  - Model: text-embedding-3-large (higher accuracy)
  - **Also supports**: Custom OpenAI-compatible endpoints (including LM Studio)

- **`AnthropicEmbeddingProvider`** - Anthropic/Voyage embeddings
  - Note: Placeholder for future Anthropic embedding API

**When to use:**
- High volume (1000+ requests/min)
- Clear, well-defined topics
- 85-95% accuracy acceptable
- Cost is a constraint
- Need <50ms response time
- Can run locally without API (SentenceTransformer)

**Benefits:**
- Fast: 10-100x faster than LLM
- Cost-effective: Free (local) or very cheap (OpenAI)
- Semantic: Understands "umbrella" → weather
- Cacheable: Pre-compute topics once
- Offline-capable: Works without API (local mode)

**Performance Comparison:**
| Approach | Speed | Accuracy | Cost | Context Understanding |
|----------|-------|----------|------|----------------------|
| Keyword | <1ms | 60-70% | Free | None (removed) |
| **Embedding** | 10-50ms | 85-95% | Very Low | Semantic similarity |
| LLM | 1-2s | 95-99% | Low-Med | Full context + reasoning |

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
   - **Example**: Brand guidelines, legal compliance, custom policies, topic validation

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
- Topic validation with semantic understanding

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
- `get_openai_api_key()` - Load OpenAI API key from environment
- `get_guardrail_model()` - Get configured model (default: claude-haiku-4-5)
- `get_embedding_model()` - Get configured embedding model
- `create_anthropic_llm()` - Create Claude instance for guardrails
- `create_openai_llm()` - Create OpenAI instance (alternative)
- `create_embedding_provider()` - Create embedding provider (local/openai/anthropic)
- `print_config_info()` - Debug configuration

**Setup:**
1. Copy `.env.example` to `.env`
2. Add your `ANTHROPIC_API_KEY` (for LLM guardrails)
3. Optionally add `OPENAI_API_KEY` (for OpenAI embeddings/LLM)
4. Use helper functions to create instances

---

### `utils.py` - Helper Functions

Common utility functions used across guardrails.

**Functions:**
- `detect_pii_patterns()` - Find PII in text
- `redact_pii()` - Remove PII from text
- `count_tokens_approximate()` - Estimate token count
- `truncate_to_length()` - Truncate text safely
- `extract_json_from_text()` - Parse JSON from LLM responses
- `sanitize_html()` - Remove HTML tags

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
from guardrails.input_guardrails import InputLengthGuardrail

guardrail = InputLengthGuardrail(min_length=3, max_length=1000)
workflow.add_node("validate", guardrail.check)
workflow.add_conditional_edges("validate", guardrail.route, {
    "allow": "agent",
    "block": "rejection"
})
```

### Pattern 2: Chained Guardrails
```python
from guardrails.base import GuardrailChain
from guardrails.input_guardrails import InputLengthGuardrail, RateLimitGuardrail
from guardrails.safety_guardrails import PIIDetectionGuardrail

chain = GuardrailChain([
    InputLengthGuardrail(max_length=1000),
    RateLimitGuardrail(max_requests_per_hour=100),
    PIIDetectionGuardrail(redact=True),
])
workflow.add_node("validate", chain.check)
```

### Pattern 3: Embedding-Based Topic Validation
```python
from guardrails.embedding_guardrails import SemanticSimilarityGuardrail
from guardrails.config import create_embedding_provider

# Option A: LM Studio (RECOMMENDED - local, GUI, flexible)
# 1. Download LM Studio from https://lmstudio.ai/
# 2. Load an embedding model (e.g., nomic-embed-text, all-MiniLM-L6-v2)
# 3. Start the local server (default: localhost:1234)
topic_guardrail = SemanticSimilarityGuardrail(
    topic_descriptions=[
        "programming and software development",
        "technology and computer science"
    ],
    embedding_provider=create_embedding_provider(
        "lm-studio",
        base_url="http://localhost:1234/v1",  # LM Studio default
        model="your-model-name"                # Model loaded in LM Studio
    ),
    similarity_threshold=0.70
)
workflow.add_node("topic_check", topic_guardrail.check)

# Option B: SentenceTransformer (local, Python-only, no GUI)
topic_guardrail = SemanticSimilarityGuardrail(
    topic_descriptions=["weather forecasts and climate"],
    embedding_provider=create_embedding_provider("local"),
    similarity_threshold=0.70
)

# Option C: OpenAI embeddings (cloud, paid, high quality)
topic_guardrail = SemanticSimilarityGuardrail(
    topic_descriptions=["weather forecasts and climate"],
    embedding_provider=create_embedding_provider("openai"),
    similarity_threshold=0.75
)
```

### Pattern 4: Custom LLM Guardrail
```python
from guardrails.llm_guardrails import LLMGuardrail
from guardrails.config import create_anthropic_llm

# Topic validation with full LLM reasoning (slower but most accurate)
topic_guardrail = LLMGuardrail(
    llm=create_anthropic_llm(),
    system_prompt="You validate if queries are about technology.",
    instruction_template="""Is this query about technology?

Query: {content}

Respond: {{"is_safe": bool, "reason": str, "confidence": float}}"""
)
workflow.add_node("topic_check", topic_guardrail.check)
```

### Pattern 5: Conditional Application
```python
from guardrails.base import ConditionalGuardrail
from guardrails.input_guardrails import RateLimitGuardrail

conditional = ConditionalGuardrail(
    condition=lambda state: state.get("user_type") != "premium",
    guardrail=RateLimitGuardrail(max_requests_per_hour=100)
)
```

## Best Practices

1. **Layer Your Guardrails**: Start with fast, cheap checks before expensive operations
   ```python
   GuardrailChain([
       InputLengthGuardrail(),              # <1ms, free
       PromptInjectionGuardrail(),          # ~5ms, pattern matching
       SemanticSimilarityGuardrail(...),   # ~30ms, embedding similarity
       ContentSafetyGuardrail(llm),         # ~1s, LLM-based (most accurate)
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

**Use simple validation (pattern/length checks) when:**
- Rules are simple and explicit (length limits, PII patterns)
- Speed is critical (< 10ms)
- Want deterministic results
- Cost is a constraint

**Use embedding-based guardrails when:**
- High volume (1000+ requests/min)
- Clear, well-defined topics
- 85-95% accuracy acceptable
- Need semantic understanding (better than keywords)
- Cost-sensitive (free with local models)
- Need <50ms response time
- Can run offline/locally

**Use LLM-based guardrails when:**
- Rules are complex or nuanced
- Need full context awareness and reasoning
- Validating tone, intent, or style
- Domain expertise required
- 95-99% accuracy required
- Edge cases and tricky phrasing common
- Can tolerate 1-2s response time

**Layer all three for optimal performance:**
```python
GuardrailChain([
    InputLengthGuardrail(),              # <1ms: Fast length check
    PromptInjectionGuardrail(),          # ~5ms: Pattern-based security
    SemanticSimilarityGuardrail(...),   # ~30ms: Embedding topic check
    ContentSafetyGuardrail(llm),         # ~1s: LLM for nuanced validation
])
```

**Decision tree:**
1. Start with simple checks (length, patterns)
2. For topic validation: Use embeddings (fast, semantic)
3. For complex validation: Use LLM (accurate, context-aware)
4. For borderline embedding scores (0.6-0.8): Fall back to LLM

## Examples

See the `examples/` directory for complete working examples:
- `01_pii_redaction.py` - PII detection and redaction
- `02_output_validation.py` - Output format validation
- `03_llm_based_safety.py` - LLM-powered safety
- `04_production_example.py` - Production setup with multi-layer validation
- `05_custom_llm_guardrails.py` - Custom LLM validation
- `06_semantic_topic_validation.py` - Why LLM-based topic validation is better
- `07_embedding_vs_llm_comparison.py` - Embedding vs LLM performance comparison
- `08_lm_studio_embeddings.py` - **Using LM Studio for local embeddings with GUI**

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
- Issues: GitHub Issues
