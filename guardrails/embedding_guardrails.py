"""
Embedding-based guardrails using semantic similarity.

These guardrails use embedding models to calculate semantic similarity between
user inputs and expected topics. They provide a fast, cost-effective middle ground
between keyword matching (fast but unreliable) and LLM validation (accurate but slower).

Benefits:
- Fast: ~10-50ms per validation (vs 1-2s for LLM)
- Cost-effective: One-time embedding generation, then just math
- Semantic understanding: "forecast" and "weather" have high similarity
- Cacheable: Pre-compute topic embeddings once
- Deterministic: Same input = same similarity score

Limitations:
- Doesn't understand negation ("not about weather" still similar to "weather")
- Can't reason about complex edge cases
- Need to tune threshold per use case
- Less accurate than LLM (85-95% vs 95-99%)

Supported embedding providers:
- Anthropic (Voyage embeddings via Anthropic API)
- OpenAI (text-embedding-3-small)
- SentenceTransformers (local, no API required)
"""

from typing import List, Optional, Callable, Dict
import numpy as np
from guardrails.base import BaseGuardrail, GuardrailResult, GuardrailAction


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """
    Anthropic embedding provider (uses Voyage embeddings).

    Note: As of the implementation date, Anthropic uses Voyage embeddings.
    Check latest docs for current embedding model.
    """

    def __init__(self, api_key: str = None, model: str = "voyage-3"):
        """
        Initialize Anthropic embedding provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                from guardrails.config import get_anthropic_api_key

                api_key = self.api_key or get_anthropic_api_key()
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")

                self._client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install: pip install anthropic")

        return self._client

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using Anthropic API."""
        # Note: This is a placeholder - update with actual Anthropic embedding API when available
        raise NotImplementedError(
            "Anthropic embeddings API is not yet available. "
            "Use OpenAIEmbeddingProvider or SentenceTransformerProvider instead."
        )


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small or compatible APIs."""

    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small", base_url: str = None):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var). For LM Studio, can be any string.
            model: Embedding model (text-embedding-3-small or text-embedding-3-large)
            base_url: Custom base URL for OpenAI-compatible APIs (e.g., LM Studio: http://localhost:1234/v1)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                import os

                api_key = self.api_key or os.getenv("OPENAI_API_KEY")

                # For custom endpoints (like LM Studio), API key may not be required
                if not api_key and not self.base_url:
                    raise ValueError("OPENAI_API_KEY not found in environment")

                # Use default "not-needed" for local endpoints if no key provided
                if not api_key and self.base_url:
                    api_key = "not-needed"

                # Initialize with custom base_url if provided (for LM Studio, etc.)
                if self.base_url:
                    self._client = OpenAI(api_key=api_key, base_url=self.base_url)
                else:
                    self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")

        return self._client

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        client = self._get_client()

        response = client.embeddings.create(
            model=self.model,
            input=text
        )

        return np.array(response.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts (more efficient than one-by-one)."""
        client = self._get_client()

        response = client.embeddings.create(
            model=self.model,
            input=texts
        )

        return [np.array(item.embedding) for item in response.data]


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    No API key required - runs locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence-transformers provider.

        Args:
            model_name: Hugging Face model name
                - all-MiniLM-L6-v2: Fast, good for most use cases (384 dims)
                - all-mpnet-base-v2: Higher quality (768 dims)
                - paraphrase-multilingual: Supports 50+ languages
        """
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy load sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install: pip install sentence-transformers"
                )

        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using local model."""
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return list(embeddings)


class SemanticSimilarityGuardrail(BaseGuardrail):
    """
    Validates input relevance using embedding-based semantic similarity.

    Fast, cost-effective middle ground between keyword matching and LLM validation.

    Example:
        # Using OpenAI embeddings
        from guardrails.embedding_guardrails import (
            SemanticSimilarityGuardrail,
            OpenAIEmbeddingProvider
        )

        guardrail = SemanticSimilarityGuardrail(
            topic_descriptions=[
                "weather forecasts and climate information",
                "temperature, precipitation, and atmospheric conditions"
            ],
            embedding_provider=OpenAIEmbeddingProvider(),
            similarity_threshold=0.75
        )

        # Using local embeddings (no API required)
        from guardrails.embedding_guardrails import SentenceTransformerProvider

        guardrail = SemanticSimilarityGuardrail(
            topic_descriptions=["programming and software development"],
            embedding_provider=SentenceTransformerProvider(),
            similarity_threshold=0.70
        )
    """

    def __init__(
        self,
        topic_descriptions: List[str],
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.75,
        aggregation: str = "max",  # "max" or "mean"
        cache_topic_embeddings: bool = True,
        name: str = None,
    ):
        """
        Initialize semantic similarity guardrail.

        Args:
            topic_descriptions: List of topic descriptions to match against
                Example: ["weather and climate", "forecasting and predictions"]
            embedding_provider: Provider for generating embeddings
            similarity_threshold: Minimum similarity score to allow (0.0-1.0)
                Typical values: 0.7-0.8
            aggregation: How to combine multiple topic similarities
                - "max": Use highest similarity score
                - "mean": Use average similarity score
            cache_topic_embeddings: Whether to pre-compute topic embeddings
            name: Guardrail name
        """
        super().__init__(name=name or "SemanticSimilarity")

        self.topic_descriptions = topic_descriptions
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.aggregation = aggregation
        self.cache_topic_embeddings = cache_topic_embeddings

        # Pre-compute topic embeddings if caching enabled
        self._topic_embeddings: Optional[List[np.ndarray]] = None
        if cache_topic_embeddings:
            self._compute_topic_embeddings()

    def _compute_topic_embeddings(self):
        """Pre-compute embeddings for all topic descriptions."""
        print(f"Computing embeddings for {len(self.topic_descriptions)} topics...")
        self._topic_embeddings = self.embedding_provider.embed_batch(
            self.topic_descriptions
        )
        print("Topic embeddings computed and cached.")

    def _get_topic_embeddings(self) -> List[np.ndarray]:
        """Get topic embeddings (cached or compute on-demand)."""
        if self._topic_embeddings is None:
            self._compute_topic_embeddings()
        return self._topic_embeddings

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate content against topic embeddings."""
        try:
            # Get embeddings
            input_embedding = self.embedding_provider.embed(content)
            topic_embeddings = self._get_topic_embeddings()

            # Calculate similarities
            similarities = [
                cosine_similarity(input_embedding, topic_emb)
                for topic_emb in topic_embeddings
            ]

            # Aggregate similarities
            if self.aggregation == "max":
                final_similarity = max(similarities)
                matched_topic_idx = similarities.index(final_similarity)
            elif self.aggregation == "mean":
                final_similarity = np.mean(similarities)
                matched_topic_idx = 0  # Not really applicable for mean
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")

            # Decide action
            if final_similarity >= self.similarity_threshold:
                return GuardrailResult(
                    action=GuardrailAction.ALLOW,
                    reason=f"Topic match (similarity: {final_similarity:.3f})",
                    metadata={
                        "similarity_score": final_similarity,
                        "all_similarities": [float(s) for s in similarities],
                        "matched_topic": self.topic_descriptions[matched_topic_idx] if self.aggregation == "max" else None,
                        "threshold": self.similarity_threshold,
                    },
                )
            else:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Off-topic (similarity: {final_similarity:.3f} < {self.similarity_threshold})",
                    metadata={
                        "similarity_score": final_similarity,
                        "all_similarities": [float(s) for s in similarities],
                        "threshold": self.similarity_threshold,
                    },
                )

        except Exception as e:
            # Fail open on errors (don't block due to technical issues)
            return GuardrailResult(
                action=GuardrailAction.WARN,
                reason=f"Semantic similarity check failed: {str(e)}",
                metadata={"error": str(e)},
            )


class MultiModalSemanticGuardrail(BaseGuardrail):
    """
    Validates content using both allowed and blocked topic embeddings.

    Allows content that is:
    - Semantically similar to allowed topics
    - NOT similar to blocked topics

    Example:
        guardrail = MultiModalSemanticGuardrail(
            allowed_topics=["technology", "programming"],
            blocked_topics=["politics", "religion", "financial advice"],
            embedding_provider=SentenceTransformerProvider(),
            allow_threshold=0.70,
            block_threshold=0.75
        )
    """

    def __init__(
        self,
        allowed_topics: List[str],
        blocked_topics: List[str] = None,
        embedding_provider: EmbeddingProvider = None,
        allow_threshold: float = 0.70,
        block_threshold: float = 0.75,
        name: str = None,
    ):
        """
        Initialize multi-modal semantic guardrail.

        Args:
            allowed_topics: Topics to allow
            blocked_topics: Topics to explicitly block
            embedding_provider: Embedding provider (defaults to local model)
            allow_threshold: Minimum similarity to allowed topics
            block_threshold: Maximum similarity to blocked topics (blocks if exceeded)
            name: Guardrail name
        """
        super().__init__(name=name or "MultiModalSemantic")

        self.allowed_topics = allowed_topics
        self.blocked_topics = blocked_topics or []

        # Default to local embeddings if not specified
        if embedding_provider is None:
            embedding_provider = SentenceTransformerProvider()

        # Create sub-guardrails
        self.allow_guardrail = SemanticSimilarityGuardrail(
            topic_descriptions=allowed_topics,
            embedding_provider=embedding_provider,
            similarity_threshold=allow_threshold,
            name=f"{self.name}_Allow"
        )

        if self.blocked_topics:
            self.block_guardrail = SemanticSimilarityGuardrail(
                topic_descriptions=blocked_topics,
                embedding_provider=embedding_provider,
                similarity_threshold=block_threshold,
                name=f"{self.name}_Block"
            )
        else:
            self.block_guardrail = None

    def validate(self, content: str, metadata: dict = None) -> GuardrailResult:
        """Validate against both allowed and blocked topics."""
        # Check if blocked
        if self.block_guardrail:
            block_result = self.block_guardrail.validate(content, metadata)
            if block_result.action == GuardrailAction.ALLOW:
                # High similarity to blocked topic = block the content
                similarity = block_result.metadata.get("similarity_score", 0)
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Content matches blocked topic (similarity: {similarity:.3f})",
                    metadata=block_result.metadata,
                )

        # Check if allowed
        allow_result = self.allow_guardrail.validate(content, metadata)
        return allow_result
