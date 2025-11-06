"""
Configuration management for guardrails.

Handles loading environment variables from .env file and providing
configuration for LLM-based guardrails.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def get_anthropic_api_key() -> Optional[str]:
    """
    Get Anthropic API key from environment.

    Returns:
        API key if set, None otherwise
    """
    return os.getenv("ANTHROPIC_API_KEY")


def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from environment.

    Returns:
        API key if set, None otherwise
    """
    return os.getenv("OPENAI_API_KEY")


def get_guardrail_model() -> str:
    """
    Get the model to use for LLM-based guardrails.

    Defaults to Claude 4.5 Haiku (fastest, most cost-efficient for guardrails).

    Returns:
        Model name string
    """
    return os.getenv("GUARDRAIL_MODEL", "claude-haiku-4-5")


def get_safety_level() -> str:
    """
    Get the safety level configuration.

    Returns:
        Safety level: "strict", "moderate", or "permissive"
    """
    return os.getenv("GUARDRAIL_SAFETY_LEVEL", "moderate")


def create_anthropic_llm(model: str = None, temperature: float = 0, **kwargs):
    """
    Create an Anthropic LLM instance for use in guardrails.

    Args:
        model: Model name (defaults to Claude 4.5 Haiku)
        temperature: Temperature setting (default 0 for deterministic guardrails)
        **kwargs: Additional arguments to pass to ChatAnthropic

    Returns:
        ChatAnthropic instance

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic is required for LLM-based guardrails. "
            "Install it with: pip install langchain-anthropic"
        )

    api_key = get_anthropic_api_key()
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    model = model or get_guardrail_model()

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=api_key,
        **kwargs
    )


def create_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0, **kwargs):
    """
    Create an OpenAI LLM instance for use in guardrails.

    Args:
        model: Model name (default gpt-4o-mini)
        temperature: Temperature setting (default 0 for deterministic guardrails)
        **kwargs: Additional arguments to pass to ChatOpenAI

    Returns:
        ChatOpenAI instance

    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required. "
            "Install it with: pip install langchain-openai"
        )

    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        **kwargs
    )


def create_embedding_provider(provider: str = "local", **kwargs):
    """
    Create an embedding provider for semantic similarity guardrails.

    Args:
        provider: Provider type - "openai", "local", or "anthropic"
        **kwargs: Additional arguments for the provider

    Returns:
        EmbeddingProvider instance

    Examples:
        # Local embeddings (no API key required)
        provider = create_embedding_provider("local")

        # OpenAI embeddings
        provider = create_embedding_provider("openai")

        # Custom model
        provider = create_embedding_provider("local", model_name="all-mpnet-base-v2")
    """
    from guardrails.embedding_guardrails import (
        OpenAIEmbeddingProvider,
        SentenceTransformerProvider,
        AnthropicEmbeddingProvider,
    )

    if provider == "openai":
        api_key = kwargs.get("api_key") or get_openai_api_key()
        model = kwargs.get("model", "text-embedding-3-small")
        return OpenAIEmbeddingProvider(api_key=api_key, model=model)

    elif provider == "anthropic":
        api_key = kwargs.get("api_key") or get_anthropic_api_key()
        model = kwargs.get("model", "voyage-3")
        return AnthropicEmbeddingProvider(api_key=api_key, model=model)

    elif provider == "local":
        model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
        return SentenceTransformerProvider(model_name=model_name)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            "Choose 'openai', 'anthropic', or 'local'"
        )


def get_embedding_model() -> str:
    """
    Get the embedding model to use for semantic similarity.

    Returns:
        Embedding model name
    """
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# Configuration info for users
def print_config_info():
    """Print configuration information for debugging."""
    print("=== Guardrails Configuration ===")
    print(f"Anthropic API Key: {'✓ Set' if get_anthropic_api_key() else '✗ Not set'}")
    print(f"OpenAI API Key: {'✓ Set' if get_openai_api_key() else '✗ Not set'}")
    print(f"Default LLM Model: {get_guardrail_model()}")
    print(f"Default Embedding Model: {get_embedding_model()}")
    print(f"Safety Level: {get_safety_level()}")
    print("================================")


if __name__ == "__main__":
    print_config_info()
