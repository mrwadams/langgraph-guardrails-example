"""
Utility functions for guardrails implementation.
"""

import re
from typing import List, Set


def detect_pii_patterns(text: str) -> dict:
    """
    Detect common PII patterns in text.

    Returns dict with detected PII types and their positions.

    Note: This is a simple pattern-based implementation.
    For production, use a dedicated PII detection library like:
    - presidio-analyzer
    - scrubadub
    - Microsoft Presidio
    """
    pii_found = {}

    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        pii_found["email"] = emails

    # Phone number pattern (US)
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
    phones = re.findall(phone_pattern, text)
    if phones:
        pii_found["phone"] = [f"{p[0]}-{p[1]}-{p[2]}" for p in phones]

    # SSN pattern (US)
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    ssns = re.findall(ssn_pattern, text)
    if ssns:
        pii_found["ssn"] = ssns

    # Credit card pattern (simple)
    cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    cards = re.findall(cc_pattern, text)
    if cards:
        pii_found["credit_card"] = cards

    # IP Address
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ips = re.findall(ip_pattern, text)
    if ips:
        pii_found["ip_address"] = ips

    return pii_found


def redact_pii(text: str, redaction_char: str = "*") -> tuple[str, dict]:
    """
    Redact PII from text.

    Args:
        text: Text to redact
        redaction_char: Character to use for redaction

    Returns:
        Tuple of (redacted_text, pii_found_dict)
    """
    pii_found = detect_pii_patterns(text)
    redacted = text

    # Redact emails
    if "email" in pii_found:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        redacted = re.sub(email_pattern, "[EMAIL_REDACTED]", redacted)

    # Redact phone numbers
    if "phone" in pii_found:
        phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
        redacted = re.sub(phone_pattern, "[PHONE_REDACTED]", redacted)

    # Redact SSNs
    if "ssn" in pii_found:
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        redacted = re.sub(ssn_pattern, "[SSN_REDACTED]", redacted)

    # Redact credit cards
    if "credit_card" in pii_found:
        cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        redacted = re.sub(cc_pattern, "[CREDIT_CARD_REDACTED]", redacted)

    # Redact IP addresses
    if "ip_address" in pii_found:
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        redacted = re.sub(ip_pattern, "[IP_REDACTED]", redacted)

    return redacted, pii_found


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (rough estimate).

    For production, use tiktoken or the model's tokenizer.
    This is a simple heuristic: ~0.75 tokens per word.

    Args:
        text: Text to count tokens for

    Returns:
        Approximate token count
    """
    words = text.split()
    return int(len(words) * 0.75)


def truncate_to_length(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum character length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def extract_json_from_text(text: str) -> dict | None:
    """
    Try to extract JSON object from text.

    Useful for output validation guardrails.

    Args:
        text: Text that might contain JSON

    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    import json

    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        json_str = text[start : end + 1]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def sanitize_html(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Text potentially containing HTML

    Returns:
        Text with HTML tags removed
    """
    import re

    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", "", text)

    # Remove HTML entities
    clean = re.sub(r"&[a-zA-Z]+;", "", clean)

    return clean
