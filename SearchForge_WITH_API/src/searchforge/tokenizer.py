"""Text normalization and tokenization utilities."""

import unicodedata


def normalize(text: str) -> str:
    """Normalize Unicode, apply case folding, and replace punctuation with spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = unicodedata.normalize("NFC", text)

    cleaned = "".join(
        char if char.isalnum() else " "
        for char in text
    )
    return " ".join(cleaned.split())


def tokenize(text: str) -> list[str]:
    """Return normalized word tokens in their original order."""
    normalized = normalize(text)
    return normalized.split() if normalized else []
