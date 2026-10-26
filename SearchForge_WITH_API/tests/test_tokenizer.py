"""Tests for text normalization and tokenization."""

from searchforge.tokenizer import normalize, tokenize


# --- Normal input ---


def test_normalize_simple_sentence():
    assert normalize("Hello World") == "hello world"


def test_tokenize_simple_sentence():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_preserves_order_and_duplicates():
    assert tokenize("the cat and the hat") == ["the", "cat", "and", "the", "hat"]


# --- Punctuation ---


def test_normalize_strips_punctuation():
    assert normalize("Hello, world!") == "hello world"


def test_tokenize_punctuation():
    assert tokenize("Search-Forge: cool?") == ["search", "forge", "cool"]


def test_tokenize_mixed_punctuation_and_spaces():
    assert tokenize("  Wow...  really?!  ") == ["wow", "really"]


# --- Empty / blank strings ---


def test_normalize_empty_string():
    assert normalize("") == ""


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_tokenize_whitespace_only():
    assert tokenize("   \t\n  ") == []


def test_tokenize_punctuation_only():
    assert tokenize("...!!!???") == []


# --- Unicode ---


def test_normalize_unicode_casefold():
    # German ß casefolds to "ss"
    assert normalize("Straße") == "strasse"


def test_normalize_composed_and_decomposed_cafe():
    composed = "café"  # é as a single character
    decomposed = "cafe\u0301"  # e + combining acute accent
    assert normalize(composed) == normalize(decomposed) == "café"


def test_tokenize_unicode_words():
    assert tokenize("Café résumé") == ["café", "résumé"]


def test_tokenize_cjk_characters():
    # CJK ideographs are word characters; no spaces → one token
    assert tokenize("你好世界") == ["你好世界"]


def test_tokenize_fullwidth_punctuation():
    # Fullwidth comma/exclamation become spaces after NFKC + cleaning
    assert tokenize("你好，世界！") == ["你好", "世界"]
