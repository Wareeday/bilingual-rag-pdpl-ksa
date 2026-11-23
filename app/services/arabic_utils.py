"""Arabic & English text normalization, tokenization, and language detection.

Improves Arabic BM25 lexical recall and ensures Arabic-Indic numerals are
normalized before PII regex scanning.
"""

from __future__ import annotations

import re
from typing import Literal

# Arabic diacritics (Tashkeel) and Tatweel
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
_TATWEEL_RE = re.compile(r"\u0640+")
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[A-Za-z0-9]+")

# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits to ASCII
_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

# Minimal bilingual stopwords for cleaner BM25 matching
_STOPWORDS = {
    # English
    "the", "is", "are", "was", "were", "a", "an", "of", "to", "in", "on",
    "for", "with", "and", "or", "by", "at", "from", "as", "it", "this",
    "that", "be", "can", "will", "what", "how", "when", "where", "who",
    # Arabic (normalized forms)
    "في", "من", "علي", "الي", "عن", "مع", "هذا", "هذه", "التي", "الذي",
    "ان", "او", "ما", "هل", "هو", "هي", "كان", "كانت", "بين", "بعد", "قبل",
}


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic digits (٠١٢٣...) into ASCII digits (0123...)."""
    return text.translate(_DIGIT_TRANSLATION)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic orthographic variants for high-recall lexical search."""
    text = normalize_digits(text)
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    # Normalize Alef variants -> bare Alef
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize Alef Maksura -> Yeh
    text = text.replace("ى", "ي")
    # Normalize Teh Marbuta -> Heh
    text = text.replace("ة", "ه")
    # Normalize Hamza on Waw/Yeh
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    return text


def detect_language(text: str) -> Literal["ar", "en", "mixed"]:
    """Detect whether text is primarily Arabic, English, or bilingual mixed."""
    arabic_count = len(_ARABIC_CHAR_RE.findall(text))
    latin_count = len(_LATIN_CHAR_RE.findall(text))
    total = arabic_count + latin_count
    if total == 0:
        return "en"
    ar_ratio = arabic_count / total
    if ar_ratio >= 0.65:
        return "ar"
    if ar_ratio <= 0.20:
        return "en"
    return "mixed"


def tokenize_bilingual(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize Arabic + English text into normalized search tokens."""
    normalized = normalize_arabic(text).lower()
    tokens = _TOKEN_RE.findall(normalized)
    cleaned: list[str] = []
    for tok in tokens:
        # Strip Arabic definite article "ال" for better lexical matching on 4+ char words
        if len(tok) > 3 and tok.startswith("ال"):
            tok = tok[2:]
        if remove_stopwords and tok in _STOPWORDS:
            continue
        if len(tok) >= 2 or tok.isdigit():
            cleaned.append(tok)
    return cleaned
