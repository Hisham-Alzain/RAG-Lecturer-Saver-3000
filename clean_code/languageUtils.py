import re
from typing import List

class LanguageUtils:
    """Language detection and handling utilities."""

    ARABIC_RANGE = ("\u0600", "\u06ff")

    ARABIC_STOP_WORDS = {
        "من",
        "في",
        "على",
        "إلى",
        "عن",
        "مع",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "التي",
        "الذي",
        "هو",
        "هي",
        "هم",
        "أن",
        "إن",
        "كان",
        "كانت",
        "ما",
        "لا",
        "لم",
        "لن",
        "قد",
        "كل",
        "بعض",
        "أي",
        "و",
        "أو",
        "ثم",
        "حتى",
        "إذا",
        "لو",
        "كما",
        "بل",
        "لكن",
        "غير",
        "بين",
        "عند",
        "منذ",
        "حول",
        "خلال",
        "بعد",
        "قبل",
        "فوق",
        "تحت",
        "ال",
        "الى",
        "هل",
        "كيف",
        "لماذا",
        "متى",
        "أين",
    }

    ENGLISH_STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "from",
        "about",
        "into",
        "through",
        "can",
        "could",
        "would",
        "should",
        "will",
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        if not text:
            return "en"

        arabic_chars = sum(
            1 for c in text if cls.ARABIC_RANGE[0] <= c <= cls.ARABIC_RANGE[1]
        )
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return "en"

        return "ar" if (arabic_chars / total_alpha) > 0.3 else "en"

    @classmethod
    def get_stop_words(cls, lang: str) -> set:
        """Get stop words for the specified language."""
        return cls.ARABIC_STOP_WORDS if lang == "ar" else cls.ENGLISH_STOP_WORDS

    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        """Extract keywords from text, removing stop words."""
        lang = cls.detect_language(text)
        stop_words = cls.get_stop_words(lang)

        # Tokenize
        if lang == "ar":
            words = re.findall(r"[\u0600-\u06ff]+|\b\w+\b", text.lower())
        else:
            words = re.findall(r"\b\w+\b", text.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Return unique keywords preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:10]  # Max 10 keywords
