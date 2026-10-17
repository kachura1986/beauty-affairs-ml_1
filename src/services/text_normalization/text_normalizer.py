import re
import unicodedata


class TextNormalizer:
    """Utility class for text cleaning and normalization."""

    # Precompiled regex patterns for performance
    _RE_QUOTES = re.compile(r"[\'\"]")
    _RE_WHITESPACE = re.compile(r"\s+")

    def normalize(self, text: str | None) -> str:
        """
        Normalize text by applying several preprocessing steps:

        Steps:
        - Unicode normalization (NFKC)
        - Removing single and double quotes
        - Replacing underscores with spaces
        - Collapsing multiple spaces into one
        - Lowercasing and stripping whitespace

        Args:
            text (str | None): Input text.

        Returns:
            str: Normalized text. Returns an empty string if input is None or becomes empty after cleaning.
        """
        if not text:
            return ""

        # Unicode normalization
        normalized_text = unicodedata.normalize("NFKC", text)
        # Remove quotes
        normalized_text = self._RE_QUOTES.sub("", normalized_text)
        # Replace underscores with spaces
        normalized_text = normalized_text.replace("_", " ")
        # Collapse multiple spaces
        normalized_text = self._RE_WHITESPACE.sub(" ", normalized_text)
        # Convert to lowercase and strip spaces
        normalized_text = normalized_text.lower().strip()

        return normalized_text
