from abc import ABC, abstractmethod

from .embedder import Embedder
from ..text_normalization.text_normalizer import TextNormalizer


class BaseEmbedder(Embedder, ABC):
    """Define the shared implementation for asynchronous embedding providers."""

    def __init__(self, normalizer: TextNormalizer) -> None:
        """
        Initialize the embedder with a text normalizer.

        Args:
            normalizer (TextNormalizer): Normalizer used to preprocess input text.
        """
        self._normalizer = normalizer

    def _validate_text(self, text: str | None) -> None:
        """
        Validate that the input text is not None and not empty.

        Args:
            text (str | None): Input text to validate.

        Raises:
            ValueError: If the input text is None or empty.
        """
        if not text:
            raise ValueError(f"{self.__class__.__name__}: Input text must not be None or empty")

    def _normalize_text(self, text: str | None) -> str:
        """
        Normalize text using the TextNormalizer.

        Args:
            text (str | None): Input text to normalize.

        Returns:
            str: Normalized text.
        """
        return self._normalizer.normalize(text)

    def _prepare_text(self, text: str) -> str:
        """
        Prepare text for embedding by normalizing and validating it.

        Args:
            text (str): Input text to prepare.

        Returns:
            str: Normalized and validated text.
        """
        normalized_text = self._normalize_text(text)
        # If the text is empty after normalization, use a placeholder
        normalized_text = normalized_text if normalized_text else 'EMPTY_TEXT'
        self._validate_text(normalized_text)

        return normalized_text

    def _validate_chunk_size(self, chunk_size: int) -> None:
        """
        Validate that the chunk size is a positive integer.

        Args:
            chunk_size (int): Number of texts sent in one API request.

        Raises:
            TypeError: If the chunk_size is not an integer.
            ValueError: If the chunk_size is not positive.
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text input asynchronously.

        Args:
            text (str): Input text to embed.

        Returns:
            list[float]: Generated embedding.
        """
        pass

    @abstractmethod
    async def batch_embed(self, texts: list[str], chunk_size: int) -> list[list[float]]:
        """
        Generate embeddings for a list of texts asynchronously in batches.

        Args:
            texts (list[str]): List of input texts.
            chunk_size (int): Number of texts processed concurrently in a batch.

        Returns:
            list[list[float]]: List of embeddings.
        """
        pass
