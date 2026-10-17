from typing import Protocol


class Embedder(Protocol):
    """Define the application-level interface for asynchronous embedding models."""

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text input asynchronously.

        Args:
            text (str): Input text to embed.

        Returns:
            list[float]: Generated embedding.
        """
        ...

    async def batch_embed(self, texts: list[str], chunk_size: int) -> list[list[float]]:
        """Generate embeddings for a list of texts asynchronously in batches.

        Args:
            texts (list[str]): List of input texts.
            chunk_size (int): Number of texts processed concurrently in a batch.

        Returns:
            list[list[float]]: List of embeddings.
        """
        ...
