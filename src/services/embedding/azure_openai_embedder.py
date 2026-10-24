from openai import AsyncAzureOpenAI

from .base_embedder import BaseEmbedder
from ..text_normalization.text_normalizer import TextNormalizer


class AzureOpenAIEmbedder(BaseEmbedder):
    """Generate embeddings using the Azure OpenAI API."""

    def __init__(
            self,
            client: AsyncAzureOpenAI,
            model_name: str,
            dimensions: int,
            normalizer: TextNormalizer,
    ) -> None:
        """
        Initialize the Azure OpenAI embedder.

        Args:
            client (AsyncAzureOpenAI): Azure OpenAI asynchronous client.
            model_name (str): Embedding model or Azure deployment name.
            dimensions (int): Number of dimensions in the embedding vectors.
            normalizer (TextNormalizer): Normalizer used to preprocess input text.
        """
        super().__init__(normalizer)

        self._client = client
        self._model_name = model_name
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text input.

        Args:
            text (str): Input text to embed.

        Returns:
            list[float]: The generated embedding vector.
        """
        # If the text is empty after normalization, use a placeholder
        normalized_text = text if text else 'EMPTY_TEXT'

        normalized_text = self._prepare_text(normalized_text)

        response = await self._client.embeddings.create(
            model=self._model_name,
            input=normalized_text,
            dimensions=self._dimensions,
        )

        return response.data[0].embedding

    async def batch_embed(self, texts: list[str], chunk_size: int = 16) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts (list[str]): Input texts to embed.
            chunk_size (int): Number of texts sent in one API request.

        Returns:
            list[list[float]]: Embedding vectors in the same order as the input texts.

        Raises:
            ValueError: If chunk_size is not positive or an input is empty.
        """
        self._validate_chunk_size(chunk_size)

        # If any text is empty after normalization, use a placeholder
        normalized_texts = [text if text else 'EMPTY_TEXT' for text in texts]

        normalized_texts = [self._prepare_text(normalized_text) for normalized_text in normalized_texts]

        embeddings = []

        for start in range(0, len(normalized_texts), chunk_size):
            chunk = normalized_texts[start: start + chunk_size]

            response = await self._client.embeddings.create(
                model=self._model_name,
                input=chunk,
                dimensions=self._dimensions,
            )

            # Preserve the original input order using the API response indices.
            chunk_embeddings = sorted(response.data, key=lambda item: item.index)

            embeddings.extend(item.embedding for item in chunk_embeddings)

        return embeddings
