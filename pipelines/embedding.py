import asyncio

from src.dependencies import PipelineDependencies

async def main() -> None:
    dependencies = await PipelineDependencies.create()
    embedder = dependencies.azure_embedder

    # Test 1: embed()
    print("=== Testing embed() ===")
    text = "Artificial intelligence simplifies everyday tasks."
    embedding = await embedder.embed(text)

    assert isinstance(embedding, list), "Embedding must be a list"
    assert len(embedding) == 1024, f"Expected 1024 dimensions, got {len(embedding)}"
    assert all(isinstance(value, (int, float)) for value in embedding)

    print(f"Input: {text}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print("embed() passed!\n")

    # Test 2: batch_embed()
    print("=== Testing batch_embed() ===")

    texts = [
        "Python is a programming language.",
        "Azure OpenAI provides AI models.",
        "Embeddings represent text as vectors.",
        "Asynchronous programming improves I/O concurrency.",
        "This is the fifth test sentence.",
    ]

    batch_size = 2
    embeddings = await embedder.batch_embed(texts, chunk_size=batch_size)

    assert len(embeddings) == len(texts), f"Expected {len(texts)} embeddings, got {len(embeddings)}"

    for index, vector in enumerate(embeddings):
        assert len(vector) == 1024, f"Embedding {index} has {len(vector)} dimensions"
        print(f"Text {index + 1}: {texts[index]}\n Dimensions: {len(vector)}\n First 5 values: {vector[:5]}\n")

    print("batch_embed() passed!")
    print("\nAll tests passed successfully.")


if __name__ == "__main__":
    asyncio.run(main())