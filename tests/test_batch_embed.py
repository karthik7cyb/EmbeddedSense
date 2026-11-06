
import asyncio

from app.ai.rag.embed import embedding
from app.ai.rag.data_loader import data_processor


async def main():

    print("Loading ESP32 documents...")

    files = data_processor.list_files("uploads")

    all_chunks = []

    for file in files:
        result = await data_processor.process_file(file)
        all_chunks.extend(result["chunks"])

    print("Total chunks available:", len(all_chunks))

    # Test only the first 5 chunks
    test_chunks = [
        chunk["text"] for chunk in all_chunks[:5]
    ]

    print("\nGenerating batch embeddings...")

    vectors = await embedding.embed_chunks(test_chunks)

    print("Batch embedding successful!")
    print("Input chunks:", len(test_chunks))
    print("Generated vectors:", len(vectors))

    for index, vector in enumerate(vectors):
        print(
            f"Chunk {index + 1}: "
            f"{len(vector)} dimensions"
        )


if __name__ == "__main__":
    asyncio.run(main())
