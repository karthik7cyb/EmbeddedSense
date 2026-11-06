
import asyncio

from app.ai.rag.embed import embedding
from app.ai.rag.data_loader import data_processor


async def main():

    print("Loading ESP32 document...")

    files = data_processor.list_files("uploads")

    if not files:
        print("No documents found.")
        return

    result = await data_processor.process_file(files[0])

    if not result["chunks"]:
        print("No chunks available.")
        return

    text = result["chunks"][0]["text"]

    print("Generating Gemini embedding...")

    vector = await embedding.embed_query(text)

    print("Embedding generated successfully!")
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])


if __name__ == "__main__":
    asyncio.run(main())
