import asyncio
from app.ai.rag.data_loader import data_processor


async def main():
    files = data_processor.list_files("uploads")

    print(f"\nTotal supported files found: {len(files)}")

    if not files:
        print("No supported files found in uploads folder.")
        return

    for file in files:
        print(f"\n{'=' * 50}")
        print(f"Processing: {file.name}")

        try:
            result = await data_processor.process_file(file)

            print("Filename:", result["filename"])
            print("File size:", result["file_size_bytes"], "bytes")
            print("Pages extracted:", len(result["pages"]))
            print("Total chunks:", result["total_chunks"])
            print("SHA-256:", result["file_hash"])

            if result["chunks"]:
                print("\nFirst chunk:")
                print(result["chunks"][0]["text"][:300])

        except Exception as error:
            print("Processing failed:", error)


if __name__ == "__main__":
    asyncio.run(main())