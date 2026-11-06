
import asyncio
import hashlib
from pathlib import Path

import pymupdf

from app.core.security import get_file_hash


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DataPreprocessor:

    @staticmethod
    def compute_hash(path: str | Path) -> str:
        """Returns SHA-256 hash of a file's bytes."""
        return get_file_hash(Path(path).read_bytes())

    @staticmethod
    def build_metadata(
        path: Path,
        file_hash: str,
        metadata: dict | None = None
    ) -> dict:
        """Creates simple document metadata."""

        return {
            "document_id": file_hash,
            "filename": path.name,
            "file_path": str(path),
            "file_hash": file_hash,
            "file_size_bytes": path.stat().st_size,
            "file_extension": path.suffix.lower(),

            # Component metadata
            "component_family": None,
            "manufacturer": None,
            "component_model": None,
            "component_variant": None,

            # Document metadata
            "document_type": None,
            "document_version": None,
            "revision": None,
            "source_type": None,

            # Additional classification
            "category": None,

            **(metadata or {})
        }

    @staticmethod
    def load_txt(
        path: str | Path,
        metadata: dict | None = None
    ) -> dict:
        """Reads a plain text or Markdown file."""

        path = Path(path)
        file_bytes = path.read_bytes()

        text = file_bytes.decode("utf-8-sig", errors="replace")
        file_hash = get_file_hash(file_bytes)

        content = DataPreprocessor.build_metadata(
            path, file_hash, metadata
        )

        content["pages"] = (
            [{"page_number": 1, "text": text}]
            if text.strip()
            else []
        )

        return content

    @staticmethod
    def load_pdf(
        path: str | Path,
        metadata: dict | None = None
    ) -> dict:
        """Reads a PDF and extracts text page by page."""

        path = Path(path)
        file_bytes = path.read_bytes()
        file_hash = get_file_hash(file_bytes)

        pages = []

        try:
            with pymupdf.open(str(path)) as pdf:

                total_pdf_pages = len(pdf)

                for page in pdf:
                    page_text = page.get_text("text").strip()

                    if page_text:
                        pages.append({
                            "page_number": page.number + 1,
                            "text": page_text
                        })

        except Exception as error:
            raise RuntimeError(
                f"Error reading PDF '{path.name}': {error}"
            ) from error

        content = DataPreprocessor.build_metadata(
            path, file_hash, metadata
        )

        content["pages"] = pages
        content["total_pdf_pages"] = total_pdf_pages
        content["total_pages_extracted"] = len(pages)

        return content

    @staticmethod
    def load_file(
        path: str | Path,
        metadata: dict | None = None
    ) -> dict | None:
        """Loads a file according to its extension."""

        extension = Path(path).suffix.lower()

        if extension in (".txt", ".md"):
            return DataPreprocessor.load_txt(path, metadata)

        if extension == ".pdf":
            return DataPreprocessor.load_pdf(path, metadata)

        return None

    @staticmethod
    def chunk_text(
        pages: list[dict],
        chunk_size: int = 300,
        overlap: int = 50,
        document_metadata: dict | None = None
    ) -> list[dict]:
        """
        Splits page text into overlapping word-based chunks.

        Default:
            chunk_size = 300 words
            overlap = 50 words
        """

        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError("chunk_size must be an integer")

        if not isinstance(overlap, int) or isinstance(overlap, bool):
            raise TypeError("overlap must be an integer")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be >= 0 and smaller than chunk_size"
            )

        chunks = []
        chunk_index = 0
        step = chunk_size - overlap

        metadata = document_metadata or {}

        for page in pages:

            page_number = page["page_number"]
            words = page["text"].split()

            if not words:
                continue

            for start in range(0, len(words), step):

                chunk_words = words[start:start + chunk_size]
                chunk_text = " ".join(chunk_words)

                chunk_identity = (
                    f"{metadata.get('file_hash', '')}:"
                    f"{page_number}:"
                    f"{chunk_index}:"
                    f"{chunk_text}"
                )

                chunk_id = hashlib.sha256(
                    chunk_identity.encode("utf-8")
                ).hexdigest()

                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "text": chunk_text,

                    # Source and component metadata
                    "filename": metadata.get("filename"),
                    "file_hash": metadata.get("file_hash"),
                    "document_id": metadata.get("document_id"),
                    "component_family": metadata.get("component_family"),
                    "manufacturer": metadata.get("manufacturer"),
                    "component_model": metadata.get("component_model"),
                    "component_variant": metadata.get("component_variant"),
                    "document_type": metadata.get("document_type"),
                    "document_version": metadata.get("document_version"),
                    "revision": metadata.get("revision"),
                    "source_type": metadata.get("source_type"),
                    "category": metadata.get("category")
                })

                chunk_index += 1

                if start + chunk_size >= len(words):
                    break

        return chunks

    @staticmethod
    def list_files(folder_path: str | Path) -> list[Path]:
        """Returns all supported files in a folder."""

        folder = Path(folder_path)

        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            return []

        return sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def _process_file(
        path: str | Path,
        chunk_size: int = 300,
        overlap: int = 50,
        metadata: dict | None = None
    ) -> dict:

        content = DataPreprocessor.load_file(path, metadata)

        if content is None:
            raise ValueError(f"Unsupported file type: {path}")

        chunks = DataPreprocessor.chunk_text(
            pages=content["pages"],
            chunk_size=chunk_size,
            overlap=overlap,
            document_metadata=content
        )

        content["chunks"] = chunks
        content["total_chunks"] = len(chunks)

        return content

    @staticmethod
    async def process_file(
        path: str | Path,
        chunk_size: int = 300,
        overlap: int = 50,
        metadata: dict | None = None
    ) -> dict:
        """
        Loads a file and creates chunks in a separate thread.

        Preserves asynchronous processing without blocking
        the application's event loop.
        """

        return await asyncio.to_thread(
            DataPreprocessor._process_file,
            path,
            chunk_size,
            overlap,
            metadata
        )


# Create reusable processor instance
data_processor = DataPreprocessor()
