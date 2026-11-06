
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.ai.rag.data_loader import data_processor
from app.ai.rag.embed import embedding
from app.ai.rag.vector_store import vector_store


logger = logging.getLogger("uvicorn.error")


class RagPipeline:

    @staticmethod
    async def ingest_file(path: str | Path):

        path = Path(path)
        uploaded_at = datetime.now(timezone.utc).isoformat()

        # 1. Check whether the document already exists
        file_hash = data_processor.compute_hash(path)

        existing_doc = await vector_store.get_document_by_hash(file_hash)

        if existing_doc:
            return {
                "status": "skipped",
                "message": (
                    f"File '{path.name}' already ingested as "
                    f"'{existing_doc.get('filename')}'."
                ),
                "document": existing_doc
            }

        # 2. Extract text and create chunks
        content = await data_processor.process_file(path)

        chunks = content["chunks"]

        if not chunks:
            return {
                "status": "empty",
                "message": f"File '{path.name}' has no extractable text.",
                "document": None
            }

        # 3. Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]

        embeddings = await embedding.embed_chunks(chunk_texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings."
            )

        # 4. Construct document record
        ingested_at = datetime.now(timezone.utc).isoformat()

        document_record = {
            "filename": content["filename"],
            "file_path": content["file_path"],
            "file_hash": file_hash,
            "file_size_bytes": content["file_size_bytes"],
            "file_extension": content["file_extension"],

            "uploaded_at": uploaded_at,
            "ingested_at": ingested_at,
            "ingestion_status": "completed",

            "chunk_count": content["total_chunks"],
            "chunking_version": "v1",

            # Document metadata
            "component_family": content.get("component_family"),
            "manufacturer": content.get("manufacturer"),
            "component_model": content.get("component_model"),
            "component_variant": content.get("component_variant"),
            "document_type": content.get("document_type"),
            "document_version": content.get("document_version"),
            "revision": content.get("revision"),
            "source_type": content.get("source_type"),
            "category": content.get("category"),

            # Embedding configuration
            "embedding": {
                "model_name": embedding.model_name,
                "dimensions": embedding.dimensions
            }
        }

        # 5. Store document, chunks and embeddings in PostgreSQL
        source_id = await vector_store.insert(
            document_record=document_record,
            chunks=chunks,
            embeddings=embeddings
        )

        return {
            "status": "completed",
            "message": (
                f"File '{path.name}' ingested with "
                f"{content['total_chunks']} chunks."
            ),
            "document": {
                **document_record,
                "source_id": str(source_id)
            }
        }

    # ---------------------------------------------------------
    # Bulk ingestion
    # ---------------------------------------------------------

    @staticmethod
    async def data_ingestion(folder_path: str | Path = "uploads"):

        """Ingests supported files and skips previously ingested documents."""

        files = data_processor.list_files(folder_path)

        summary = {
            "total": len(files),
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "results": []
        }

        if not files:
            logger.info("No files to ingest in '%s'", folder_path)
            return summary

        for path in files:

            try:
                result = await RagPipeline.ingest_file(path)

            except Exception as error:
                logger.exception(
                    "Failed to ingest '%s': %s",
                    path.name,
                    error
                )

                result = {
                    "status": "failed",
                    "message": str(error),
                    "document": None
                }

            status = result["status"]

            if status == "completed":
                summary["ingested"] += 1

            elif status == "failed":
                summary["failed"] += 1

            else:
                summary["skipped"] += 1

            logger.info("[%s] %s", status, result["message"])

            summary["results"].append({
                "file": path.name,
                **result
            })

        logger.info(
            "Ingestion finished: %s ingested, %s skipped, "
            "%s failed (of %s files)",
            summary["ingested"],
            summary["skipped"],
            summary["failed"],
            summary["total"]
        )

        return summary

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    @staticmethod
    async def retrieve(
        query: str,
        n_results: int = 5,
        document_id: str | None = None
    ):

        query_embedding = await embedding.embed_query(query)

        results = await vector_store.retrieve(
            query_embedding=query_embedding,
            n_results=n_results,
            document_id=document_id
        )

        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }

    # ---------------------------------------------------------
    # Context construction
    # ---------------------------------------------------------

    @staticmethod
    def build_context(
        results: list[dict]
    ) -> tuple[str, list[dict]]:

        """Formats retrieved chunks and creates a unique source list."""

        context_blocks = []
        sources = []
        seen = set()

        for result in results:

            metadata = result.get("metadata") or {}

            filename = metadata.get("filename", "unknown")
            page_number = metadata.get("page_number")

            context_blocks.append(
                f"[Source: {filename}, page {page_number}]\n"
                f"{result['document']}"
            )

            source_key = (filename, page_number)

            if source_key not in seen:

                seen.add(source_key)

                sources.append({
                    "filename": filename,
                    "page_number": page_number,
                    "distance": result.get("distance")
                })

        return "\n\n".join(context_blocks), sources


rag_pipeline = RagPipeline()
