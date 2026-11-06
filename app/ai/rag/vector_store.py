
import json
import logging
from uuid import UUID

from app.db.postgres import get_pool

logger = logging.getLogger("uvicorn.error")

# Must match embed.py and the PostgreSQL vector column.
EMBEDDING_DIMENSIONS = 768


class VectorStore:

    # ---------------------------------------------------------
    # DATABASE INITIALIZATION
    # ---------------------------------------------------------

    async def ensure_vector_index(self):
        """
        Creates the pgvector extension, source and chunk tables,
        and the HNSW vector index if they do not exist.
        """

        pool = get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                await cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;
                """)

                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS sources (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                        file_hash TEXT UNIQUE NOT NULL,
                        filename TEXT NOT NULL,
                        file_path TEXT,
                        file_size_bytes BIGINT,
                        file_extension TEXT,

                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                await cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                        source_id UUID NOT NULL
                            REFERENCES sources(id)
                            ON DELETE CASCADE,

                        chunk_index INTEGER NOT NULL,
                        page_number INTEGER NOT NULL,
                        text TEXT NOT NULL,

                        filename TEXT,
                        file_hash TEXT,
                        document_id TEXT,

                        component_family TEXT,
                        manufacturer TEXT,
                        component_model TEXT,
                        component_variant TEXT,
                        document_type TEXT,
                        document_version TEXT,
                        revision TEXT,
                        source_type TEXT,
                        category TEXT,

                        embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,

                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)

                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_source_id_idx
                    ON chunks(source_id);
                """)

                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_file_hash_idx
                    ON chunks(file_hash);
                """)

                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                    ON chunks
                    USING hnsw (embedding vector_cosine_ops);
                """)

            await conn.commit()

        logger.info(
            "PostgreSQL tables and pgvector HNSW index are ready."
        )

    # ---------------------------------------------------------
    # DOCUMENT LOOKUPS
    # ---------------------------------------------------------

    async def get_document_by_hash(
        self,
        file_hash: str
    ) -> dict | None:
        """Finds a source document using its SHA-256 hash."""

        pool = get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                await cur.execute("""
                    SELECT *
                    FROM sources
                    WHERE file_hash = %s;
                """, (file_hash,))

                doc = await cur.fetchone()

                if doc:
                    return {
                        **(doc["metadata"] or {}),
                        "source_id": str(doc["id"]),
                        "file_hash": doc["file_hash"],
                        "filename": doc["filename"],
                        "file_path": doc["file_path"],
                        "file_size_bytes": doc["file_size_bytes"],
                        "file_extension": doc["file_extension"],
                    }

        return None

    async def get_document_by_id(
        self,
        source_id: str
    ) -> dict | None:
        """Finds a source document using its UUID."""

        try:
            source_uuid = UUID(source_id)
        except (ValueError, TypeError):
            return None

        pool = get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                await cur.execute("""
                    SELECT *
                    FROM sources
                    WHERE id = %s;
                """, (source_uuid,))

                doc = await cur.fetchone()

                if doc:
                    return {
                        **(doc["metadata"] or {}),
                        "source_id": str(doc["id"]),
                        "file_hash": doc["file_hash"],
                        "filename": doc["filename"],
                        "file_path": doc["file_path"],
                        "file_size_bytes": doc["file_size_bytes"],
                        "file_extension": doc["file_extension"],
                    }

        return None

    # ---------------------------------------------------------
    # DOCUMENT AND CHUNK INSERTION
    # ---------------------------------------------------------

    async def insert(
        self,
        document_record: dict,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> str:
        """
        Inserts a source document and its chunks with embeddings.

        If the source already exists, its existing chunks are
        replaced within the same database transaction.

        Returns the source UUID as a string.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) count do not match."
            )

        if not chunks:
            raise ValueError("Cannot insert a document without chunks.")

        for vector in embeddings:
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
                    f"received {len(vector)}."
                )

        pool = get_pool()

        # Exclude extracted page/chunk content from source metadata.
        excluded_fields = {
            "pages",
            "chunks",
            "total_chunks",
            "total_pdf_pages",
            "total_pages_extracted",
        }

        metadata = {
            key: value
            for key, value in document_record.items()
            if key not in excluded_fields
        }

        async with pool.connection() as conn:

            async with conn.cursor() as cur:

                # 1. Insert or find the source document.
                await cur.execute("""
                    INSERT INTO sources (
                        file_hash,
                        filename,
                        file_path,
                        file_size_bytes,
                        file_extension,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)

                    ON CONFLICT (file_hash)
                    DO UPDATE SET
                        filename = EXCLUDED.filename,
                        file_path = EXCLUDED.file_path,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        file_extension = EXCLUDED.file_extension,
                        metadata = EXCLUDED.metadata

                    RETURNING id;
                """, (
                    document_record["file_hash"],
                    document_record["filename"],
                    document_record.get("file_path"),
                    document_record.get("file_size_bytes"),
                    document_record.get("file_extension"),
                    json.dumps(metadata, default=str),
                ))

                result = await cur.fetchone()
                source_id = result["id"]

                # 2. Remove previous chunks for this source.
                await cur.execute("""
                    DELETE FROM chunks
                    WHERE source_id = %s;
                """, (source_id,))

                # 3. Prepare new chunk records.
                chunk_values = []

                for chunk, vector in zip(chunks, embeddings):

                    vector_string = "[" + ",".join(
                        str(float(value)) for value in vector
                    ) + "]"

                    chunk_values.append((
                        source_id,
                        chunk["chunk_index"],
                        chunk["page_number"],
                        chunk["text"],
                        chunk.get("filename"),
                        chunk.get("file_hash"),
                        chunk.get("document_id"),
                        chunk.get("component_family"),
                        chunk.get("manufacturer"),
                        chunk.get("component_model"),
                        chunk.get("component_variant"),
                        chunk.get("document_type"),
                        chunk.get("document_version"),
                        chunk.get("revision"),
                        chunk.get("source_type"),
                        chunk.get("category"),
                        vector_string,
                    ))

                # 4. Insert chunks and embeddings.
                await cur.executemany("""
                    INSERT INTO chunks (
                        source_id,
                        chunk_index,
                        page_number,
                        text,
                        filename,
                        file_hash,
                        document_id,
                        component_family,
                        manufacturer,
                        component_model,
                        component_variant,
                        document_type,
                        document_version,
                        revision,
                        source_type,
                        category,
                        embedding
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s::vector
                    );
                """, chunk_values)

            # Commit source and chunks together.
            await conn.commit()

        logger.info(
            "Stored source '%s' with %s chunks in PostgreSQL.",
            document_record["filename"],
            len(chunks),
        )

        return str(source_id)

    # ---------------------------------------------------------
    # VECTOR SEARCH / RETRIEVAL
    # ---------------------------------------------------------

    async def retrieve(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> list[dict]:
        """
        Retrieves the most relevant chunks using cosine distance.

        Optionally restricts retrieval to a specific source UUID.
        Lower cosine distance indicates greater similarity.
        """

        if len(query_embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
                f"received {len(query_embedding)}."
            )

        if n_results <= 0:
            return []

        pool = get_pool()

        vector_string = "[" + ",".join(
            str(float(value)) for value in query_embedding
        ) + "]"

        sql = """
            SELECT
                id,
                source_id,
                filename,
                chunk_index,
                page_number,
                text,
                embedding <=> %s::vector AS distance

            FROM chunks
        """

        params = [vector_string]

        if document_id:

            try:
                source_uuid = UUID(document_id)
            except (ValueError, TypeError):
                return []

            sql += """
                WHERE source_id = %s
            """

            params.append(source_uuid)

        sql += """
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        params.extend([vector_string, n_results])

        results = []

        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                await cur.execute(sql, params)

                rows = await cur.fetchall()

                for doc in rows:

                    results.append({
                        "id": str(doc["id"]),
                        "document": doc["text"],

                        "metadata": {
                            "source_id": str(doc["source_id"]),
                            "filename": doc["filename"],
                            "chunk_index": doc["chunk_index"],
                            "page_number": doc["page_number"],
                        },

                        "distance": float(doc["distance"]),
                    })

        return results


vector_store = VectorStore()

