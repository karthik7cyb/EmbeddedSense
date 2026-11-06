
import os
import logging

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

pool: AsyncConnectionPool | None = None


async def connect_to_postgres():
    """Initialize the asynchronous PostgreSQL connection pool."""

    global pool

    if not DATABASE_URL:
        raise ValueError("SUPABASE_DB_URL is missing from environment variables.")

    logger.info("Initializing Supabase PostgreSQL connection pool...")

    pool = AsyncConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        open=False,
    )

    try:
        await pool.open()
        await pool.wait()

        logger.info("Successfully connected to Supabase PostgreSQL.")

    except Exception:
        logger.exception("Failed to connect to Supabase PostgreSQL.")

        await pool.close()
        pool = None

        raise


async def close_postgres():
    """Close the PostgreSQL connection pool."""

    global pool

    if pool is not None:
        await pool.close()
        pool = None

        logger.info("PostgreSQL connection pool closed.")


def get_pool() -> AsyncConnectionPool:
    """Return the active PostgreSQL connection pool."""

    if pool is None:
        raise RuntimeError("PostgreSQL connection pool is not initialized.")

    return pool
