
import math

from google import genai
from google.genai import types

from app.ai.batchs.batch_processing import (
    BatchProcess,
    GEMINI_FREE_TIER_SETTINGS,
)
from app.core.config import settings


class Embedding:

    MODEL_NAME = "gemini-embedding-001"
    DIMENSIONS = 768

    def __init__(self, model_name: str | None = None):

        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.model_name = model_name or self.MODEL_NAME
        self.dimensions = self.DIMENSIONS

        self.batch_processor = BatchProcess(
            **GEMINI_FREE_TIER_SETTINGS
        )

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:

        magnitude = math.sqrt(
            sum(value * value for value in vector)
        )

        if magnitude == 0:
            raise ValueError("Cannot normalize a zero embedding vector")

        return [
            value / magnitude
            for value in vector
        ]

    async def _embed(
        self,
        contents: str | list[str],
        task_type: str
    ) -> list[list[float]]:

        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensions,
            ),
        )

        if not response.embeddings:
            raise ValueError("Gemini returned no embeddings")

        embeddings = []

        for item in response.embeddings:

            vector = item.values

            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, "
                    f"but received {len(vector)}"
                )

            # Gemini Embedding 001 requires normalization
            # when using reduced dimensions.
            vector = self._normalize(vector)

            embeddings.append(vector)

        return embeddings

    async def _embed_batch(
        self,
        batch: list[str]
    ) -> list[list[float]]:

        return await self._embed(
            batch,
            "RETRIEVAL_DOCUMENT"
        )

    async def embed_chunks(
        self,
        chunks: list[str]
    ) -> list[list[float]]:

        if not chunks:
            return []

        return await self.batch_processor.run(
            chunks,
            self._embed_batch
        )

    async def embed_query(
        self,
        prompt: str
    ) -> list[float]:

        if not prompt.strip():
            raise ValueError("Query cannot be empty")

        embeddings = await self._embed(
            prompt,
            "RETRIEVAL_QUERY"
        )

        return embeddings[0]


embedding = Embedding()
