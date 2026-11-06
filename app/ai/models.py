
from collections.abc import AsyncIterator

from google import genai
from groq import AsyncGroq
from google.genai import types
from app.core.config import settings
from app.ai.prompts.gemini_prompt import SYSTEM_PROMPT


class Models:

    # Model configuration
    GEMINI_MODEL = "gemini-2.5-flash"
    GROQ_MODEL = "openai/gpt-oss-20b"

    def __init__(self):

        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing from .env")

        self.gemini_client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                api_version="v1beta"
            )
        )

        self.groq_client = AsyncGroq(
            api_key=settings.GROQ_API_KEY
        )

    # Gemini - Primary LLM
    async def stream_gemini(
        self,
        prompt: str
    ) -> AsyncIterator[str]:

        response = await self.gemini_client.aio.models.generate_content_stream(
            model=self.GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT
            }
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    # Groq - Fallback LLM
    async def stream_groq(
        self,
        prompt: str
    ) -> AsyncIterator[str]:

        response = await self.groq_client.chat.completions.create(
            model=self.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
            temperature=1.0
        )

        async for chunk in response:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                yield content

    # Fallback priority
    def get_fallback_chain(self):

        return [
            ("Gemini", self.stream_gemini),
            ("Groq", self.stream_groq)
        ]


models = Models()
