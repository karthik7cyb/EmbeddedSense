
import asyncio
import json
import logging
import random
import re

from collections.abc import AsyncIterator

from app.ai.rag.pipeline import rag_pipeline
from app.ai.models import models
from app.ai.prompts.gemini_prompt import RAG_PROMPT_TEMPLATE


logger = logging.getLogger("uvicorn.error")

NO_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the available "
    "hardware documentation to answer your question."
)


PAGE_CITATION_PATTERN = re.compile(
    r"\s*\((?:see\s+)?pages?\s*[^)]*\d+[^)]*\)",
    re.IGNORECASE
)


class ChatService:

    # ---------------------------------------------------------
    # RETRY WITH EXPONENTIAL BACKOFF AND JITTER
    # ---------------------------------------------------------

    @staticmethod
    async def retry_with_jitter(
        func,
        *args,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
        **kwargs
    ):

        for attempt in range(1, max_retries + 1):

            try:
                return await func(*args, **kwargs)

            except Exception as error:

                if attempt >= max_retries:

                    logger.error(
                        "Request failed after %s attempts: %s",
                        attempt,
                        error
                    )

                    raise

                backoff = min(
                    max_delay,
                    base_delay * (2 ** (attempt - 1))
                )

                sleep_time = random.uniform(0, backoff)

                logger.warning(
                    "Attempt %s/%s failed: %s. Retrying in %.2fs.",
                    attempt,
                    max_retries,
                    error,
                    sleep_time
                )

                await asyncio.sleep(sleep_time)

    # ---------------------------------------------------------
    # CLEAN STREAM (REMOVE PAGE CITATIONS)
    # ---------------------------------------------------------

    @staticmethod
    async def _clean_stream(
        stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:

        buffer = ""

        async for token in stream:

            buffer += token

            while True:

                open_paren = buffer.find("(")

                if open_paren == -1:

                    if buffer:
                        yield buffer
                        buffer = ""

                    break

                close_paren = buffer.find(")", open_paren)

                if close_paren != -1:

                    start_idx = open_paren

                    while start_idx > 0 and buffer[start_idx - 1] == " ":
                        start_idx -= 1

                    candidate = buffer[start_idx:close_paren + 1]

                    if (
                        PAGE_CITATION_PATTERN.fullmatch(candidate.strip())
                        or PAGE_CITATION_PATTERN.fullmatch(candidate)
                    ):

                        if start_idx > 0:
                            yield buffer[:start_idx]

                        buffer = buffer[close_paren + 1:]

                        if start_idx == 0:
                            buffer = buffer.lstrip()

                    else:

                        yield buffer[:close_paren + 1]
                        buffer = buffer[close_paren + 1:]

                else:

                    if open_paren > 0:

                        start_idx = open_paren

                        while start_idx > 0 and buffer[start_idx - 1] == " ":
                            start_idx -= 1

                        if start_idx > 0:
                            yield buffer[:start_idx]
                            buffer = buffer[start_idx:]

                    if len(buffer) > 100:
                        yield buffer
                        buffer = ""

                    break

        if buffer:
            cleaned = PAGE_CITATION_PATTERN.sub("", buffer)
            if cleaned:
                yield cleaned

    # ---------------------------------------------------------
    # OPEN MODEL STREAM
    # ---------------------------------------------------------

    @staticmethod
    async def _open_stream(
        model_func,
        prompt: str
    ) -> tuple[str, AsyncIterator[str]]:

        stream = ChatService._clean_stream(model_func(prompt))

        first_token = await anext(stream)

        return first_token, stream

    # ---------------------------------------------------------
    # SERVER-SENT EVENTS
    # ---------------------------------------------------------

    @staticmethod
    def _sse(event: dict) -> str:

        return f"data: {json.dumps(event)}\n\n"

    # ---------------------------------------------------------
    # MAIN CHAT FUNCTION
    # ---------------------------------------------------------

    @staticmethod
    async def chat(user_prompt: str) -> AsyncIterator[str]:

        """
        Retrieves relevant hardware documentation and generates
        a streamed response using the configured LLM fallback chain.

        Event sequence:
        model -> token -> sources -> done

        If retrieval or all providers fail:
        error
        """

        # 1. RETRIEVE RELEVANT DOCUMENTATION

        try:

            retrieval = await ChatService.retry_with_jitter(
                rag_pipeline.retrieve,
                user_prompt
            )

        except Exception:

            logger.exception("Document retrieval failed.")

            yield (
                "Unable to search the hardware documentation. "
                "Please try again."
            )

            return

        # 2. CHECK RETRIEVAL RESULTS

        if not retrieval["results"]:

            yield NO_CONTEXT_MESSAGE
            return

        # 3. BUILD RAG PROMPT

        context, sources = rag_pipeline.build_context(
            retrieval["results"]
        )

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=user_prompt
        )

        logger.info("Starting EmbeddedSense LLM request.")

        # 4. GET PROVIDER FALLBACK CHAIN

        fallback_chain = models.get_fallback_chain()

        # Expected priority:
        # Gemini -> Groq

        for model_name, model_func in fallback_chain:

            logger.info("Trying LLM provider: %s", model_name)

            try:

                # Retry until the first token is received.
                first_token, stream = await ChatService.retry_with_jitter(
                    ChatService._open_stream,
                    model_func,
                    prompt
                )

            except Exception:

                logger.exception(
                    "Provider %s failed. Trying next provider.",
                    model_name
                )

                continue

            # 5. STREAM THE RESPONSE

            yield first_token

            try:

                async for token in stream:

                    yield token

            except Exception:

                logger.exception(
                    "Provider %s failed during streaming.",
                    model_name
                )

                yield (
                    f"{model_name} failed while generating "
                    "the response. Please retry your question."
                )

                return

            # 6. RESPONSE COMPLETED

            logger.info(
                "Successfully answered using %s.",
                model_name
            )

            return

        # 7. ALL PROVIDERS FAILED

        yield (
            "All configured LLM providers failed. "
            "Please check API credentials, quotas and network connectivity."
        )


chat_service = ChatService()
