
import json
import logging
import re

from collections.abc import AsyncIterator

from app.ai.rag.pipeline import rag_pipeline
from app.ai.models import models
from app.ai.tools.router import route_query
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
    # SERVER-SENT EVENTS
    # ---------------------------------------------------------

    @staticmethod
    def _sse(event: str, data: dict) -> str:

        return (
            f"event: {event}\n"
            f"data: {json.dumps(data)}\n\n"
        )

    # ---------------------------------------------------------
    # MAIN CHAT FUNCTION
    # ---------------------------------------------------------

    @staticmethod
    async def chat(user_prompt: str) -> AsyncIterator[str]:

        """
        Routes the user query to RAG or TOOL
        and streams structured SSE events.

        Router: Groq
        Main LLM: Gemini
        """

        # 1. ROUTER STATUS

        yield ChatService._sse(
            "status",
            {"message": "Classifying your query..."}
        )

        # 2. ROUTE THE USER QUERY

        try:

            route = await route_query(user_prompt)

            logger.info("Selected route: %s", route)

        except Exception:

            logger.exception("Query routing failed.")

            yield ChatService._sse(
                "error",
                {"message": "Unable to classify your query. Please try again."}
            )
            return

        # 3. SEND ROUTE EVENT

        yield ChatService._sse(
            "route",
            {"route": route}
        )

        # -----------------------------------------------------
        # 4. TOOL ROUTE
        # -----------------------------------------------------

        if route == "tool":

            logger.info("Executing tool-based request.")

            yield ChatService._sse(
                "status",
                {"message": "Processing your tool request..."}
            )

            try:

                async for token in ChatService._clean_stream(
                    models.stream_llm(user_prompt)
                ):

                    yield ChatService._sse(
                        "token",
                        {"content": token}
                    )

            except Exception:

                logger.exception("Tool agent execution failed.")

                yield ChatService._sse(
                    "error",
                    {"message": "Unable to execute the requested tool. Please try again."}
                )
                return

            logger.info("Tool request completed.")

            yield ChatService._sse(
                "done",
                {"status": "completed"}
            )

            return

        # -----------------------------------------------------
        # 5. RAG ROUTE
        # -----------------------------------------------------

        yield ChatService._sse(
            "status",
            {"message": "Searching hardware documentation..."}
        )

        try:

            retrieval = await rag_pipeline.retrieve(user_prompt)

        except Exception:

            logger.exception("Document retrieval failed.")

            yield ChatService._sse(
                "error",
                {"message": "Unable to search the hardware documentation. Please try again."}
            )
            return

        # 6. CHECK RETRIEVAL RESULTS

        if not retrieval["results"]:

            yield ChatService._sse(
                "token",
                {"content": NO_CONTEXT_MESSAGE}
            )

            yield ChatService._sse(
                "done",
                {"status": "completed"}
            )

            return

        # 7. BUILD RAG PROMPT

        context, sources = rag_pipeline.build_context(
            retrieval["results"]
        )

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=user_prompt
        )

        logger.info("Starting EmbeddedSense RAG request.")

        yield ChatService._sse(
            "status",
            {"message": "Generating hardware diagnosis..."}
        )

        # 8. STREAM RAG RESPONSE

        try:

            async for token in ChatService._clean_stream(
                models.stream_llm(prompt)
            ):

                yield ChatService._sse(
                    "token",
                    {"content": token}
                )

        except Exception:

            logger.exception("RAG agent execution failed.")

            yield ChatService._sse(
                "error",
                {"message": "Unable to generate the diagnosis. Please try again."}
            )
            return

        # 9. RESPONSE COMPLETED

        logger.info("Successfully completed EmbeddedSense request.")

        yield ChatService._sse(
            "done",
            {"status": "completed"}
        )


chat_service = ChatService()
