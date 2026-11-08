
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai.chat import chat_service


router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        description="Hardware troubleshooting question",
        examples=[
            "Why does my ESP32 reset when a motor starts?"
        ]
    )


@router.post("/llm")
async def chat(
    request: Optional[ChatRequest] = None,
    prompt: Optional[str] = Query(None)
):

    """
    Accepts prompts through:

    JSON body:
        {"prompt": "Why does my ESP32 reset?"}

    Query parameter:
        /llm?prompt=Why%20does%20my%20ESP32%20reset

    Streams the response.
    """

    user_prompt = (request.prompt if request else None) or prompt

    if not user_prompt or not user_prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )

    return StreamingResponse(
        chat_service.chat(user_prompt.strip()),
        media_type="text/plain",
        status_code=200,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
