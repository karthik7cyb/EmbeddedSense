
from typing import Literal

from pydantic import BaseModel, Field

from app.ai.tools.agent import router_llm


class RouteDecision(BaseModel):
    route: Literal["rag", "tool", "github"] = Field(
        description=(
            "Choose rag for hardware questions, "
            "tool for existing tools such as currency conversion, "
            "or github for GitHub repository operations."
        )
    )


structured_router = router_llm.with_structured_output(RouteDecision)


async def route_query(query: str) -> str:

    decision = await structured_router.ainvoke(
        [
            (
                "system",
                """
                You are the router for EmbeddedSense.

                Choose:
                - rag: Hardware, electronics, embedded systems,
                  troubleshooting and technical documentation.
                - tool: Queries requiring existing general tools,
                  currently USD to INR currency conversion.
                - github: Requests involving GitHub repositories,
                  source code, commits, branches, issues and pull requests.

                Do not answer the query. Only classify it.
                """
            ),
            ("human", query),
        ]
    )

    return decision.route
