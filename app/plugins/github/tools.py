
from typing import Any

from app.plugins.github.client import GitHubMCPClient


class GitHubTools:
    """
    Discovers and executes tools exposed by GitHub MCP.
    """

    def __init__(self, client: GitHubMCPClient):
        self.client = client

    async def discover_tools(self) -> list[dict[str, Any]]:
        """Retrieve GitHub MCP tools and their schemas."""

        tools = await self.client.list_tools()

        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a discovered GitHub MCP tool."""

        result = await self.client.call_tool(
            tool_name=tool_name,
            arguments=arguments,
        )

        return {
            "is_error": result.isError,
            "content": [
                {
                    "type": item.type,
                    "text": getattr(item, "text", None),
                }
                for item in result.content
            ],
        }
