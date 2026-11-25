
from contextlib import AsyncExitStack
from typing import Any

import httpx2

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool, CallToolResult

from app.core.config import settings


class GitHubMCPClient:
    """
    Manages an authenticated GitHub MCP connection.

    Create one client instance per connected EmbeddedSense user.
    Never share an authenticated client across users.
    """

    def __init__(self):
        self.session: ClientSession | None = None
        self.tools: list[Tool] = []

        self._exit_stack: AsyncExitStack | None = None
        self._connected = False

    async def connect(self, access_token: str) -> list[Tool]:
        """
        Establish an authenticated Streamable HTTP connection
        to GitHub's remote MCP server.
        """

        if self._connected and self.session is not None:
            return self.tools

        if not access_token:
            raise ValueError("GitHub access token is required")

        exit_stack = AsyncExitStack()

        try:
            http_client = httpx2.AsyncClient(
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json, text/event-stream",
                },
                timeout=httpx2.Timeout(
                    30.0,
                    read=300.0,
                ),
            )

            await exit_stack.enter_async_context(http_client)

            read_stream, write_stream = (
                await exit_stack.enter_async_context(
                    streamable_http_client(
                        settings.GITHUB_MCP_URL,
                        http_client=http_client,
                    )
                )
            )

            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await session.initialize()

            result = await session.list_tools()

            self.session = session
            self.tools = result.tools
            self._exit_stack = exit_stack
            self._connected = True

            return self.tools

        except Exception:
            await exit_stack.aclose()

            self.session = None
            self.tools = []
            self._exit_stack = None
            self._connected = False

            raise

    async def list_tools(self) -> list[Tool]:
        """Discover tools exposed by GitHub MCP."""

        if not self._connected or self.session is None:
            raise RuntimeError("GitHub MCP is not connected")

        result = await self.session.list_tools()

        self.tools = result.tools

        return self.tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        """Execute an available GitHub MCP tool."""

        if not self._connected or self.session is None:
            raise RuntimeError("GitHub MCP is not connected")

        allowed_tools = {
            tool.name for tool in self.tools
        }

        if tool_name not in allowed_tools:
            raise ValueError(
                f"Unknown GitHub MCP tool: {tool_name}"
            )

        return await self.session.call_tool(
            tool_name,
            arguments=arguments,
        )

    async def disconnect(self) -> None:
        """Close the MCP connection and release resources."""

        exit_stack = self._exit_stack

        self.session = None
        self.tools = []
        self._exit_stack = None
        self._connected = False

        if exit_stack is not None:
            await exit_stack.aclose()

    @property
    def is_connected(self) -> bool:
        return self._connected
