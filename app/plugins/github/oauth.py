
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata

load_dotenv()


GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

GITHUB_REDIRECT_URI = os.getenv(
    "GITHUB_REDIRECT_URI",
    "http://localhost:8000/auth/github/callback",
)


def validate_github_oauth_config():
    """Validate GitHub OAuth configuration."""

    if not GITHUB_CLIENT_ID:
        raise RuntimeError("GITHUB_CLIENT_ID is missing")

    if not GITHUB_CLIENT_SECRET:
        raise RuntimeError("GITHUB_CLIENT_SECRET is missing")

    parsed_uri = urlparse(GITHUB_REDIRECT_URI)

    if parsed_uri.scheme not in ("http", "https"):
        raise RuntimeError("Invalid GitHub redirect URI")

    if not parsed_uri.netloc:
        raise RuntimeError("Invalid GitHub redirect URI")


def create_oauth_provider(
    redirect_handler,
    callback_handler,
    token_storage,
):
    """
    Create an OAuthClientProvider for GitHub MCP.

    redirect_handler:
        Handles opening or returning the authorization URL.

    callback_handler:
        Handles the OAuth callback authorization code.

    token_storage:
        Persistent per-user OAuth token storage.
    """

    validate_github_oauth_config()

    metadata = OAuthClientMetadata(
        client_name="EmbeddedSense",
        client_uri="https://github.com/",
        redirect_uris=[GITHUB_REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )

    return OAuthClientProvider(
        server_url=GITHUB_MCP_URL,
        client_metadata=metadata,
        storage=token_storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
