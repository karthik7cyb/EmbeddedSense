import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # GitHub OAuth configuration
    GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
    GITHUB_REDIRECT_URI = os.getenv(
        "GITHUB_REDIRECT_URI",
        "http://localhost:8000/auth/github/callback",
    )

    # GitHub MCP configuration
    GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

    SUPABASE_URL = os.getenv("SUPABASE_URL")


settings = Settings()