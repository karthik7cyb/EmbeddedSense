from app.core.config import settings

print("Gemini API Key Loaded:", bool(settings.GEMINI_API_KEY))
print("Embedding Model:", settings.EMBEDDING_MODEL)