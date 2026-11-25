
import jwt

from jwt import PyJWKClient
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings


security = HTTPBearer()

if not settings.SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is not configured")

SUPABASE_URL = settings.SUPABASE_URL.rstrip("/")

JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
JWT_ISSUER = f"{SUPABASE_URL}/auth/v1"

jwks_client = PyJWKClient(JWKS_URL)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Verify a Supabase ES256 access token."""

    token = credentials.credentials

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
            issuer=JWT_ISSUER,
        )

        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
            )

        return {
            "id": user_id,
            "email": payload.get("email"),
            "claims": payload,
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please sign in again.",
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
        )

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed",
        )
