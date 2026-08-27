import secrets
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def verify_internal_token(
    internal_access_token: Optional[str] = Header(None, convert_underscores=False),
) -> None:
    """Gate internal-only endpoints behind a shared secret (internal_access_token
    header — underscored, not hyphenated, to stay consistent with FastAPI's Header param naming).

    Header is optional so a missing one fails with the same 401 as a wrong one,
    not a 422 that leaks the auth mechanism. compare_digest on utf-8/
    surrogateescape-encoded bytes gives constant-time comparison (no timing
    side-channel) without crashing on non-ASCII header bytes.
    """
    if (
        not settings.INTERNAL_API_TOKEN
        or not internal_access_token
        or not secrets.compare_digest(
            internal_access_token.encode("utf-8", "surrogateescape"),
            settings.INTERNAL_API_TOKEN.encode("utf-8", "surrogateescape"),
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing internal token")


def verify_admin_token(
    admin_auth_token: Optional[str] = Header(None, convert_underscores=False),
) -> None:
    """ Gate admin-only endpoints behind a shared secret (admin_auth_token header — underscored, not hyphenated, to stay consistent with FastAPI's Header param naming).
    """
    if (
        not settings.ADMIN_API_TOKEN
        or not admin_auth_token
        or not secrets.compare_digest(
            admin_auth_token.encode("utf-8", "surrogateescape"),
            settings.ADMIN_API_TOKEN.encode("utf-8", "surrogateescape"),
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")
