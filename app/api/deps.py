import secrets
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def verify_internal_token(
    internal_access_token: Optional[str] = Header(None, convert_underscores=False),
    # Nginx drops underscored header names by default (underscores_in_headers
    # off) unless an operator explicitly re-enables them, and no Nginx config
    # ships in this repo to confirm that's been done. A hyphenated fallback
    # survives that default, so a deployment sitting behind an unconfigured
    # proxy doesn't silently lose this header in transit.
    internal_access_token_alias: Optional[str] = Header(
        None, alias="internal-access-token", convert_underscores=False
    ),
) -> None:
    """Gate internal-only endpoints behind a shared secret (internal_access_token
    header — underscored, not hyphenated, to stay consistent with FastAPI's Header param naming;
    internal-access-token also accepted, see above).

    Header is optional so a missing one fails with the same 401 as a wrong one,
    not a 422 that leaks the auth mechanism. compare_digest on utf-8/
    surrogateescape-encoded bytes gives constant-time comparison (no timing
    side-channel) without crashing on non-ASCII header bytes.
    """
    token = internal_access_token or internal_access_token_alias
    if (
        not settings.INTERNAL_API_TOKEN
        or not token
        or not secrets.compare_digest(
            token.encode("utf-8", "surrogateescape"),
            settings.INTERNAL_API_TOKEN.encode("utf-8", "surrogateescape"),
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing internal token")


def verify_admin_token(
    admin_auth_token: Optional[str] = Header(None, convert_underscores=False),
    admin_auth_token_alias: Optional[str] = Header(
        None, alias="admin-auth-token", convert_underscores=False
    ),
) -> None:
    """Gate admin-only endpoints behind a shared secret (admin_auth_token header —
    underscored, not hyphenated, to stay consistent with FastAPI's Header param naming;
    admin-auth-token also accepted — same Nginx reasoning as verify_internal_token).
    """
    token = admin_auth_token or admin_auth_token_alias
    if (
        not settings.ADMIN_API_TOKEN
        or not token
        or not secrets.compare_digest(
            token.encode("utf-8", "surrogateescape"),
            settings.ADMIN_API_TOKEN.encode("utf-8", "surrogateescape"),
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")
