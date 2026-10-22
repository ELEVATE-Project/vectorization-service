import secrets
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings
from app.constants import messages as msg


def verify_internal_token(
    internal_access_token: Optional[str] = Header(None, convert_underscores=False),
    # Hyphenated name also accepted: Nginx drops headers with underscores by default.
    internal_access_token_alias: Optional[str] = Header(
        None, alias="internal-access-token", convert_underscores=False
    ),
) -> None:
    """401 unless the internal_access_token header matches INTERNAL_API_TOKEN.

    A missing token gets the same 401 as a wrong one. The comparison is
    constant-time and does not fail on non-ASCII header bytes.
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
        raise HTTPException(status_code=401, detail=msg.INTERNAL_TOKEN_INVALID)


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
        raise HTTPException(status_code=401, detail=msg.ADMIN_TOKEN_INVALID)
