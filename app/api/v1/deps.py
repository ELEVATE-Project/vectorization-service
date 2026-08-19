import secrets
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def verify_internal_token(x_internal_token: Optional[str] = Header(None)) -> None:
    """Gate internal-only endpoints behind a shared secret (X-Internal-Token header).

    Header is optional at the FastAPI level so a missing header fails with the
    same 401 as a wrong one, instead of a 422 that would otherwise leak "a header
    is expected here" ahead of the auth check. No fallback/default token — an
    unset INTERNAL_API_TOKEN rejects every request rather than accepting an
    empty header value as valid.

    secrets.compare_digest, not `!=`: this guards a write path over the whole
    acronym dictionary (POST /api/acronyms/bulk), and plain string `!=` on
    Python str short-circuits at the first differing character — an attacker
    with network access could recover the token byte-by-byte from response
    timing. compare_digest runs in constant time for equal-length inputs.
    """
    if (
        not settings.INTERNAL_API_TOKEN
        or not x_internal_token
        or not secrets.compare_digest(x_internal_token, settings.INTERNAL_API_TOKEN)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing internal token")
