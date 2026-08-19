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

    Encoded to bytes first (utf-8, surrogateescape) rather than compared as
    str directly: compare_digest raises TypeError on any non-ASCII str
    (hmac.compare_digest's restriction), and Starlette decodes headers as
    latin-1, so any byte > 0x7F in the header produces a non-ASCII str.
    Unhandled, that TypeError isn't an HTTPException and surfaces as an
    opaque 500 instead of a clean 401 — and worse, becomes a client-
    triggerable 500 available to anyone, or (if INTERNAL_API_TOKEN itself
    were ever set to a non-ASCII value) breaks every single request.
    surrogateescape never raises on encode, so this comparison is now total
    over all possible header values while staying constant-time.
    """
    if (
        not settings.INTERNAL_API_TOKEN
        or not x_internal_token
        or not secrets.compare_digest(
            x_internal_token.encode("utf-8", "surrogateescape"),
            settings.INTERNAL_API_TOKEN.encode("utf-8", "surrogateescape"),
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing internal token")
