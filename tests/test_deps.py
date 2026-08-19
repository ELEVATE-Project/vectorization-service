"""Regression tests for app/api/v1/deps.py's internal-token auth gate.

Devin-flagged (app/api/v1/deps.py:17): the token comparison used plain `!=`,
which short-circuits at the first differing character on Python str — an
attacker with network access to the internal endpoint could recover the
shared secret byte-by-byte from response timing. This guards a write path
over the whole acronym dictionary (POST /api/acronyms/bulk), so the secret
is worth protecting. Fixed with secrets.compare_digest (constant-time for
equal-length inputs).

Second Devin finding on the same function (app/api/v1/deps.py:24), found
one round later: secrets.compare_digest itself raises TypeError on any
non-ASCII str — and Starlette decodes headers as latin-1, so any header
byte > 0x7F produces a non-ASCII str. Unhandled, that TypeError isn't an
HTTPException and surfaced as an opaque 500 instead of a clean 401 (a
client-triggerable 500, confirmed live against the real endpoint before
this second fix). Fixed by encoding both sides to bytes (utf-8,
surrogateescape, which never raises on encode) before comparing.

Pure function tests — verify_internal_token is called directly, no FastAPI
app/TestClient needed.
"""
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.v1.deps import verify_internal_token


class TestVerifyInternalToken:
    def test_correct_token_passes(self):
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "correct-secret"):
            verify_internal_token("correct-secret")  # must not raise

    def test_wrong_token_rejected(self):
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "correct-secret"):
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_token("wrong-secret")
        assert exc_info.value.status_code == 401

    def test_missing_token_rejected(self):
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "correct-secret"):
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_token(None)
        assert exc_info.value.status_code == 401

    def test_unset_server_token_rejects_every_request(self):
        # No fallback/default token — an unset INTERNAL_API_TOKEN must reject
        # every request, not accept an empty header value as a "match".
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", ""):
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_token("")
        assert exc_info.value.status_code == 401

    def test_uses_constant_time_comparison_not_plain_equality(self):
        # Confirms the fix is actually in place — not just that behavior
        # happens to look right, but that the timing-safe primitive is what
        # decided the outcome. Called with encoded bytes, not raw str — see
        # TestNonAsciiTokenHandling below for why.
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "correct-secret"), \
             patch("app.api.v1.deps.secrets.compare_digest", return_value=True) as mock_compare:
            verify_internal_token("anything")
        mock_compare.assert_called_once_with(b"anything", b"correct-secret")

    def test_correct_length_wrong_content_still_rejected(self):
        # Same length as the real secret (a case where naive `!=` and
        # compare_digest could differ in behavior if implemented wrong) —
        # must still be rejected.
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "abcdefgh"):
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_token("abcdefgx")
        assert exc_info.value.status_code == 401


class TestNonAsciiTokenHandling:
    """secrets.compare_digest raises TypeError on non-ASCII str — Starlette
    decodes headers as latin-1, so any header byte > 0x7F produces exactly
    that. Must degrade to a clean 401, never an unhandled 500."""

    def test_non_ascii_header_rejected_cleanly_not_500(self):
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "correct-secret"):
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_token("café-token")  # non-ASCII, wrong anyway
        assert exc_info.value.status_code == 401

    def test_non_ascii_configured_secret_does_not_crash(self):
        # The degenerate case explicitly called out in the finding: if
        # INTERNAL_API_TOKEN itself were ever set to a non-ASCII value,
        # every request must still degrade to 401, not break entirely.
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "café-secret"):
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_token("wrong-guess")
        assert exc_info.value.status_code == 401

    def test_matching_non_ascii_token_still_authenticates(self):
        # Not just "doesn't crash" — a genuinely correct non-ASCII token on
        # both sides must still be accepted.
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "café-secret"):
            verify_internal_token("café-secret")  # must not raise
