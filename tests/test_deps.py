"""Regression tests for app/api/v1/deps.py's internal-token auth gate.

Devin-flagged (app/api/v1/deps.py:17): the token comparison used plain `!=`,
which short-circuits at the first differing character on Python str — an
attacker with network access to the internal endpoint could recover the
shared secret byte-by-byte from response timing. This guards a write path
over the whole acronym dictionary (POST /api/acronyms/bulk), so the secret
is worth protecting. Fixed with secrets.compare_digest (constant-time for
equal-length inputs).

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
        # decided the outcome.
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "correct-secret"), \
             patch("app.api.v1.deps.secrets.compare_digest", return_value=True) as mock_compare:
            verify_internal_token("anything")
        mock_compare.assert_called_once_with("anything", "correct-secret")

    def test_correct_length_wrong_content_still_rejected(self):
        # Same length as the real secret (a case where naive `!=` and
        # compare_digest could differ in behavior if implemented wrong) —
        # must still be rejected.
        with patch("app.api.v1.deps.settings.INTERNAL_API_TOKEN", "abcdefgh"):
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_token("abcdefgx")
        assert exc_info.value.status_code == 401
