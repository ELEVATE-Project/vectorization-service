"""Tests for metadata.markdown_url safety checks: allowed hosts, no redirects, size limit."""
import asyncio

import httpx
import pytest
from fastapi import HTTPException

from app.config import settings
from app.services import url_text_extractor as extractor_module
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.services.url_text_extractor import URLTextExtractor
from app.validations.generic import is_url_host_allowed

ALLOWED_HOST = "static.example.org"


def validate_markdown_url(markdown_url):
    """Run the upload field check with only metadata.markdown_url set."""
    operation = BaseDocumentOperation()
    metadata = {"markdown_url": markdown_url}
    return operation.validate_document_fields("doc-1", None, None, None, None, metadata)


def use_fake_server(monkeypatch, handler):
    """Make URLTextExtractor talk to `handler` instead of the network."""
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(extractor_module.httpx, "AsyncClient", fake_async_client)


class TestIsUrlHostAllowed:
    def test_star_allows_any_host(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", "*")
        assert is_url_host_allowed("anything.example.com")
        assert is_url_host_allowed("localhost")

    def test_listed_host_is_allowed_any_case_and_spacing(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", f" other.org , {ALLOWED_HOST.upper()} ")
        assert is_url_host_allowed(ALLOWED_HOST)
        assert is_url_host_allowed("Other.Org")

    def test_unlisted_hosts_are_refused(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", ALLOWED_HOST)
        assert not is_url_host_allowed("localhost")
        assert not is_url_host_allowed("169.254.169.254")
        assert not is_url_host_allowed(f"sub.{ALLOWED_HOST}")

    def test_empty_setting_refuses_every_host(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", "")
        assert not is_url_host_allowed(ALLOWED_HOST)


class TestMarkdownUrlUploadCheck:
    def test_allowed_host_passes_and_url_is_stripped(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", ALLOWED_HOST)
        _, _, _, _, metadata = validate_markdown_url(f"  https://{ALLOWED_HOST}/doc.md  ")
        assert metadata["markdown_url"] == f"https://{ALLOWED_HOST}/doc.md"

    def test_unlisted_host_is_refused_with_400(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", ALLOWED_HOST)
        with pytest.raises(HTTPException) as error:
            validate_markdown_url("http://localhost:6333/collections")
        assert error.value.status_code == 400
        assert "localhost" in error.value.detail

    def test_user_info_trick_is_checked_against_the_real_host(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", ALLOWED_HOST)
        with pytest.raises(HTTPException) as error:
            validate_markdown_url(f"http://{ALLOWED_HOST}@169.254.169.254/latest/meta-data/")
        assert error.value.status_code == 400
        assert "169.254.169.254" in error.value.detail

    def test_star_allows_any_host(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", "*")
        _, _, _, _, metadata = validate_markdown_url("http://internal-docs.local/page")
        assert metadata["markdown_url"] == "http://internal-docs.local/page"

    def test_http_and_https_are_both_accepted(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", ALLOWED_HOST)
        validate_markdown_url(f"http://{ALLOWED_HOST}/doc.md")
        validate_markdown_url(f"https://{ALLOWED_HOST}/doc.md")

    def test_other_schemes_are_still_refused(self, monkeypatch):
        monkeypatch.setattr(settings, "ALLOWED_URL_HOSTS", "*")
        with pytest.raises(HTTPException) as error:
            validate_markdown_url(f"ftp://{ALLOWED_HOST}/doc.md")
        assert error.value.status_code == 400


class TestUrlFetch:
    def test_plain_text_page_is_returned(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="Zebra garden notes")

        use_fake_server(monkeypatch, handler)
        text = asyncio.run(URLTextExtractor().extract_text(f"https://{ALLOWED_HOST}/notes.txt"))
        assert text == "Zebra garden notes"

    def test_redirect_is_refused_and_not_followed(self, monkeypatch):
        requested_urls = []

        def handler(request):
            requested_urls.append(str(request.url))
            return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})

        use_fake_server(monkeypatch, handler)
        with pytest.raises(HTTPException) as error:
            asyncio.run(URLTextExtractor().extract_text(f"https://{ALLOWED_HOST}/doc.md"))
        assert error.value.status_code == 400
        assert "169.254.169.254" in error.value.detail
        assert requested_urls == [f"https://{ALLOWED_HOST}/doc.md"]

    def test_page_over_size_limit_is_refused_with_413(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, headers={"Content-Type": "text/plain"}, content=b"x" * (1024 * 1024 + 1))

        use_fake_server(monkeypatch, handler)
        monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
        with pytest.raises(HTTPException) as error:
            asyncio.run(URLTextExtractor().extract_text(f"https://{ALLOWED_HOST}/big.txt"))
        assert error.value.status_code == 413
