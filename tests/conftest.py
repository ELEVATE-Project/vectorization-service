"""
Pytest configuration and fixtures for API testing
"""
import pytest
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from typing import Dict, Any, List
import io

# Set test environment variables before importing app
os.environ["ENVIRONMENT"] = "test"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_CACHE_ENABLED"] = "False"

try:
    from app.main import app
except Exception as exc:
    # Pure unit tests don't need the full app, so we tolerate an import failure here
    # rather than erroring out collection. But keep the real exception around: any
    # fixture that actually needs `app` re-raises it (see the `client` fixture) so the
    # original traceback surfaces instead of a confusing NoneType error downstream.
    app = None
    _app_import_error = exc
else:
    _app_import_error = None

try:
    from tests.logger.test_logger import test_logger, log_test_start, log_test_end
except Exception:
    test_logger = None
    def log_test_start(logger, name): pass
    def log_test_end(logger, name, status): pass


def pytest_configure(config):
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "requires_qdrant: test needs a live Qdrant server; skipped if unreachable.",
    )
    config.addinivalue_line(
        "markers",
        "compat: client/server version-compatibility guard (live Qdrant required).",
    )


@pytest.fixture(scope="session")
def logger():
    """Provide test logger for all tests"""
    return test_logger


@pytest.fixture(scope="function")
def log_test(logger, request):
    """Automatically log test start and end"""
    test_name = request.node.name
    log_test_start(logger, test_name)
    
    yield logger
    
    # Determine test status
    if hasattr(request.node, 'rep_call'):
        status = "PASSED" if request.node.rep_call.passed else "FAILED"
    else:
        status = "COMPLETED"
    
    log_test_end(logger, test_name, status)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for logging"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(scope="function")
def mock_qdrant_client():
    """Mock Qdrant client for testing"""
    mock_client = Mock()
    
    # Mock get_collections
    mock_client.get_collections.return_value = Mock(
        collections=[
            Mock(name="documents"),
            Mock(name="qa_cache")
        ]
    )
    
    # Mock search
    mock_client.search.return_value = [
        Mock(
            id="test-id-1",
            score=0.95,
            payload={
                "text": "Sample document text",
                "source_id": "test-source-1",
                "company_id": "test-company",
                "metadata": {"type": "pdf", "page": 1},
                "tags": ["tag1", "tag2"],
                "title": "Test Document",
                "summary": "Test summary"
            }
        )
    ]
    
    # Mock upsert
    mock_client.upsert.return_value = Mock(status="completed")
    
    # Mock delete
    mock_client.delete.return_value = Mock(status="completed")
    
    # Mock scroll
    mock_client.scroll.return_value = (
        [
            Mock(
                id="test-id-1",
                payload={
                    "text": "Sample text",
                    "source_id": "test-source-1",
                    "metadata": {}
                }
            )
        ],
        None  # next_page_offset
    )
    
    # Mock count
    mock_client.count.return_value = Mock(count=10)
    
    return mock_client


@pytest.fixture(scope="function")
def mock_redis_client():
    """Mock Redis client for testing"""
    mock_redis = Mock()
    
    # Mock ping - should be synchronous, not async
    mock_redis.ping.return_value = True
    
    # Mock get
    mock_redis.get.return_value = None
    
    # Mock set
    mock_redis.set.return_value = True
    
    # Mock delete
    mock_redis.delete.return_value = 1
    
    # Mock flushdb
    mock_redis.flushdb.return_value = True
    
    return mock_redis


@pytest.fixture(scope="function")
def client(mock_qdrant_client, mock_redis_client):
    """Create FastAPI test client with mocked dependencies"""

    # This fixture needs the real app. If it failed to import at collection time,
    # re-raise the original error (chained) so the actual traceback is visible,
    # instead of letting TestClient(None) blow up with an opaque NoneType error.
    if app is None:
        raise RuntimeError(
            "The 'client' fixture requires app.main.app, which failed to import. "
            "See the chained traceback below for the real cause."
        ) from _app_import_error

    # Create a mock redis_cache instance
    mock_redis_cache_instance = Mock()
    mock_redis_cache_instance.redis_client = mock_redis_client
    mock_redis_cache_instance.cache_enabled = False
    mock_redis_cache_instance.get = Mock(return_value=None)
    mock_redis_cache_instance.set = Mock(return_value=None)
    mock_redis_cache_instance.clear = AsyncMock(return_value=None)
    
    with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant_client), \
         patch("app.core.clients.redis_cache.redis_cache", mock_redis_cache_instance), \
         patch("app.core.clients.qdrant.ensure_collections_exist", AsyncMock(return_value=None)):
        
        test_client = TestClient(app)
        yield test_client


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing"""
    # Create a simple text file that mimics a PDF upload
    content = b"Sample PDF content for testing"
    file = io.BytesIO(content)
    file.name = "test_document.pdf"
    return file


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing"""
    content = b"Sample text content for testing"
    file = io.BytesIO(content)
    file.name = "test_document.txt"
    return file


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Sample metadata for document testing"""
    return {
        "type": "pdf",
        "author": "Test Author",
        "company": "Test Company",
        "KEY ENTITIES": "Test Entity"
    }


@pytest.fixture
def sample_tags() -> List[str]:
    """Sample tags for document testing"""
    return ["test", "sample", "document"]


@pytest.fixture
def mock_document_processor():
    """Mock DocumentProcessor for testing"""
    with patch("app.api.v1.endpoints.documents.document_processor") as mock_processor:
        # Mock process_upload method (it's async in the actual code via process_and_store)
        mock_processor.process_upload = AsyncMock(return_value={
            "message": "Documents processed successfully",
            "source_id": "test-source-1",
            "chunks_created": 5,
            "priority": "P1"
        })
        
        # Mock update_documents method
        mock_processor.update_documents = AsyncMock(return_value={
            "message": "Documents updated successfully",
            "source_id": "test-source-1",
            "chunks_created": 5
        })
        
        # Mock upsert_documents method
        mock_processor.upsert_documents = AsyncMock(return_value={
            "message": "Documents upserted successfully",
            "source_id": "test-source-1",
            "chunks_created": 5,
            "operation": "updated"
        })
        
        # Mock update_metadata method
        mock_processor.update_metadata = AsyncMock(return_value={
            "message": "Metadata updated successfully",
            "source_id": "test-source-1",
            "updated_count": 5
        })
        
        # Mock delete_documents method
        mock_processor.delete_documents = AsyncMock(return_value={
            "message": "Documents deleted successfully",
            "source_id": "test-source-1",
            "deleted_count": 5
        })
        
        yield mock_processor


@pytest.fixture
def mock_similarity_service():
    """Mock SimilarityService for testing"""
    with patch("app.api.v1.endpoints.documents.similarity_service") as mock_service:
        mock_service.check_similarity = Mock(return_value={
            "has_similar": False,
            "similar_documents": []
        })
        yield mock_service


@pytest.fixture
def mock_prioritized_search_service():
    """Mock PrioritizedSearchService for testing"""
    with patch("app.api.v1.endpoints.documents.prioritized_search_service") as mock_service:
        def search_side_effect(request):
            return {
                "query": request.query,
                "total_results": 1,
                "top_k": request.top_k,
                "results": [
                    {
                        "id": "test-id-1",
                        "text": "Sample text",
                        "title": "Test Title",
                        "summary": "Test summary",
                        "tags": ["tag1"],
                        "metadata": {"type": "pdf"},
                        "source_id": "test-source-1",
                        "score": 0.95,
                        "field_scores": {"title": 0.95}
                    }
                ],
                "search_config": {}
            }
        mock_service.search = Mock(side_effect=search_side_effect)
        yield mock_service


@pytest.fixture
def mock_text_embedding_search_service():
    """Mock TextEmbeddingSearchService for testing"""
    with patch("app.api.v1.endpoints.documents.text_embedding_search_service") as mock_service:
        def search_side_effect(request):
            return {
                "query": request.query,
                "total_results": 1 if request.threshold < 0.95 else 0,
                "results": [
                    {
                        "source_id": "test-source-1",
                        "text": "Sample text",
                        "score": 0.85,
                        "metadata": {"type": "pdf"}
                    }
                ] if request.threshold < 0.95 else []
            }
        mock_service.search = Mock(side_effect=search_side_effect)
        yield mock_service


@pytest.fixture
def mock_query_service():
    """Mock QueryService for testing"""
    with patch("app.api.v1.endpoints.query.query_service") as mock_service:
        def process_query_side_effect(request):
            return {
                "relevant_texts": [
                    {
                        "text": "Sample relevant text",
                        "metadata": {"source": "test"},
                        "score": 0.9
                    }
                ],
                "original_query": request.query,
                "translated_query": None,
                "language": "en"
            }
        mock_service.process_query = Mock(side_effect=process_query_side_effect)
        yield mock_service


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache for testing"""
    with patch("app.api.v1.endpoints.cache.redis_cache") as mock_cache:
        mock_cache.clear = AsyncMock(return_value=None)
        yield mock_cache
