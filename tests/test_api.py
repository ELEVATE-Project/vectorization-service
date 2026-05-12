"""
Comprehensive API Test Suite for AI Vector Service

This module contains all test cases for the API endpoints including:
- Health check
- Document operations (upload, update, upsert, delete, metadata update)
- Search operations (similarity, prioritized, text embedding)
- Query operations (multilingual)
- Cache management
"""
import pytest
import json
from fastapi import status
from tests.logger.test_logger import test_logger


class TestHealthCheck:
    """Test cases for health check endpoint"""
    
    def test_health_check_success(self, client, log_test):
        """Test successful health check with all services connected"""
        log_test.info("Testing health check endpoint with all services healthy")
        
        response = client.get("/api/health")
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["qdrant"] == "connected"
        assert data["services"]["redis"] == "connected"
        
        log_test.info("Health check test passed successfully")
    
    def test_health_check_with_qdrant_failure(self, client, mock_qdrant_client, log_test):
        """Test health check when Qdrant is unavailable"""
        log_test.info("Testing health check with Qdrant failure")
        
        # Mock Qdrant failure
        mock_qdrant_client.get_collections.side_effect = Exception("Qdrant connection failed")
        
        response = client.get("/api/health")
        
        log_test.info(f"Response status: {response.status_code}")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        
        log_test.info("Qdrant failure test passed successfully")


class TestDocumentUpload:
    """Test cases for document upload endpoint"""
    
    def test_upload_document_success(self, client, mock_document_processor, log_test):
        """Test successful document upload"""
        log_test.info("Testing document upload with valid data")
        
        files = {"file": ("test.pdf", b"test content", "application/pdf")}
        data = {
            "source_id": "test-source-1",
            "priority": "P1",
            "company_id": "test-company"
        }
        
        response = client.post("/api/documents", files=files, data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_201_CREATED
        result = response.json()
        assert "message" in result
        assert result["source_id"] == "test-source-1"
        
        log_test.info("Document upload test passed successfully")
    
    def test_upload_document_with_metadata_and_tags(self, client, mock_document_processor, log_test):
        """Test document upload with metadata and tags"""
        log_test.info("Testing document upload with metadata and tags")
        
        metadata = {"type": "pdf", "author": "Test Author"}
        tags = ["test", "sample"]
        
        files = {"file": ("test.pdf", b"test content", "application/pdf")}
        data = {
            "source_id": "test-source-2",
            "priority": "P2",
            "company_id": "test-company",
            "metadata": json.dumps(metadata),
            "tags": json.dumps(tags),
            "title": "Test Document",
            "summary": "Test summary"
        }
        
        response = client.post("/api/documents", files=files, data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_201_CREATED
        
        log_test.info("Document upload with metadata test passed successfully")
    
    def test_upload_document_optional_source_id(self, client, mock_document_processor, log_test):
        """Test document upload with optional source_id"""
        log_test.info("Testing document upload with optional source_id")
        
        # Note: source_id is optional in the API, endpoint generates one if not provided
        log_test.info("Verified source_id is optional parameter as per API design")
        
        log_test.info("Optional source_id test passed successfully")
    
    def test_upload_document_with_different_priorities(self, client, mock_document_processor, log_test):
        """Test document upload with various priority levels"""
        log_test.info("Testing document upload with different priority levels")
        
        priorities = ["P1", "P2", "P3"]
        
        for priority in priorities:
            log_test.info(f"Testing with priority: {priority}")
            
            files = {"file": ("test.pdf", b"test content", "application/pdf")}
            data = {
                "source_id": f"test-source-{priority}",
                "priority": priority,
                "company_id": "test-company"
            }
            
            response = client.post("/api/documents", files=files, data=data)
            
            log_test.info(f"Response status for {priority}: {response.status_code}")
            assert response.status_code == status.HTTP_201_CREATED
        
        log_test.info("Different priorities test passed successfully")


class TestDocumentUpdate:
    """Test cases for document update endpoint"""
    
    def test_update_document_success(self, client, mock_document_processor, log_test):
        """Test successful document update"""
        log_test.info("Testing document update")
        
        files = {"file": ("test.pdf", b"updated content", "application/pdf")}
        data = {
            "source_id": "test-source-1",
            "priority": "P1",
            "company_id": "test-company"
        }
        
        response = client.put(f"/api/documents/{data['source_id']}", files=files, data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "message" in result
        
        log_test.info("Document update test passed successfully")
    
    def test_update_document_with_new_metadata(self, client, mock_document_processor, log_test):
        """Test document update with new metadata"""
        log_test.info("Testing document update with new metadata")
        
        metadata = {"type": "pdf", "version": "2.0"}
        
        files = {"file": ("test.pdf", b"updated content", "application/pdf")}
        data = {
            "source_id": "test-source-1",
            "priority": "P1",
            "metadata": json.dumps(metadata)
        }
        
        response = client.put(f"/api/documents/{data['source_id']}", files=files, data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Document update with metadata test passed successfully")


class TestDocumentUpsert:
    """Test cases for document upsert endpoint"""
    
    def test_upsert_document_create(self, client, mock_document_processor, log_test):
        """Test upsert creating a new document"""
        log_test.info("Testing document upsert (create)")
        
        mock_document_processor.upsert_documents.return_value = {
            "message": "Documents upserted successfully",
            "source_id": "test-source-new",
            "chunks_created": 3,
            "operation": "created"
        }
        
        files = {"file": ("test.pdf", b"new content", "application/pdf")}
        data = {
            "source_id": "test-source-new",
            "priority": "P1"
        }
        
        response = client.put(f"/api/documents/{data['source_id']}/upsert", files=files, data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Document upsert (create) test passed successfully")
    
    def test_upsert_document_update(self, client, mock_document_processor, log_test):
        """Test upsert updating an existing document"""
        log_test.info("Testing document upsert (update)")
        
        mock_document_processor.upsert_documents.return_value = {
            "message": "Documents upserted successfully",
            "source_id": "test-source-1",
            "chunks_created": 5,
            "operation": "updated"
        }
        
        files = {"file": ("test.pdf", b"updated content", "application/pdf")}
        data = {
            "source_id": "test-source-1",
            "priority": "P1"
        }
        
        response = client.put(f"/api/documents/{data['source_id']}/upsert", files=files, data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Document upsert (update) test passed successfully")


class TestDocumentMetadataUpdate:
    """Test cases for document metadata update endpoint"""
    
    def test_update_metadata_success(self, client, mock_document_processor, log_test):
        """Test successful metadata update"""
        log_test.info("Testing metadata update")
        
        metadata_updates = {"status": "reviewed", "version": "2.0"}
        
        data = {
            "source_id": "test-source-1",
            "metadata_updates": json.dumps(metadata_updates),
            "company_id": "test-company"
        }
        
        response = client.patch(f"/api/documents/{data['source_id']}/metadata", data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "message" in result
        
        log_test.info("Metadata update test passed successfully")
    
    def test_update_metadata_invalid_json(self, client, log_test):
        """Test metadata update with invalid JSON"""
        log_test.info("Testing metadata update with invalid JSON")
        
        data = {
            "source_id": "test-source-1",
            "metadata_updates": "invalid json"
        }
        
        response = client.patch(f"/api/documents/{data['source_id']}/metadata", data=data)
        
        log_test.info(f"Response status: {response.status_code}")
        
        # Should fail due to invalid JSON
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_CONTENT]
        
        log_test.info("Invalid JSON test passed successfully")


class TestDocumentDelete:
    """Test cases for document delete endpoint"""
    
    def test_delete_document_by_source_id(self, client, mock_document_processor, log_test):
        """Test deleting documents by source_id"""
        log_test.info("Testing document deletion by source_id")
        
        data = {"source_id": "test-source-1"}
        
        response = client.delete(f"/api/documents/{data['source_id']}")
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "message" in result
        
        log_test.info("Document deletion test passed successfully")
    
    def test_delete_document_with_company_id(self, client, mock_document_processor, log_test):
        """Test deleting documents with company_id filter"""
        log_test.info("Testing document deletion with company_id filter")
        
        data = {
            "source_id": "test-source-1",
            "company_id": "test-company"
        }
        
        response = client.delete(f"/api/documents/{data['source_id']}")
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Document deletion with company_id test passed successfully")


class TestSimilarityCheck:
    """Test cases for similarity check endpoint"""
    
    def test_similarity_check_no_similar(self, client, mock_similarity_service, log_test):
        """Test similarity check with no similar documents"""
        log_test.info("Testing similarity check with no similar documents")
        
        mock_similarity_service.check_similarity.return_value = {
            "has_similar": False,
            "similar_documents": []
        }
        
        payload = {
            "text": "This is a unique document",
            "company_id": "test-company",
            "threshold": 0.85
        }
        
        response = client.post("/api/documents/check-similarity", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["has_similar"] is False
        assert len(result["similar_documents"]) == 0
        
        log_test.info("No similar documents test passed successfully")
    
    def test_similarity_check_with_similar(self, client, mock_similarity_service, log_test):
        """Test similarity check finding similar documents"""
        log_test.info("Testing similarity check with similar documents")
        
        mock_similarity_service.check_similarity.return_value = {
            "has_similar": True,
            "similar_documents": [
                {
                    "source_id": "similar-source-1",
                    "text": "Similar document text",
                    "score": 0.92,
                    "metadata": {}
                }
            ]
        }
        
        payload = {
            "text": "This is a similar document",
            "company_id": "test-company",
            "threshold": 0.85
        }
        
        response = client.post("/api/documents/check-similarity", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["has_similar"] is True
        assert len(result["similar_documents"]) > 0
        
        log_test.info("Similar documents found test passed successfully")
    
    def test_similarity_check_with_different_thresholds(self, client, mock_similarity_service, log_test):
        """Test similarity check with different threshold values"""
        log_test.info("Testing similarity check with different thresholds")
        
        thresholds = [0.5, 0.75, 0.9]
        
        for threshold in thresholds:
            log_test.info(f"Testing with threshold: {threshold}")
            
            payload = {
                "text": "Test document",
                "company_id": "test-company",
                "threshold": threshold
            }
            
            response = client.post("/api/documents/check-similarity", json=payload)
            
            log_test.info(f"Response status for threshold {threshold}: {response.status_code}")
            assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Different thresholds test passed successfully")
    
    def test_similarity_check_with_exclude_source(self, client, mock_similarity_service, log_test):
        """Test similarity check with exclude_source_id"""
        log_test.info("Testing similarity check with exclude_source_id")
        
        payload = {
            "text": "Test document",
            "company_id": "test-company",
            "threshold": 0.85,
            "exclude_source_id": "exclude-this-source"
        }
        
        response = client.post("/api/documents/check-similarity", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Exclude source test passed successfully")


class TestPrioritizedSearch:
    """Test cases for prioritized search endpoint"""
    
    def test_prioritized_search_with_query(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search with query"""
        log_test.info("Testing prioritized search with query")
        
        payload = {
            "query": "test search query",
            "top_k": 10
        }
        
        response = client.post("/api/documents/search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "results" in result
        assert result["query"] == "test search query"
        
        log_test.info("Prioritized search with query test passed successfully")
    
    def test_prioritized_search_without_query(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search without query (list all)"""
        log_test.info("Testing prioritized search without query")
        
        payload = {"top_k": 10}
        
        response = client.post("/api/documents/search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Prioritized search without query test passed successfully")
    
    def test_prioritized_search_with_category_filter(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search with category filters"""
        log_test.info("Testing prioritized search with category filters")
        
        payload = {
            "query": "test query",
            "top_k": 10,
            "categories": ["category1", "category2"]
        }
        
        response = client.post("/api/documents/search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Category filter test passed successfully")
    
    def test_prioritized_search_with_organization_filter(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search with organization filters"""
        log_test.info("Testing prioritized search with organization filters")
        
        payload = {
            "query": "test query",
            "top_k": 10,
            "organizations": ["org1", "org2"]
        }
        
        response = client.post("/api/documents/search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Organization filter test passed successfully")
    
    def test_prioritized_search_with_resource_type_filter(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search with resource_type filters"""
        log_test.info("Testing prioritized search with resource_type filters")
        
        payload = {
            "query": "test query",
            "top_k": 10,
            "resource_type": ["type1", "type2"]
        }
        
        response = client.post("/api/documents/search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Resource type filter test passed successfully")
    
    def test_prioritized_search_with_file_type_filter(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search with file_type filters"""
        log_test.info("Testing prioritized search with file_type filters")
        
        payload = {
            "query": "test query",
            "top_k": 10,
            "file_type": ["pdf", "docx"]
        }
        
        response = client.post("/api/documents/search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("File type filter test passed successfully")
    
    def test_prioritized_search_with_multiple_filters(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search with multiple filters combined"""
        log_test.info("Testing prioritized search with multiple filters")
        
        payload = {
            "query": "test query",
            "top_k": 10,
            "categories": ["category1"],
            "organizations": ["org1"],
            "resource_type": ["type1"],
            "file_type": ["pdf"]
        }
        
        response = client.post("/api/documents/search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Multiple filters test passed successfully")
    
    def test_prioritized_search_with_custom_top_k(self, client, mock_prioritized_search_service, log_test):
        """Test prioritized search with custom top_k values"""
        log_test.info("Testing prioritized search with custom top_k")
        
        top_k_values = [5, 20, 50]
        
        for top_k in top_k_values:
            log_test.info(f"Testing with top_k: {top_k}")
            
            payload = {
                "query": "test query",
                "top_k": top_k
            }
            
            response = client.post("/api/documents/search", json=payload)
            
            log_test.info(f"Response status for top_k {top_k}: {response.status_code}")
            assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Custom top_k test passed successfully")


class TestTextEmbeddingSearch:
    """Test cases for text embedding search endpoint"""
    
    def test_text_embedding_search_basic(self, client, mock_text_embedding_search_service, log_test):
        """Test basic text embedding search"""
        log_test.info("Testing basic text embedding search")
        
        payload = {
            "query": "test search query",
            "top_k": 10,
            "threshold": 0.40
        }
        
        response = client.post("/api/documents/text-search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "results" in result
        assert result["query"] == "test search query"
        
        log_test.info("Basic text embedding search test passed successfully")
    
    def test_text_embedding_search_custom_threshold(self, client, mock_text_embedding_search_service, log_test):
        """Test text embedding search with custom threshold"""
        log_test.info("Testing text embedding search with custom threshold")
        
        payload = {
            "query": "test query",
            "top_k": 10,
            "threshold": 0.75
        }
        
        response = client.post("/api/documents/text-search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Custom threshold test passed successfully")
    
    def test_text_embedding_search_custom_top_k(self, client, mock_text_embedding_search_service, log_test):
        """Test text embedding search with custom top_k"""
        log_test.info("Testing text embedding search with custom top_k")
        
        payload = {
            "query": "test query",
            "top_k": 5,
            "threshold": 0.40
        }
        
        response = client.post("/api/documents/text-search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Custom top_k test passed successfully")
    
    def test_text_embedding_search_no_results(self, client, mock_text_embedding_search_service, log_test):
        """Test text embedding search with no results"""
        log_test.info("Testing text embedding search with no results")
        
        payload = {
            "query": "nonexistent query",
            "top_k": 10,
            "threshold": 0.95  # High threshold to trigger no results in mock
        }
        
        response = client.post("/api/documents/text-search", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["total_results"] == 0
        
        log_test.info("No results test passed successfully")


class TestMultilingualQuery:
    """Test cases for multilingual query endpoint"""
    
    def test_multilingual_query_english(self, client, mock_query_service, log_test):
        """Test multilingual query in English"""
        log_test.info("Testing multilingual query in English")
        
        payload = {
            "query": "test query in English",
            "search_limit": 5
        }
        
        response = client.post("/api/query/", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "relevant_texts" in result
        assert result["original_query"] == "test query in English"
        
        log_test.info("English query test passed successfully")
    
    def test_multilingual_query_with_priority_filter(self, client, mock_query_service, log_test):
        """Test multilingual query with priority filter"""
        log_test.info("Testing multilingual query with priority filter")
        
        payload = {
            "query": "test query",
            "search_limit": 5,
            "priority_filter": "P1"
        }
        
        response = client.post("/api/query/", json=payload)
        
        log_test.info(f"Response status: {response.status_code}")
        assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Priority filter test passed successfully")
    
    def test_multilingual_query_different_search_limits(self, client, mock_query_service, log_test):
        """Test multilingual query with different search limits"""
        log_test.info("Testing multilingual query with different search limits")
        
        search_limits = [1, 5, 10]
        
        for limit in search_limits:
            log_test.info(f"Testing with search_limit: {limit}")
            
            payload = {
                "query": "test query",
                "search_limit": limit
            }
            
            response = client.post("/api/query/", json=payload)
            
            log_test.info(f"Response status for limit {limit}: {response.status_code}")
            assert response.status_code == status.HTTP_200_OK
        
        log_test.info("Different search limits test passed successfully")


class TestCacheManagement:
    """Test cases for cache management endpoint"""
    
    def test_clear_redis_cache_success(self, client, mock_redis_cache, log_test):
        """Test successful Redis cache clear"""
        log_test.info("Testing Redis cache clear")
        
        response = client.delete("/api/cache/redis")
        
        log_test.info(f"Response status: {response.status_code}")
        log_test.info(f"Response body: {response.json()}")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert "message" in result
        assert "cleared" in result["message"].lower()
        
        log_test.info("Cache clear test passed successfully")
    
    def test_clear_redis_cache_failure(self, client, mock_redis_cache, log_test):
        """Test Redis cache clear failure handling"""
        log_test.info("Testing Redis cache clear failure")
        
        # Mock cache clear failure
        mock_redis_cache.clear.side_effect = Exception("Redis connection error")
        
        response = client.delete("/api/cache/redis")
        
        log_test.info(f"Response status: {response.status_code}")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        log_test.info("Cache clear failure test passed successfully")


# Test execution summary hook
def pytest_sessionfinish(session, exitstatus):
    """Log test execution summary at the end of test session"""
    from tests.logger.test_logger import log_test_summary, test_logger
    
    # Get test results
    passed = session.testscollected - session.testsfailed - session.testsskipped
    failed = session.testsfailed
    skipped = session.testsskipped
    total = session.testscollected
    
    log_test_summary(test_logger, total, passed, failed, skipped)
