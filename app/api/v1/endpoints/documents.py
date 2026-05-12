from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Optional, Dict, Any, List
import json
import logging
from app.services.document_processor import DocumentProcessor
from app.services.similarity_service import SimilarityService
from app.services.prioritized_search_service import PrioritizedSearchService
from app.services.text_embedding_search_service import TextEmbeddingSearchService
from app.services.source_verification_service import source_verification_service
from app.models.api_models import (
    DeleteRequest, 
    SimilarityCheckRequest, 
    SimilarityCheckResponse,
    PrioritizedSearchRequest,
    PrioritizedSearchResponse,
    TextSearchRequest,
    TextSearchResponse,
    SourceVerificationRequest,
    SourceVerificationResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)
document_processor = DocumentProcessor()
similarity_service = SimilarityService()
prioritized_search_service = PrioritizedSearchService()
text_embedding_search_service = TextEmbeddingSearchService()


def parse_metadata_form(metadata: Optional[str] = Form(default=None)) -> Optional[Dict[str, Any]]:
    """Parse metadata from JSON string to dict"""
    if not metadata or not metadata.strip():
        return None
    try:
        parsed = json.loads(metadata)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="Metadata must be a JSON object/dict")
        return parsed
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {str(e)}")


def parse_tags_form(tags: Optional[str] = Form(default=None)) -> Optional[List[str]]:
    """Parse tags from JSON array or comma-separated string to list"""
    if not tags or not tags.strip():
        return None
    
    tags = tags.strip()
    
    # Try parsing as JSON array first
    if tags.startswith('['):
        try:
            parsed = json.loads(tags)
            if not isinstance(parsed, list):
                raise HTTPException(status_code=400, detail="Tags must be a JSON array/list")
            return parsed
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid tags JSON: {str(e)}")
    
    # Otherwise, treat as comma-separated string
    return [tag.strip() for tag in tags.split(',') if tag.strip()]


@router.post("/documents", status_code=201)
async def create_documents(
        file: UploadFile = File(...),
        priority: str = Form(default="P1"),
        source_id: Optional[str] = Form(default=None),
        company_id: Optional[str] = Form(default=None),
        title: Optional[str] = Form(default=None),
        summary: Optional[str] = Form(default=None),
        metadata: Optional[Dict[str, Any]] = Depends(parse_metadata_form),
        tags: Optional[List[str]] = Depends(parse_tags_form)
):
    """Create new documents by uploading and processing a file
    
    Args:
        file: File to upload
        priority: Priority level (P1, P2, P3, etc.)
        source_id: Unique identifier for the source
        company_id: Company identifier
        title: Document title
        summary: Document summary
        metadata: Additional metadata as dict/object (sent as JSON string in form)
                 Example: {"author": "John", "department": "Finance"}
        tags: Tags as list/array or comma-separated string
             JSON format: ["tag1", "tag2", "tag3"]
             CSV format: "tag1, tag2, tag3"
    
    Note: 
    - metadata must be a JSON dict/object: {"key": "value"}
    - tags can be either JSON array ["tag1", "tag2"] OR comma-separated "tag1,tag2"
    - Leave metadata/tags empty or don't send them if not needed
    """
    return await document_processor.process_upload(
        file, priority, metadata, source_id, company_id, title, summary, tags
    )


@router.put("/documents/{source_id}")
async def update_documents(
        source_id: str,
        file: UploadFile = File(...),
        priority: str = Form(default="P1"),
        metadata: Optional[str] = Form(default=None),
        company_id: Optional[str] = Form(default=None)
):
    """Update existing documents by replacing all documents with the same source_id and company_id"""
    return await document_processor.update_documents(file, priority, metadata, source_id, company_id)


@router.put("/documents/{source_id}/upsert")
async def upsert_documents(
        source_id: str,
        file: UploadFile = File(...),
        priority: str = Form(default="P1"),
        metadata: Optional[str] = Form(default=None),
        company_id: Optional[str] = Form(default=None)
):
    """Upsert documents - update if exists, create if not"""
    return await document_processor.upsert_documents(file, priority, metadata, source_id, company_id)


@router.patch("/documents/{source_id}/metadata")
async def update_document_metadata(
        source_id: str,
        metadata_updates: str = Form(...),
        company_id: Optional[str] = Form(default=None)
):
    """Update only metadata of documents without reprocessing content"""
    try:
        metadata_dict = json.loads(metadata_updates)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    return await document_processor.update_metadata(source_id, metadata_dict, company_id)


@router.delete("/documents/{source_id}")
async def delete_documents(
        source_id: str,
        company_id: Optional[str] = Form(default=None)
):
    """Delete all documents by source ID and optional company ID"""
    request = DeleteRequest(source_id=source_id, company_id=company_id)
    return await document_processor.delete_documents(request)


@router.post("/documents/check-similarity")
async def check_similarity(request: SimilarityCheckRequest) -> SimilarityCheckResponse:
    """Check if similar content already exists"""
    return similarity_service.check_similarity(request)


@router.post("/documents/search", response_model=PrioritizedSearchResponse)
async def prioritized_search(request: PrioritizedSearchRequest) -> PrioritizedSearchResponse:
    """
    Perform prioritized multi-field search across documents with optional filters.
    
    **Query Modes:**
    1. **With Query**: Performs semantic search across all fields with scoring
    2. **Without Query**: Returns all documents with unique source_id (one document per source)
    
    **Filtering Modes (Simplified):**
    
    **Mode 1: Detail Filter Score (Field-Level Thresholds)**
    - **When to use**: Provide `detail_filter_score` (NOT null)
    - **How it works**: Each field (title, text, tags, summary, metadata) has its own threshold
    - **Pass condition**: Document passes if ANY field score meets its threshold (OR logic)
    - **Use case**: Want granular control where documents strong in any single field qualify
    - **Note**: `filter_score` is ignored when `detail_filter_score` is provided
    
    **Mode 2: Filter Score (Weighted Score Threshold)**
    - **When to use**: Set `detail_filter_score = null` (or omit it)
    - **How it works**: Uses final weighted score as threshold
    - **Pass condition**: Document passes if weighted_score >= filter_score
    - **Use case**: Want overall quality threshold across all fields combined
    
    Scoring Formula (when query is provided):
    S_Final = (W_Title × S_Title) + (W_Text × S_Text) + (W_Tags × S_Tags) + 
              (W_Summary × S_Summary) + (W_Metadata × S_Metadata)
    
    Where:
    - W_* are the weights from configuration (sum to 1.0)
    - S_* are the similarity scores from each field (0-1 range)
    
    Search Priority Order (configured in settings):
    1. **Title** (36% weight) - Highest priority for document title matches
    2. **Text/Chunks** (27% weight) - High priority for content matches
    3. **Tags** (14% weight) - Medium priority for tag-based categorization  
    4. **Summary** (14% weight) - Medium priority for summary matches
    5. **Metadata** (9% weight) - Base priority for metadata field matches
    
    Features:
    - **Multi-field scoring**: Combines scores from all fields with configured weights
    - **Dual filtering modes**: Choose between field-level or weighted score filtering
    - **Optional filters**: Filter by categories, organizations, resource types, and file types
    - **Top-K results**: Get top N recommendations
    - **Unique source results**: Returns only the highest scoring document per source_id
    - **Unique source mode**: When query is empty, returns one document per unique source_id
    
    Args:
        request: PrioritizedSearchRequest containing:
            - query: Search query text (optional - if not provided, returns unique source_id documents)
            - top_k: Number of top results to return (default: 1000000)
            - filter_score: Minimum weighted score threshold (0-1), used when detail_filter_score is None
            - detail_filter_score: Field-level thresholds (title, text, tags, summary, metadata)
                                   Applied when filter_score=0, uses OR logic
            - categories: Optional list of tag values to filter (searches in 'tags' field, OR condition)
            - organizations: Optional list of company names to filter (searches in 'metadata.company' field, OR condition)
            - resource_type: Optional list of key entities to filter (searches in 'metadata.KEY ENTITIES' field, OR condition)
            - file_type: Optional list of document types to filter (searches in 'metadata.type' field, OR condition)
    
    Filter Field Mappings:
    - categories → 'tags' field (list of tags)
    - organizations → 'metadata.company' field
    - resource_type → 'metadata.KEY ENTITIES' field
    - file_type → 'metadata.type' field
    
    Filter Logic:
    - All filter types and values use OR logic
    - Example: categories=["Entrepreneurship"] and organizations=["shikshaloakamstaging"]
      means: (tags contains "Entrepreneurship" OR metadata.company="shikshaloakamstaging")
    - Filters are combined with the query using AND: filters OR filters... AND query
    
    Returns:
        PrioritizedSearchResponse with ranked results and scoring details
        
    Example Request (Mode 1 - Detail Filter Score):
        ```json
        {
            "query": "machine learning algorithms",
            "top_k": 10,
            "detail_filter_score": {
                "title": 0.36,
                "text": 0.27,
                "tags": 0.14,
                "summary": 0.14,
                "metadata": 0.09
            }
        }
        ```
        Note: filter_score is ignored when detail_filter_score is provided
    
    Example Request (Mode 2 - Filter Score):
        ```json
        {
            "query": "machine learning algorithms",
            "top_k": 10,
            "filter_score": 0.5
        }
        ```
    
    Example Request (Get All Unique Sources):
        ```json
        {
            "top_k": 50
        }
        ```
    
    Example Request (With Category Filters):
        ```json
        {
            "query": "entrepreneurship insights",
            "top_k": 20,
            "filter_score": 0.3,
            "categories": ["Entrepreneurship"],
            "organizations": ["shikshaloakamstaging"],
            "resource_type": ["Peter Thiel", "Stanford"],
            "file_type": ["pdf"]
        }
        ```
    
    Error Handling:
    - Returns 422 if top_k is less than or equal to 0
    - Returns 500 if embedding generation fails (when query is provided)
    - Returns empty results if no documents match filters
    
    Note: Search configuration (priority order and weights) is set in app/config.py
    """
    return prioritized_search_service.search(request)


@router.post("/documents/text-search", response_model=TextSearchResponse)
async def text_embedding_search(request: TextSearchRequest) -> TextSearchResponse:
    """
    Simple text embedding search that returns top chunk per unique document.
    
    This endpoint performs a straightforward vector similarity search using only
    the text embeddings stored in Qdrant. It returns the highest scoring chunk
    for each unique source_id.
    
    **How it works:**
    1. Generates embedding for the query text
    2. Searches Qdrant using the 'text' vector field only
    3. Groups results by source_id
    4. Returns the top scoring chunk for each unique document
    
    Args:
        request: TextSearchRequest containing:
            - query: Search query text (required)
            - top_k: Number of unique documents to return (default: 5)
            - threshold: Minimum similarity score threshold (default: 0.40)
    
    Returns:
        TextSearchResponse with:
            - query: The original search query
            - total_results: Number of unique documents returned
            - results: List of top chunks, one per unique source_id
    
    Example Request:
        ```json
        {
            "query": "machine learning algorithms",
            "top_k": 5,
            "threshold": 0.40
        }
        ```
    
    Example Response:
        ```json
        {
            "query": "machine learning algorithms",
            "total_results": 5,
            "results": [
                {
                    "source_id": "doc_123",
                    "text": "Machine learning algorithms are...",
                    "score": 0.89,
                    "metadata": {"title": "ML Guide", "type": "pdf"}
                },
                ...
            ]
        }
        ```
    
    Note: This is a simpler alternative to /documents/search which only uses
    text embeddings and returns one chunk per document.
    """
    return text_embedding_search_service.search(request)


@router.post("/documents/verify-sources", response_model=SourceVerificationResponse)
async def verify_sources(request: SourceVerificationRequest) -> SourceVerificationResponse:
    """
    Verify which source IDs exist in Qdrant and return categorized lists.
    
    Checks a list of source IDs against the Qdrant database and returns which ones were found and not found.
    """
    return source_verification_service.verify_sources(request)

