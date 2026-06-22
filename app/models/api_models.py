from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.config import settings

class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    row: Optional[int] = None

class DocumentUploadRequest(BaseModel):
    priority: str

class QueryRequest(BaseModel):
    query: str
    search_limit: Optional[int] = settings.VECTOR_SEARCH_LIMIT
    force_llm: Optional[bool] = False

class DeleteRequest(BaseModel):
    source_id: str = Field(..., description="Unique source identifier to delete documents")
    company_id: Optional[str] = Field(None, description="Company ID to filter deletion")

class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float

class MultilingualQueryResponse(BaseModel):
    relevant_texts: List[Dict[str, Any]]
    original_query: str
    translated_query: Optional[str] = None
    language: str

class MultilingualQueryRequest(BaseModel):
    query: str
    search_limit: int = settings.VECTOR_SEARCH_LIMIT
    priority_filter: Optional[str] = None

class SimilarityCheckRequest(BaseModel):
    text: str = Field(..., description="Text to check for similarity")
    company_id: str = Field(..., description="Company ID to filter by")
    threshold: float = Field(default=0.85, description="Similarity threshold")
    exclude_source_id: Optional[str] = Field(None, description="Source ID to exclude")

class SimilarityCheckResponse(BaseModel):
    has_similar: bool
    similar_documents: List[Dict[str, Any]]

class DetailFilterScore(BaseModel):
    """Field-level threshold configuration for granular filtering"""
    title: float = Field(default=0.36, ge=0.0, le=1.0, description="Minimum score threshold for title field")
    text: float = Field(default=0.27, ge=0.0, le=1.0, description="Minimum score threshold for text field")
    tags: float = Field(default=0.14, ge=0.0, le=1.0, description="Minimum score threshold for tags field")
    summary: float = Field(default=0.14, ge=0.0, le=1.0, description="Minimum score threshold for summary field")
    metadata: float = Field(default=0.09, ge=0.0, le=1.0, description="Minimum score threshold for metadata field")

class PrioritizedSearchRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Search query text (optional - if not provided, returns all documents with unique source_id)")
    top_k: int = Field(default=1000000, description="Number of top results to return")
    filter_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Minimum weighted score threshold (0–1). "
                    "Results below this score will be filtered out. "
                    "Used when detail_filter_score is None."
    )
    detail_filter_score: Optional[DetailFilterScore] = Field(
        default=None,
        description="Field-level score thresholds. When filter_score=0, this enables granular filtering. "
                    "A document passes if ANY field meets its threshold (OR logic)."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tag values to filter (searches in 'tags' field, OR condition)"
    )
    organizations: Optional[List[str]] = Field(
        default=None,
        description="Optional list of company names to filter (searches in 'metadata.company' field, OR condition)"
    )
    resource_type: Optional[List[str]] = Field(
        default=None,
        description="Optional list of key entities to filter (searches in 'metadata.KEY ENTITIES' field, OR condition)"
    )
    file_type: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document types to filter (searches in 'metadata.type' field, OR condition)"
    )
    search_mode: str = Field(
        default="hybrid",
        description="Search mode: 'hybrid' (semantic + title boost), 'semantic' (vector only), "
                    "or 'keyword' (title/keyword match only). Defaults to 'hybrid'."
    )

class SearchResultItem(BaseModel):
    id: str
    text: str
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Dict[str, Any]
    source_id: str
    score: float
    field_scores: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description=(
            "Per-field cosine similarity scores for semantically retrieved documents "
            "(keys: title, tags, summary, metadata, text, values 0–1). "
            "Internal fusion keys (rrf, bm25_sparse) are excluded. "
            "Keyword-injected documents that bypassed vector scoring carry None "
            "for unscored fields to distinguish them from a genuine zero-score."
        )
    )
    keyword_score: Optional[float] = Field(
        default=None,
        description="BM25 sparse vector score (Phase 2 only; None when sparse search is disabled)"
    )
    title_match: Optional[str] = Field(
        default=None,
        description="Title match type: 'exact', 'partial', or None if no title match"
    )
    summary_match: Optional[str] = Field(
        default=None,
        description="Summary match type: 'exact', 'partial', or None if no summary match"
    )

class PrioritizedSearchResponse(BaseModel):
    query: Optional[str] = None
    total_results: int
    top_k: int
    results: List[SearchResultItem]
    search_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration used for this search"
    )

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, description="Number of unique documents to return (default: 5)")
    threshold: float = Field(default=0.40, description="Minimum similarity score threshold (default: 0.40)")

class TextSearchResultItem(BaseModel):
    source_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class TextSearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[TextSearchResultItem]

class SourceVerificationRequest(BaseModel):
    source_ids: List[str] = Field(..., description="List of source IDs to verify")

class SourceVerificationResponse(BaseModel):
    total_requested: int = Field(..., description="Total number of source IDs requested")
    found: List[str] = Field(..., description="List of source IDs that exist in Qdrant")
    not_found: List[str] = Field(..., description="List of source IDs that don't exist in Qdrant")
    found_count: int = Field(..., description="Count of found source IDs")
    not_found_count: int = Field(..., description="Count of not found source IDs")
