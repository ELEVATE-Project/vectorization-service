# config.py
import math
import os
from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

# Disable tokenizers parallelism warning when using pdf2image for OCR
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Settings(BaseSettings):
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    # QA runs Qdrant 1.12 with client 1.18 (needed for BM25); the version check only warns.
    # All features used work on 1.12. Set true once QA is on 1.18.
    QDRANT_CHECK_COMPATIBILITY: bool = os.getenv("QDRANT_CHECK_COMPATIBILITY", "false").lower() == "true"
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    QA_CACHE_COLLECTION: str = os.getenv("QA_CACHE_COLLECTION", "qa_cache")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    LLAMA_MODEL_ID: str = os.getenv("LLAMA_MODEL_ID", "meta.llama3-70b-instruct-v1:0")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    TRANSLATION_API_URL: str = "https://demo-api.models.ai4bharat.org/inference/translation/v2"
    PDF_CHUNK_SIZE: int = 3000
    PDF_CHUNK_OVERLAP: int = 500
    PAGE_TEXT_THRESHOLD: int = 20  # Minimum characters per page before OCR is triggered
    VECTOR_SEARCH_LIMIT: int = 1
    SIMILARITY_THRESHOLD: float = 0.40

    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 500

    # Markdown-specific settings (larger chunks to fit multiple rows for better RAG context)
    MARKDOWN_CHUNK_SIZE: int = 3500
    MARKDOWN_CHUNK_OVERLAP: int = 800
    MAX_CACHE_RESULTS: int = 1
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    # Commons uses db 0 on the shared Redis, so this service defaults to db 2.
    REDIS_DB: int = int(os.getenv("REDIS_DB", 2))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    # Small on purpose: redis-py has no default timeout, and a blackholed
    # connection (packets silently dropped) would hang cache reads for minutes,
    # stalling search instead of falling back to Postgres.
    REDIS_SOCKET_CONNECT_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", 1))
    REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", 1))
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", 86400))  # 24 hours in seconds
    # Caches "not an acronym" so ordinary words skip Postgres; shorter than REDIS_CACHE_TTL.
    # Uploads refresh their own keys, so this only bounds staleness in rare races.
    REDIS_NEGATIVE_CACHE_TTL: int = int(os.getenv("REDIS_NEGATIVE_CACHE_TTL", 3600))  # 1 hour
    # Shorter than REDIS_NEGATIVE_CACHE_TTL: this caches a DB-error miss, not a
    # genuine "not an acronym" miss. DB outages are often transient — caching
    # "not found" for a full hour would hide real acronyms after Postgres recovers.
    REDIS_DB_ERROR_CACHE_TTL: int = int(os.getenv("REDIS_DB_ERROR_CACHE_TTL", 30))
    REDIS_MAX_CACHE_SIZE: int = int(os.getenv("REDIS_MAX_CACHE_SIZE", 1000))
    DATABASE_URL: str = os.getenv("POSTGRES_DATABASE_URI", "postgresql://anuj:1234@localhost:5432/ai_vector_service")
    # Without a connect timeout, a blackholed Postgres host hangs for minutes.
    # See release-doc/acronym-design-notes.md, Postgres timeouts.
    POSTGRES_CONNECT_TIMEOUT: int = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", 3))
    # Bounds each query on an open connection (a stuck server hung requests forever).
    # All queries here are small indexed lookups, so 5s never fires when healthy.
    POSTGRES_STATEMENT_TIMEOUT_MS: int = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", 5000))
    REDIS_CACHE_ENABLED: bool = False

    # Shared secret for internal-only endpoints (e.g. acronym bulk upload), checked
    # against the X-Internal-Token request header. No default — must be set explicitly.
    INTERNAL_API_TOKEN: str = os.getenv("INTERNAL_API_TOKEN", "")
    # some api endpoints are more sensitive than others (e.g. acronym bulk upload) — require a second, stricter shared secret for those. No default — must be set explicitly.
    ADMIN_API_TOKEN: str = os.getenv("ADMIN_API_TOKEN", "")

    # URL extraction settings
    URL_EXTRACTION_CHUNK_SIZE: int = 1500
    URL_EXTRACTION_CHUNK_OVERLAP: int = 300  # 20% overlap
    URL_REQUEST_TIMEOUT: int = 30  # seconds

    # File upload settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 1024))  # 1GB default (in MB)
    # Acronym bulk-upload CSVs are small tabular text, not documents —
    # a much lower cap than MAX_FILE_SIZE_MB.
    ACRONYM_BULK_UPLOAD_MAX_SIZE_MB: int = int(os.getenv("ACRONYM_BULK_UPLOAD_MAX_SIZE_MB", 5))


    # Prioritized Search Configuration
    # Order determines search priority: Title > Chunk > Tags > Summary > Metadata
    SEARCH_PRIORITY_ORDER: list = ["title", "text", "tags", "summary", "metadata"]
    SEARCH_PRIORITY_WEIGHTS: dict = {
        "title": 0.34,      # 34% weight for title matches
        "text": 0.26,       # 26% weight for chunk/content matches
        "tags": 0.20,       # 20% weight for tag/category matches (up from 14%)
        "summary": 0.12,    # 12% weight for summary matches
        "metadata": 0.08    # 8% weight for metadata matches
    }
    DEFAULT_SEARCH_TOP_K: int = 10
    MAX_SEARCH_TOP_K: int = 100
    MIN_SEARCH_FILTER_SCORE: int = 0
    MIN_WEIGHTED_SCORE_THRESHOLD: float = 0.0  # Minimum weighted score (15%) to include in results

    # Hybrid Search Configuration (Phase 1 — works with qdrant-client<=1.6.9)
    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    EXACT_TITLE_BOOST: float = float(os.getenv("EXACT_TITLE_BOOST", "2.5"))
    PARTIAL_TITLE_BOOST: float = float(os.getenv("PARTIAL_TITLE_BOOST", "1.5"))
    # Summary boosts are lower than title (summary carries less weight than title).
    EXACT_SUMMARY_BOOST: float = float(os.getenv("EXACT_SUMMARY_BOOST", "1.4"))
    PARTIAL_SUMMARY_BOOST: float = float(os.getenv("PARTIAL_SUMMARY_BOOST", "1.2"))
    # Acronym queries only: text boost on top of the title/summary boosts above, when
    # the body uses the acronym as written (capitals).
    EXACT_TEXT_BOOST: float = float(os.getenv("EXACT_TEXT_BOOST", "2.0"))
    METADATA_MATCH_BOOST: float = float(os.getenv("METADATA_MATCH_BOOST", "1.2"))
    # How many keyword-matched documents we are willing to score per search. Scoring
    # each one downloads 5 vectors, and a short query can match thousands of titles.
    # Past this limit we skip scoring them and leave their field_scores as None.
    INJECTED_DOC_SCORING_MAX: int = int(os.getenv("INJECTED_DOC_SCORING_MAX", "200"))
    # Queries shorter than this word count skip spaCy stop-word removal
    SHORT_QUERY_THRESHOLD: int = int(os.getenv("SHORT_QUERY_THRESHOLD", "3"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))  # standard Reciprocal Rank Fusion constant

    # Minimum stem length for prefix-matching expansion words ("institute" ~ "institutes").
    # Not an acronym length limit. See design notes, Settings.
    ACRONYM_MIN_PREFIX_MATCH_LEN: int = int(os.getenv("ACRONYM_MIN_PREFIX_MATCH_LEN", "4"))

    # Max extra characters in a prefix match: "tests" ~ "test", but not "testimony".
    ACRONYM_PREFIX_SUFFIX_CAP: int = int(os.getenv("ACRONYM_PREFIX_SUFFIX_CAP", "3"))

    # Dense variants per query, including the original: one per expansion meaning
    # (SSC, DM, MIP have two). See design notes, Settings.
    ACRONYM_MAX_DENSE_VARIANTS: int = int(os.getenv("ACRONYM_MAX_DENSE_VARIANTS", "3"))

    # Acronym ranking: pre-boost score = (1 - W) x query score + W x expansion score,
    # then the title, summary and text boosts. W: see design notes, Settings.
    ACRONYM_EXPANSION_SCORE_WEIGHT: float = float(os.getenv("ACRONYM_EXPANSION_SCORE_WEIGHT", "0.5"))

    # Only the top N candidates are rescored by the blend (uncapped it tripled latency).
    ACRONYM_RESCORE_POOL_LIMIT: int = int(os.getenv("ACRONYM_RESCORE_POOL_LIMIT", "200"))
    # Longest multi-word acronym (in words) that detection finds and upload accepts.
    ACRONYM_MAX_PHRASE_WORDS: int = int(os.getenv("ACRONYM_MAX_PHRASE_WORDS", "4"))
    # Top BM25 chunks per document read when checking its body backs an acronym claim.
    ACRONYM_BODY_CHECK_TOP_CHUNKS: int = int(os.getenv("ACRONYM_BODY_CHECK_TOP_CHUNKS", "3"))
    # Most documents whose body the text boost checks per query: only candidates that
    # can still reach the page are checked, best first, up to this many.
    ACRONYM_BODY_CHECK_MAX_SOURCES: int = int(os.getenv("ACRONYM_BODY_CHECK_MAX_SOURCES", "200"))

    # Master switch for acronym search; off gives exactly the pre-feature ranking.
    ACRONYM_SEARCH_ENABLED: bool = os.getenv("ACRONYM_SEARCH_ENABLED", "true").lower() == "true"
    # explicitly enable/disable caching of redis results (default true)
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"

    # Sparse Vector Configuration (Phase 2 — requires qdrant-client>=1.9.0)
    SPARSE_VECTOR_NAME: str = os.getenv("SPARSE_VECTOR_NAME", "bm25")
    SPARSE_SEARCH_ENABLED: bool = os.getenv("SPARSE_SEARCH_ENABLED", "false").lower() == "true"

    # Hybrid fusion: dense and sparse scores are min-max normalized, then weighted,
    # so weighted_score stays 0-1 and comparable to filter_score.
    HYBRID_DENSE_WEIGHT: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
    HYBRID_SPARSE_WEIGHT: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))

    # "weighted" (min-max score fusion) or "rrf" (rank fusion, min-max normalized).
    # Only the dense+sparse step differs; the dense side is the same in both.
    HYBRID_FUSION_METHOD: str = os.getenv("HYBRID_FUSION_METHOD", "weighted").lower()

    # Default for include_scoring_debug (fusion breakdown per result); keep false in production.
    INCLUDE_SCORING_DEBUG: bool = os.getenv("INCLUDE_SCORING_DEBUG", "false").lower() == "true"

    # Candidates per field = min(top_k x FANOUT, MAX); MAX bounds HNSW ef, the main cost.
    # A larger MAX lets more documents clear filter_score but is slower.
    SEARCH_CANDIDATE_FANOUT: int = int(os.getenv("SEARCH_CANDIDATE_FANOUT", "8"))
    SEARCH_CANDIDATE_MAX: int = int(os.getenv("SEARCH_CANDIDATE_MAX", "2000"))

    # Environment configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")  # local, production, staging, etc.

    @model_validator(mode='after')
    def _validate_fusion_config(self) -> 'Settings':
        valid_methods = {"weighted", "rrf"}
        if self.HYBRID_FUSION_METHOD not in valid_methods:
            raise ValueError(
                f"HYBRID_FUSION_METHOD must be one of {sorted(valid_methods)}, "
                f"got {self.HYBRID_FUSION_METHOD!r}"
            )
        dw, sw = self.HYBRID_DENSE_WEIGHT, self.HYBRID_SPARSE_WEIGHT
        if not (math.isfinite(dw) and dw >= 0.0):
            raise ValueError(
                f"HYBRID_DENSE_WEIGHT must be a finite non-negative number, got {dw}"
            )
        if not (math.isfinite(sw) and sw >= 0.0):
            raise ValueError(
                f"HYBRID_SPARSE_WEIGHT must be a finite non-negative number, got {sw}"
            )
        if dw + sw > 1.0 + 1e-9:
            raise ValueError(
                f"HYBRID_DENSE_WEIGHT + HYBRID_SPARSE_WEIGHT must not exceed 1.0 "
                f"(got {dw} + {sw} = {dw + sw:.6f})"
            )
        return self


settings = Settings()
