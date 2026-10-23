# config.py
import math
import os
from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

from app.constants import constants as const

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
    # Small on purpose: redis-py has no default timeout, so a stuck Redis would stall search.
    REDIS_SOCKET_CONNECT_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", 1))
    REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", 1))
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", 86400))  # 24 hours in seconds
    # Caches "not an acronym" so ordinary words skip Postgres; shorter than REDIS_CACHE_TTL.
    # Uploads reload the cache, so this only bounds staleness in rare races.
    REDIS_NEGATIVE_CACHE_TTL: int = int(os.getenv("REDIS_NEGATIVE_CACHE_TTL", 3600))  # 1 hour
    # Short: a DB-error miss must not hide real acronyms once Postgres is back.
    REDIS_DB_ERROR_CACHE_TTL: int = int(os.getenv("REDIS_DB_ERROR_CACHE_TTL", 30))
    REDIS_MAX_CACHE_SIZE: int = int(os.getenv("REDIS_MAX_CACHE_SIZE", 1000))
    DATABASE_URL: str = os.getenv("POSTGRES_DATABASE_URI", "postgresql://anuj:1234@localhost:5432/ai_vector_service")
    # Seconds to open a connection; without it an unreachable host hangs for minutes.
    POSTGRES_CONNECT_TIMEOUT: int = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", 3))
    # Longest query; enforced by Postgres, so it stops slow or locked queries, not a frozen server.
    POSTGRES_STATEMENT_TIMEOUT_MS: int = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", 5000))
    REDIS_CACHE_ENABLED: bool = False

    # Secret for the internal_access_token header on the acronym endpoints; no default.
    INTERNAL_API_TOKEN: str = os.getenv("INTERNAL_API_TOKEN", "")

    # URL extraction settings
    URL_EXTRACTION_CHUNK_SIZE: int = 1500
    URL_EXTRACTION_CHUNK_OVERLAP: int = 300  # 20% overlap
    URL_REQUEST_TIMEOUT: int = 30  # seconds

    # File upload settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 1024))  # 1GB default (in MB)
    # Acronym bulk-upload CSVs are small tabular text, not documents —
    # a much lower cap than MAX_FILE_SIZE_MB.
    ACRONYM_BULK_UPLOAD_MAX_SIZE_MB: int = int(os.getenv("ACRONYM_BULK_UPLOAD_MAX_SIZE_MB", 5))
    # Rows saved per database statement during a bulk upload.
    ACRONYM_BULK_UPLOAD_BATCH_SIZE: int = int(os.getenv("ACRONYM_BULK_UPLOAD_BATCH_SIZE", 200))


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
    # Acronym queries only: the title boost for the query with the acronym written out
    # ("PTM document" -> "Parent Teacher Meeting document"); both can apply. 1.0 = off.
    # Title equals the written-out query:
    ACRONYM_EXPANSION_EXACT_TITLE_BOOST: float = float(os.getenv("ACRONYM_EXPANSION_EXACT_TITLE_BOOST", "1.7"))
    # Title contains the written-out query ("Parent Teacher Meeting document in school"):
    ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST: float = float(os.getenv("ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST", "1.4"))
    # The same two matches in the summary (equals / contains).
    ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST: float = float(os.getenv("ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST", "1.4"))
    ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST: float = float(os.getenv("ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST", "1.2"))
    METADATA_MATCH_BOOST: float = float(os.getenv("METADATA_MATCH_BOOST", "1.2"))
    # How many keyword-matched documents we are willing to score per search. Scoring
    # each one downloads 5 vectors, and a short query can match thousands of titles.
    # Past this limit we skip scoring them and leave their field_scores as None.
    INJECTED_DOC_SCORING_MAX: int = int(os.getenv("INJECTED_DOC_SCORING_MAX", "200"))
    # Queries shorter than this word count skip spaCy stop-word removal
    SHORT_QUERY_THRESHOLD: int = int(os.getenv("SHORT_QUERY_THRESHOLD", "3"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))  # standard Reciprocal Rank Fusion constant

    # Minimum stem length for prefix-matching expansion words ("institute" ~ "institutes").
    ACRONYM_MIN_PREFIX_MATCH_LEN: int = int(os.getenv("ACRONYM_MIN_PREFIX_MATCH_LEN", "4"))

    # Max extra characters in a prefix match: "tests" ~ "test", but not "testimony".
    ACRONYM_PREFIX_SUFFIX_CAP: int = int(os.getenv("ACRONYM_PREFIX_SUFFIX_CAP", "3"))

    # dual_query only: dense texts per query (the original plus one per meaning).
    ACRONYM_MAX_DENSE_VARIANTS: int = int(os.getenv("ACRONYM_MAX_DENSE_VARIANTS", "3"))

    # inline (default): one text with each meaning next to its acronym ("PTM (Parent Teacher Meeting)").
    # dual_query: the query and each meaning searched separately; each field keeps the higher score.
    ACRONYM_SEARCH_METHOD: str = os.getenv("ACRONYM_SEARCH_METHOD", const.ACRONYM_SEARCH_METHOD_INLINE)
    # One BM25 search per meaning ("diet" OR "District Institute ..."); each chunk keeps the higher score.
    ACRONYM_KEYWORD_OR: bool = os.getenv("ACRONYM_KEYWORD_OR", "false").lower() == "true"
    # Longest multi-word acronym (in words) that detection finds and upload accepts.
    ACRONYM_MAX_PHRASE_WORDS: int = int(os.getenv("ACRONYM_MAX_PHRASE_WORDS", "4"))
    # Words of the query checked for acronyms; bounds lookups for very long queries.
    ACRONYM_MAX_QUERY_WORDS: int = int(os.getenv("ACRONYM_MAX_QUERY_WORDS", "30"))

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

    # Hybrid fusion method — how the dense and sparse modalities are combined into
    # the final weighted_score that orders results. Selectable per deployment:
    #   "weighted" (default): HYBRID_DENSE_WEIGHT * minmax(dense) +
    #                         HYBRID_SPARSE_WEIGHT * minmax(sparse). Score-based.
    #   "rrf": Reciprocal Rank Fusion over two lists — the combined dense list
    #          (ranked by the weighted multi-field cosine sum) and the sparse list —
    #          rrf = 1/(RRF_K+dense_rank) + 1/(RRF_K+sparse_rank), then min-max
    #          normalized to [0, 1]. Rank-based; robust across retrievers.
    # In BOTH modes the dense component remains the weighted multi-field cosine sum
    # (SEARCH_PRIORITY_WEIGHTS) — only the dense+sparse fusion step differs.
    HYBRID_FUSION_METHOD: str = os.getenv("HYBRID_FUSION_METHOD", "weighted").lower()

    # Default for the per-request `include_scoring_debug` flag. When true, search
    # responses surface the hybrid fusion breakdown (keyword_score, rrf_score,
    # dense_rank, sparse_rank) on every result. A request may still override this
    # per call by sending include_scoring_debug explicitly. Keep false in
    # production (responses stay lean given the large default top_k).
    INCLUDE_SCORING_DEBUG: bool = os.getenv("INCLUDE_SCORING_DEBUG", "false").lower() == "true"

    # Candidates fetched per field search: min(top_k * FANOUT, MAX).
    # FANOUT fetches more than top_k, so ranking has extra results to choose from.
    # MAX limits the cost when top_k is large; a higher MAX is slower but lets more
    # documents pass filter_score (scores are normalized across this pool).
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
        # BaseSettings reads the env var as typed, so normalize here: empty means inline.
        self.ACRONYM_SEARCH_METHOD = (
            (self.ACRONYM_SEARCH_METHOD or "").strip().lower() or const.ACRONYM_SEARCH_METHOD_INLINE
        )
        if self.ACRONYM_SEARCH_METHOD not in const.ACRONYM_SEARCH_METHODS:
            raise ValueError(
                f"ACRONYM_SEARCH_METHOD must be one of {list(const.ACRONYM_SEARCH_METHODS)} "
                f"(empty means {const.ACRONYM_SEARCH_METHOD_INLINE}), got {self.ACRONYM_SEARCH_METHOD!r}"
            )
        # Redis rejects a TTL of 0 or less, which would quietly stop the acronym cache.
        for name in ("REDIS_CACHE_TTL", "REDIS_NEGATIVE_CACHE_TTL", "REDIS_DB_ERROR_CACHE_TTL"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be a whole number of seconds above 0, got {value}")
        # 6 values per row; Postgres allows at most 65,535 per statement.
        if not 1 <= self.ACRONYM_BULK_UPLOAD_BATCH_SIZE <= 10000:
            raise ValueError(
                f"ACRONYM_BULK_UPLOAD_BATCH_SIZE must be between 1 and 10000, "
                f"got {self.ACRONYM_BULK_UPLOAD_BATCH_SIZE}"
            )
        for name in ("ACRONYM_EXPANSION_EXACT_TITLE_BOOST", "ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST",
                     "ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST", "ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 1.0):
                raise ValueError(f"{name} must be a finite number >= 1.0, got {value}")
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
