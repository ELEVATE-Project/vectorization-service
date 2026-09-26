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
    # QA runs Qdrant server 1.12 while the client is pinned at 1.18 (required for BM25 sparse search).
    # The 6-minor-version gap exceeds Qdrant's allowed ≤1 diff, causing a blanket UserWarning on every
    # startup. Setting this to false suppresses that check. All operations the service uses have been
    # verified to work on server 1.12 — the warning is a false alarm for our feature set.
    # Set to true once QA server is upgraded to 1.18 to re-enable the check.
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
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    # Small on purpose: redis-py has no default timeout, and a blackholed
    # connection (packets silently dropped) would hang cache reads for minutes,
    # stalling search instead of falling back to Postgres.
    REDIS_SOCKET_CONNECT_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", 1))
    REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", 1))
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", 86400))  # 24 hours in seconds
    # Shorter than REDIS_CACHE_TTL: caches "this word isn't an acronym" so ordinary
    # non-acronym words in a query don't re-hit Postgres on every request. A newly
    # bulk-uploaded acronym overwrites any stale negative entry immediately (warm_cache()
    # runs after every upload), so this TTL only bounds staleness for the rare case that
    # invariant doesn't hold — not load-bearing correctness.
    REDIS_NEGATIVE_CACHE_TTL: int = int(os.getenv("REDIS_NEGATIVE_CACHE_TTL", 3600))  # 1 hour
    # Shorter than REDIS_NEGATIVE_CACHE_TTL: this caches a DB-error miss, not a
    # genuine "not an acronym" miss. DB outages are often transient — caching
    # "not found" for a full hour would hide real acronyms after Postgres recovers.
    REDIS_DB_ERROR_CACHE_TTL: int = int(os.getenv("REDIS_DB_ERROR_CACHE_TTL", 30))
    REDIS_MAX_CACHE_SIZE: int = int(os.getenv("REDIS_MAX_CACHE_SIZE", 1000))
    DATABASE_URL: str = os.getenv("POSTGRES_DATABASE_URI", "postgresql://anuj:1234@localhost:5432/ai_vector_service")
    # psycopg2 defaults to no connect timeout at all — a blackholed Postgres host
    # (network silently drops packets, unlike a clean "connection refused") makes
    # a connection attempt hang until the OS-level TCP timeout, which can be
    # minutes. Confirmed live with a real local "black hole" TCP server: an
    # unbounded psycopg2.connect() hung past 15s with no sign of returning.
    POSTGRES_CONNECT_TIMEOUT: int = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", 3))
    # connect_timeout above only bounds opening a connection — once one is
    # established, nothing bounded how long a QUERY running on it could take.
    # A Postgres that accepts new connections fine but is unresponsive to
    # queries (stuck lock, frozen backend) hung every request touching it
    # indefinitely instead of failing; found live via a pause test. Every
    # query this service issues is a small indexed lookup (acronym_mapping,
    # a few hundred rows; translations by chunk_id) with no legitimate reason
    # to run anywhere near this long, so 5s is generous headroom, not a tight
    # squeeze — it will essentially never fire against a healthy database.
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
        "title": 0.36,      # 36% weight for title matches
        "text": 0.27,       # 27% weight for chunk/content matches
        "tags": 0.14,       # 14% weight for tag matches
        "summary": 0.14,    # 14% weight for summary matches
        "metadata": 0.09    # 9% weight for metadata matches
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
    METADATA_MATCH_BOOST: float = float(os.getenv("METADATA_MATCH_BOOST", "1.2"))
    # Queries shorter than this word count skip spaCy stop-word removal
    SHORT_QUERY_THRESHOLD: int = int(os.getenv("SHORT_QUERY_THRESHOLD", "3"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))  # standard Reciprocal Rank Fusion constant

    # Expansion words match by prefix so inflections line up: institute /
    # institutes, program / programme. But a prefix only means something once the
    # shorter side is a word stem — below that, a single letter is a prefix of
    # anything starting with it, so "s" stands in for "school" and "S M C
    # Handbook" reads as "School Management Committee", "R.E.A.D" as "Right to
    # Education". Four characters is where a prefix stops being an initial and
    # starts being a stem.
    #
    # This is the minimum STEM length for matching an expansion's words against a
    # document. It is NOT a limit on how long an acronym may be — that is the
    # acronym_mapping column width (32), read off the model in acronym_service so
    # the two can never drift apart. Lives here rather than in constants.py so it
    # can be retuned per deployment without a code change, like every other
    # ranking knob above.
    ACRONYM_MIN_PREFIX_MATCH_LEN: int = int(os.getenv("ACRONYM_MIN_PREFIX_MATCH_LEN", "4"))

    # ACRONYM_MIN_PREFIX_MATCH_LEN alone can't tell a real inflection ("test" ~
    # "tests") apart from an unrelated word that happens to share a prefix
    # ("test" ~ "testimony"). Genuine inflections almost always add a handful
    # of characters; unrelated words tend to add far more. This caps how many
    # EXTRA characters the longer word may have over the shorter one for a
    # prefix match to count — verified against the live acronym dictionary to
    # keep standard plural/tense inflections working while blocking most
    # coincidental prefix collisions on short, common expansion words (e.g.
    # "post", "work", "home", "master").
    ACRONYM_PREFIX_SUFFIX_CAP: int = int(os.getenv("ACRONYM_PREFIX_SUFFIX_CAP", "3"))

    # AC-11: an ambiguous acronym (SSC = Staff Selection Commission OR Sainik
    # School Society) used to only ever get a dense variant for its FIRST
    # listed expansion — semantic search never explored the other meaning at
    # all, while the sparse/BM25 query mixed both meanings' words together
    # regardless. Now one dense variant is built per expansion (capped by
    # this setting, total dense texts including the original query), so
    # semantic search genuinely considers each registered meaning. Only 3
    # dictionary entries are ever ambiguous today (SSC, DM, MIP), so this
    # rarely adds more than one extra variant in practice.
    ACRONYM_MAX_DENSE_VARIANTS: int = int(os.getenv("ACRONYM_MAX_DENSE_VARIANTS", "3"))

    # Soft acronym ranking — replaces the hard tiers. On an acronym query:
    #   relevance = (1 - W) x score_against_query + W x score_against_expansion
    # each side being the usual 70/30 dense/sparse fusion, then
    #   final = relevance x (1 + bonus)
    # with the best bonus that applies below. A title can therefore close at most
    # a (bonus) relative gap; it can never lift a weak document over a much
    # stronger one, which is what the tiers did. The acronym bonuses additionally
    # require the document's content to mention the acronym (BM25), so a title
    # alone never earns them.
    #
    # W: the expansion phrase is both specific ("Parent Teacher Meeting") and
    # generic ("District Institute of Education and Training") depending on the
    # acronym, so it counts for part of the score, not all of it or none.
    # Defaults chosen by simulation on the local corpus (DIET/PTM/SMC/SSC,
    # scripts/simulate_soft_acronym_boost.py) — tune against real expected
    # results before treating them as settled.
    #
    # 0.2, not 0.4: measured on the SLEM fixture (see handoff/03), 0.4 lets a
    # generic expansion phrase ("School Library...") pollute the score enough
    # that unrelated documents sharing that vocabulary outrank a document that
    # names the acronym literally 12 times. 0.2 keeps the content gradient
    # among acronym-bearing documents while avoiding that pollution; 0.0 loses
    # the gradient entirely.
    ACRONYM_EXPANSION_SCORE_WEIGHT: float = float(os.getenv("ACRONYM_EXPANSION_SCORE_WEIGHT", "0.2"))
    ACRONYM_BONUS_TITLE_ACRONYM: float = float(os.getenv("ACRONYM_BONUS_TITLE_ACRONYM", "0.20"))
    ACRONYM_BONUS_TITLE_EXPANSION: float = float(os.getenv("ACRONYM_BONUS_TITLE_EXPANSION", "0.15"))
    ACRONYM_BONUS_SUMMARY_ACRONYM: float = float(os.getenv("ACRONYM_BONUS_SUMMARY_ACRONYM", "0.10"))
    ACRONYM_BONUS_SUMMARY_EXPANSION: float = float(os.getenv("ACRONYM_BONUS_SUMMARY_EXPANSION", "0.05"))

    # A document matching several detected acronyms sums each one's own best
    # grade (AC-15: it used to just take the single best grade across all of
    # them, so a document about both SMC and DIET ranked identically to one
    # about only DIET). Summing is capped so it stays bounded -- two acronyms
    # both hitting a full title match would otherwise sum to 0.40 (2x
    # ACRONYM_BONUS_TITLE_ACRONYM), letting the multiplier run away for a
    # query naming several acronyms at once. 0.35 gives a genuinely higher
    # ceiling than a single match's 0.20 (multiplier up to 1.35x vs 1.2x) --
    # a visible reward for matching more -- without approaching 2x.
    ACRONYM_BONUS_MULTI_MATCH_CAP: float = float(os.getenv("ACRONYM_BONUS_MULTI_MATCH_CAP", "0.35"))

    # _blended_acronym_relevance rescores the retrieved pool against the query
    # and the expansion separately (see above) -- a second and third Qdrant
    # round trip on top of the initial retrieval call. Uncapped, this doubled
    # to nearly quadrupled search latency (+115% at top_k=10, +297% at
    # top_k=1000, measured). Capping to the top N candidates by the already-
    # computed retrieval score keeps the cost bounded. Blending only reweights
    # between two relevance signals the pool is already ranked by, so a
    # document far down that ranking is not going to leapfrog into the top
    # results after blending; those are left at their retrieval score instead
    # of being rescored.
    ACRONYM_RESCORE_POOL_LIMIT: int = int(os.getenv("ACRONYM_RESCORE_POOL_LIMIT", "200"))

    # Acronym Search — on by default, matching .env.sample and the 2.1.0 release
    # note, which both already described it that way while the code still defaulted
    # to false. Set ACRONYM_SEARCH_ENABLED=false to disable the whole
    # detect -> expand -> tiered-rank path; retrieval and ranking then behave
    # exactly as they did before the feature (non-acronym queries are byte-identical
    # either way, which TestFusionFormulaHasNoAcronymBranch and the offline
    # comparison both pin).
    ACRONYM_SEARCH_ENABLED: bool = os.getenv("ACRONYM_SEARCH_ENABLED", "true").lower() == "true"
    # explicitly enable/disable caching of redis results (default true)
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"

    # Sparse Vector Configuration (Phase 2 — requires qdrant-client>=1.9.0)
    SPARSE_VECTOR_NAME: str = os.getenv("SPARSE_VECTOR_NAME", "bm25")
    SPARSE_SEARCH_ENABLED: bool = os.getenv("SPARSE_SEARCH_ENABLED", "false").lower() == "true"

    # Hybrid score fusion weights. In hybrid mode the dense (cosine) and sparse
    # (BM25) scores are each min-max normalized to [0, 1] across the candidate
    # pool, then combined as: HYBRID_DENSE_WEIGHT * dense + HYBRID_SPARSE_WEIGHT * sparse.
    # This yields a calibrated 0-1 weighted_score comparable to filter_score
    # (raw RRF fused scores are ~0-0.1 and would never clear a cosine-scale
    # threshold like 0.35, silently dropping every hybrid result).
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

    # Candidate pool sizing for multi-field search. Each dense named-vector search
    # (title/text/tags/summary/metadata) and the sparse BM25 search retrieves up to
    # `min(top_k * SEARCH_CANDIDATE_FANOUT, SEARCH_CANDIDATE_MAX)` candidates; the
    # union is fused/ranked client-side. The CAP bounds HNSW `ef` — the dominant
    # query cost — so a large top_k cannot trigger a 10k-deep traversal per field;
    # the FANOUT gives small-top_k callers a re-ranking margin (retrieve more than
    # you return). For top_k=1000 the CAP wins → 2000/field (was 10000).
    #
    # CAP also influences result `count`: the hybrid score is min-max normalized
    # across the candidate pool, so a larger pool lets more docs clear filter_score.
    # 2000 balances latency (~2x faster than the old 10000) against count fidelity.
    # Raise toward 10000 for fuller counts, lower toward 500 for max speed.
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
