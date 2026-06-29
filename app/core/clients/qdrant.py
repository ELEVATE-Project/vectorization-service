from typing import Any
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import PayloadSchemaType
from app.config import settings
from app.core.clients.embedding import embedding_model
import logging

logger = logging.getLogger(__name__)

# Initialize client.
# check_compatibility driven by QDRANT_CHECK_COMPATIBILITY env var (default false).
# QA server is pinned at 1.12, client at 1.18
# (required for BM25 sparse search). The client emits a blanket version-gap warning
# because the minor diff exceeds 1, but every operation this service uses — multi-field
# dense + BM25 sparse query_batch_points, scroll/MatchText/MatchAny, retrieve, named-
# dense + sparse(IDF) upsert, payload indexes, set_payload — is supported by server 1.12
# and verified working. See the "Qdrant 1.18 client <-> 1.12 server compatibility"
# section in CLAUDE.md for the supported/unsupported feature matrix before adding any
# newer Qdrant feature (MatchPhrase, FormulaQuery, post-1.12 TextIndexParams fields).
qdrant_client = QdrantClient(
    settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    check_compatibility=settings.QDRANT_CHECK_COMPATIBILITY,
)

# Prefix-tokenized text index for title/summary. The PREFIX tokenizer indexes every
# prefix of each token (e.g. "insurance" → "in", "ins", "insu", ...), so partial and
# typed-mid queries retrieve candidates at the Qdrant level via MatchText. lowercase
# makes matching case-insensitive; min/max bound the prefix lengths that get indexed.
_PREFIX_TEXT_INDEX = models.TextIndexParams(
    type="text",
    tokenizer=models.TokenizerType.PREFIX,
    min_token_len=2,
    max_token_len=20,
    lowercase=True,
)

# Payload fields to index and their schema/params.
# Keyword indexes support exact MatchAny/MatchValue filters (used for source_id, company, tags).
# Text indexes support MatchText substring/full-text filters (title, summary, DOCUMENT_TYPE).
_PAYLOAD_INDEXES = [
    ("source_id",              PayloadSchemaType.KEYWORD),
    ("metadata.company",       PayloadSchemaType.KEYWORD),
    ("metadata.type",          PayloadSchemaType.KEYWORD),
    ("tags",                   PayloadSchemaType.KEYWORD),
    ("metadata.DOCUMENT_TYPE", PayloadSchemaType.TEXT),
    ("title",                  _PREFIX_TEXT_INDEX),
    ("summary",                _PREFIX_TEXT_INDEX),
]


async def ensure_collections_exist():
    """Ensure both document and cache collections exist with named vectors support"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        # Single vector config for backward compatibility
        single_vector_config = models.VectorParams(
            size=embedding_model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        )

        # Named vectors config for multiple embeddings (text, title, summary, tags, metadata)
        named_vectors_config = {
            "text": models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
            "title": models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
            "summary": models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
            "tags": models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
            "metadata": models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        }

        if settings.COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection with named vectors: {settings.COLLECTION_NAME}")
            create_kwargs: dict = dict(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=named_vectors_config,
            )
            if settings.SPARSE_SEARCH_ENABLED:
                # SparseVectorParams and Modifier require qdrant-client>=1.9.0
                try:
                    from qdrant_client.models import SparseVectorParams, Modifier  # type: ignore[import]
                    create_kwargs["sparse_vectors_config"] = {
                        settings.SPARSE_VECTOR_NAME: SparseVectorParams(
                            modifier=Modifier.IDF
                        )
                    }
                    logger.info(
                        f"Sparse vector field '{settings.SPARSE_VECTOR_NAME}' "
                        "added to collection config"
                    )
                except ImportError:
                    logger.warning(
                        "SPARSE_SEARCH_ENABLED=true but qdrant-client<1.9.0 is installed. "
                        "Sparse vectors will not be created. Upgrade qdrant-client to enable."
                    )
            qdrant_client.create_collection(**create_kwargs)
        elif settings.SPARSE_SEARCH_ENABLED:
            # Collection already exists — try to add the sparse vector field non-destructively.
            _ensure_sparse_vector_field(settings.COLLECTION_NAME)

        if settings.QA_CACHE_COLLECTION not in collection_names:
            logger.info(f"Creating collection: {settings.QA_CACHE_COLLECTION}")
            qdrant_client.create_collection(
                collection_name=settings.QA_CACHE_COLLECTION,
                vectors_config=single_vector_config
            )

        # Create payload indexes on frequently filtered/searched fields.
        # create_payload_index is idempotent — safe to call on every startup.
        _ensure_payload_indexes(settings.COLLECTION_NAME)

        return True
    except Exception as e:
        logger.error(f"Failed to create collections: {str(e)}")
        raise


def _ensure_sparse_vector_field(collection_name: str) -> None:
    """Add the BM25 sparse vector field to an existing collection.

    Uses ``update_collection`` which is non-destructive — dense vectors and
    existing payload are preserved. Requires qdrant-client>=1.9.0 and a
    Qdrant server that supports sparse vectors (>=1.7.0).
    """
    try:
        from qdrant_client.models import SparseVectorParams, Modifier  # type: ignore[import]
    except ImportError:
        logger.warning(
            "Cannot add sparse vector field: qdrant-client<1.9.0. "
            "Upgrade to enable Phase 2 hybrid search."
        )
        return

    from qdrant_client.http.exceptions import UnexpectedResponse
    from app.config import settings as _s

    try:
        qdrant_client.update_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                _s.SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            },
        )
        logger.info(
            f"Sparse vector field '{_s.SPARSE_VECTOR_NAME}' "
            f"added/verified on existing collection '{collection_name}'"
        )
    except UnexpectedResponse as exc:
        content = exc.content.decode("utf-8", errors="replace").lower()
        if "already" in content:
            # Benign: field is already present; update_collection is idempotent.
            logger.debug(
                f"Sparse vector field already present on '{collection_name}', skipping update"
            )
        else:
            logger.error(
                f"Server rejected sparse-vector field update for '{collection_name}': {exc}"
            )
            raise


def _index_params_match(existing_schema: Any, desired_schema: Any) -> bool:
    """Return True if an existing payload index already matches the desired schema.

    For simple schema types (KEYWORD/TEXT) presence alone is treated as a match —
    re-creating them is a cheap no-op. For TextIndexParams we compare the tokenizer
    so that switching to the PREFIX tokenizer triggers a rebuild exactly once.
    """
    if isinstance(desired_schema, models.TextIndexParams):
        existing_params = getattr(existing_schema, "params", None)
        existing_tokenizer = getattr(existing_params, "tokenizer", None)
        return existing_tokenizer == desired_schema.tokenizer
    # Simple schema type: an existing index of any kind is good enough.
    return True


def _ensure_payload_indexes(collection_name: str) -> None:
    """Create payload indexes for fast filtering and MatchText search.

    Simple-schema indexes are created idempotently. For text indexes whose params
    changed (e.g. switching ``title`` to the PREFIX tokenizer), the stale index is
    deleted and rebuilt once — subsequent startups detect a match and skip the
    rebuild, so there is no per-restart index churn.
    """
    try:
        existing_schema = qdrant_client.get_collection(collection_name).payload_schema or {}
    except Exception as exc:
        logger.warning(f"Could not read payload schema for '{collection_name}': {exc}")
        existing_schema = {}

    for field_name, schema_type in _PAYLOAD_INDEXES:
        try:
            current = existing_schema.get(field_name)
            if current is not None and not _index_params_match(current, schema_type):
                logger.info(f"Payload index '{field_name}' params changed — rebuilding")
                qdrant_client.delete_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                )
                current = None

            if current is None:
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.info(f"Payload index created: {field_name} ({schema_type})")
            else:
                logger.debug(f"Payload index already present: {field_name}")
        except Exception as exc:
            # Non-fatal: log and continue. An existing index or unsupported
            # schema on an older Qdrant server version will not break search.
            logger.warning(f"Could not ensure payload index for '{field_name}': {exc}")


def batch_points(points: list, batch_size: int = 100):
    """Yield successive batch_size chunks from points list"""
    for i in range(0, len(points), batch_size):
        yield points[i:i + batch_size]


def upload_to_qdrant(points: list, collection_name: str, batch_size: int = 100):
    """Upload points to Qdrant in batches with error handling"""
    total_points = len(points)
    success_count = 0
    error_count = 0

    logger.info(f"Starting batched upload of {total_points} points")

    for i, batch in enumerate(batch_points(points, batch_size)):
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            success_count += len(batch)
            logger.info(f"Uploaded batch {i + 1} ({success_count}/{total_points} points)")
        except Exception as e:
            error_count += len(batch)
            logger.error(f"Failed to upload batch {i + 1}: {str(e)}")
            continue

    return {
        "total_points": total_points,
        "success_count": success_count,
        "error_count": error_count
    }
