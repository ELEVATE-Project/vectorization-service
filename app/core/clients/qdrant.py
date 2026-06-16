from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PayloadSchemaType
from app.config import settings
from app.core.clients.embedding import embedding_model
import logging

logger = logging.getLogger(__name__)

# Initialize client
qdrant_client = QdrantClient(settings.QDRANT_HOST, port=settings.QDRANT_PORT)

# Payload fields to index and their schema types.
# Keyword indexes support exact MatchAny/MatchValue filters (used for source_id, company, tags).
# Text indexes support MatchText substring/full-text filters (used for title, DOCUMENT_TYPE).
_PAYLOAD_INDEXES = [
    ("source_id",              PayloadSchemaType.KEYWORD),
    ("metadata.company",       PayloadSchemaType.KEYWORD),
    ("tags",                   PayloadSchemaType.KEYWORD),
    ("metadata.DOCUMENT_TYPE", PayloadSchemaType.TEXT),
    ("title",                  PayloadSchemaType.TEXT),
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
                    from qdrant_client.http.models import SparseVectorParams, Modifier  # type: ignore[import]
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
        from qdrant_client.http.models import SparseVectorParams, Modifier  # type: ignore[import]
        from app.config import settings as _s
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
    except ImportError:
        logger.warning(
            "Cannot add sparse vector field: qdrant-client<1.9.0. "
            "Upgrade to enable Phase 2 hybrid search."
        )
    except Exception as exc:
        # Non-fatal: the field may already exist, or the server version may not
        # support sparse vectors yet.
        logger.warning(f"Could not update collection with sparse vectors: {exc}")


def _ensure_payload_indexes(collection_name: str) -> None:
    """Create payload indexes for fast filtering and MatchText search.

    Indexes are created idempotently — if one already exists Qdrant returns a
    success response, so calling this on every startup is safe.
    """
    for field_name, schema_type in _PAYLOAD_INDEXES:
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.info(f"Payload index ensured: {field_name} ({schema_type})")
        except Exception as exc:
            # Non-fatal: log and continue. An existing index or unsupported
            # schema on an older Qdrant server version will not break search.
            logger.warning(f"Could not create payload index for '{field_name}': {exc}")


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
