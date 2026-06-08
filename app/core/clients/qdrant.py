from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.core.clients.embedding import embedding_model
import logging

logger = logging.getLogger(__name__)

# Initialize client
qdrant_client = QdrantClient(settings.QDRANT_HOST, port=settings.QDRANT_PORT)


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
            qdrant_client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=named_vectors_config
            )

        if settings.QA_CACHE_COLLECTION not in collection_names:
            logger.info(f"Creating collection: {settings.QA_CACHE_COLLECTION}")
            qdrant_client.create_collection(
                collection_name=settings.QA_CACHE_COLLECTION,
                vectors_config=single_vector_config
            )

        return True
    except Exception as e:
        logger.error(f"Failed to create collections: {str(e)}")
        raise


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
