"""BM25 sparse vector encoder for Phase 2 hybrid search.

Requires: qdrant-client[fastembed]>=1.9.0  (see requirements.txt)

This module is intentionally guarded behind SPARSE_SEARCH_ENABLED so that the
service continues to start and operate normally when the qdrant-client upgrade
has not yet been applied (Phase 1 deployments).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy singleton — loaded only when sparse search is first requested.
_sparse_client: Optional[object] = None
_SPARSE_MODEL = "Qdrant/bm25"


def _get_sparse_client():
    """Return a qdrant-client TextEmbeddings instance backed by BM25 via FastEmbed.

    The 'in-memory' QdrantClient is used purely as a container for the FastEmbed
    sparse encoder — no vectors are stored in it. This approach avoids spinning up
    a second real Qdrant connection.
    """
    global _sparse_client
    if _sparse_client is not None:
        return _sparse_client

    try:
        from qdrant_client import QdrantClient  # type: ignore[import]
        client = QdrantClient(":memory:")
        client.set_sparse_model(_SPARSE_MODEL)
        _sparse_client = client
        logger.info(f"Sparse BM25 encoder initialised (model: {_SPARSE_MODEL})")
    except Exception as exc:
        logger.error(
            f"Failed to initialise sparse BM25 encoder: {exc}. "
            "Ensure qdrant-client[fastembed]>=1.9.0 is installed."
        )
        raise

    return _sparse_client


def generate_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """Encode *text* into a BM25 sparse vector.

    Returns:
        (indices, values) — parallel lists suitable for building a
        ``qdrant_client.models.SparseVector(indices=..., values=...)``.

    Raises:
        RuntimeError: if the sparse encoder cannot be initialised.
    """
    if not text or not text.strip():
        return [], []

    try:
        client = _get_sparse_client()
        # embed_documents returns a list of SparseEmbedding objects
        results = list(client.embed_sparse(documents=[text], model_name=_SPARSE_MODEL))
        if not results:
            return [], []

        embedding = results[0]
        indices = embedding.indices.tolist()
        values = embedding.values.tolist()
        return indices, values

    except Exception as exc:
        logger.error(f"Sparse vector generation failed: {exc}")
        raise RuntimeError(f"Sparse vector generation failed: {exc}") from exc


def is_sparse_available() -> bool:
    """Return True if the BM25 sparse encoder can be loaded."""
    try:
        _get_sparse_client()
        return True
    except Exception:
        return False
