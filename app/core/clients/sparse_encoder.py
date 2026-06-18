"""BM25 sparse vector encoder for Phase 2 hybrid search.

Requires: qdrant-client[fastembed]>=1.18.0 (the ``[fastembed]`` extra pulls in the
``fastembed`` package used here directly).

This module is intentionally guarded behind SPARSE_SEARCH_ENABLED so that the
service continues to start and operate normally when sparse search is disabled.

Note: earlier versions used an in-memory ``QdrantClient`` + ``embed_sparse``;
that helper was removed in qdrant-client 1.14+, so we use the underlying
``fastembed.SparseTextEmbedding`` model directly.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy singleton — loaded only when sparse search is first requested.
_sparse_encoder: Optional[object] = None
_SPARSE_MODEL = "Qdrant/bm25"


def _get_sparse_encoder():
    """Return a cached ``fastembed.SparseTextEmbedding`` BM25 encoder."""
    global _sparse_encoder
    if _sparse_encoder is not None:
        return _sparse_encoder

    try:
        from fastembed import SparseTextEmbedding  # type: ignore[import]
        _sparse_encoder = SparseTextEmbedding(model_name=_SPARSE_MODEL)
        logger.info(f"Sparse BM25 encoder initialised (model: {_SPARSE_MODEL})")
    except Exception as exc:
        logger.error(
            f"Failed to initialise sparse BM25 encoder: {exc}. "
            "Ensure qdrant-client[fastembed]>=1.18.0 (or fastembed) is installed."
        )
        raise

    return _sparse_encoder


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
        encoder = _get_sparse_encoder()
        # embed() returns an iterator of SparseEmbedding objects
        results = list(encoder.embed([text]))
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
        _get_sparse_encoder()
        return True
    except Exception:
        return False
