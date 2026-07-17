import logging
import math
from typing import Any, List

from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

# Expected dimensionality of every vector this service produces/queries (e.g. 384 for
# all-MiniLM-L6-v2). Resolved once from the loaded model so it tracks the configured
# EMBEDDING_MODEL rather than being hard-coded.
EMBEDDING_DIM: int = embedding_model.get_embedding_dimension()


class EmbeddingError(ValueError):
    """Raised when a query embedding cannot be produced or is malformed.

    Carries enough context for structured logging and for the API layer to return a
    meaningful error instead of leaking a cryptic Qdrant ``400`` (e.g.
    ``Vector dimension error: expected dim: 384, got 0``).
    """


def generate_embeddings(texts: list):
    """Generate embeddings for a list of texts"""
    return embedding_model.encode(texts)


def generate_single_embedding(text: str):
    """Generate embedding for a single text"""
    return embedding_model.encode(text)


def validate_vector(vec: Any) -> List[float]:
    """Coerce *vec* to a validated 1-D Python list of floats.

    Guards the embedding→Qdrant boundary: a 0-length or wrong-dimension vector reaching
    ``query_points``/``query_batch_points`` produces an opaque server-side 400. We catch
    it in-process instead and raise :class:`EmbeddingError` with the offending dimension.

    Returns:
        The vector as a ``list[float]`` of length ``EMBEDDING_DIM``.

    Raises:
        EmbeddingError: if the vector is empty, the wrong dimension, or non-finite.
    """
    # numpy arrays / tensors expose tolist(); fall back to list() for plain sequences.
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    elif not isinstance(vec, list):
        try:
            vec = list(vec)
        except TypeError as exc:
            raise EmbeddingError(f"Embedding is not a sequence: {type(vec).__name__}") from exc

    got_dim = len(vec)
    if got_dim != EMBEDDING_DIM:
        raise EmbeddingError(
            f"Invalid embedding dimension: expected {EMBEDDING_DIM}, got {got_dim}"
        )

    if any((v is None) or (isinstance(v, float) and not math.isfinite(v)) for v in vec):
        raise EmbeddingError("Embedding contains null or non-finite values")

    return vec


def embed_query(text: str) -> List[float]:
    """Embed a *query* string and return a validated ``list[float]``.

    This is the single entry point every search service should use for query vectors.
    It rejects empty/whitespace input up front and validates the produced vector before
    it can reach Qdrant.

    Raises:
        EmbeddingError: if *text* is empty/whitespace, embedding fails, or the produced
            vector is empty/malformed.
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed an empty or whitespace-only query")

    # Let genuine model failures (e.g. RuntimeError/OOM) propagate unchanged so they
    # surface as 5xx — only empty/whitespace input and malformed *output* vectors are
    # EmbeddingError (mapped to 422). validate_vector guards the output.
    raw = generate_embeddings([text])[0]
    return validate_vector(raw)
