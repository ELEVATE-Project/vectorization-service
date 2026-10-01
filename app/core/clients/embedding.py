import logging
import math
from typing import Any, List, Union

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

    Lets the API return a clear error instead of Qdrant's cryptic dimension 400.
    """


def generate_embeddings(texts: list):
    """Generate embeddings for a list of texts"""
    return embedding_model.encode(texts)


def generate_single_embedding(text: str):
    """Generate embedding for a single text"""
    return embedding_model.encode(text)


def validate_vector(vec: Any) -> List[float]:
    """Return *vec* as a 1-D list of EMBEDDING_DIM finite floats.

    Raises EmbeddingError if it is empty, the wrong dimension, or non-finite.
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


def embed_query(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """Embed one query or a list of them in one encode() call; the return mirrors the input shape.

    Raises EmbeddingError for empty text, a failed encode, or a malformed vector.
    """
    single = isinstance(text, str)
    texts = [text] if single else list(text)

    for one in texts:
        if not one or not one.strip():
            raise EmbeddingError("Cannot embed an empty or whitespace-only query")

    # Let genuine model failures (e.g. RuntimeError/OOM) propagate unchanged so they
    # surface as 5xx — only empty/whitespace input and malformed *output* vectors are
    # EmbeddingError (mapped to 422). validate_vector guards the output.
    vectors = [validate_vector(vec) for vec in generate_embeddings(texts)]
    return vectors[0] if single else vectors
