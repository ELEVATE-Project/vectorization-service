"""Report the percentage of points missing the title/summary/tags named vectors.

title/summary/tags vectors are document-level (generated from optional upload
fields), so documents uploaded without one can end up with that named vector
missing or empty on all of their points. This is a read-only audit — no data
is modified.

Talks directly to Qdrant via its own client, independent of the vectorization
service. It does not import app.core.clients.qdrant (which pulls in the
SentenceTransformer embedding model) or start the FastAPI app — only
app.config (host/port/collection settings) and the qdrant-client SDK.

Usage (from the vectorization-service root):
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py --collection-name documents
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py --batch-size 512
"""
import argparse
import logging

from qdrant_client import QdrantClient

from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
# qdrant-client logs every HTTP call via httpx at INFO; keep this script's output to
# just the summary.
logging.getLogger("httpx").setLevel(logging.WARNING)

AUDITED_VECTORS = ("title", "summary", "tags")


def build_qdrant_client() -> QdrantClient:
    # Mirrors app/core/clients/qdrant.py's client construction, without importing
    # that module (which also imports the embedding model).
    host_is_url = settings.QDRANT_HOST.startswith(("http://", "https://"))
    return QdrantClient(
        settings.QDRANT_HOST,
        port=None if host_is_url else settings.QDRANT_PORT,
        check_compatibility=settings.QDRANT_CHECK_COMPATIBILITY,
    )


def is_missing(vector, name):
    # Defensive: a point's vector payload may be None, absent the key, or an empty list
    return not (vector or {}).get(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--collection-name",
        default=settings.COLLECTION_NAME,
        help="Qdrant collection to audit (default: COLLECTION_NAME from env/settings)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    qdrant_client = build_qdrant_client()

    total_points = 0
    missing_counts = {name: 0 for name in AUDITED_VECTORS}

    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=args.collection_name,
            limit=args.batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=list(AUDITED_VECTORS),
        )
        for point in points:
            total_points += 1
            vector = getattr(point, "vector", None)
            for name in AUDITED_VECTORS:
                if is_missing(vector, name):
                    missing_counts[name] += 1
        if offset is None:
            break

    logger.info(f"collection={args.collection_name} total_points={total_points}")
    for name in AUDITED_VECTORS:
        missing = missing_counts[name]
        pct = (missing / total_points * 100) if total_points else 0.0
        logger.info(f"{name}: missing={missing} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
