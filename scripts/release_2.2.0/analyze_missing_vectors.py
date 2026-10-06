"""Report the percentage of points missing the title/summary/tags named vectors.

title/summary/tags vectors are document-level (generated from optional upload
fields), so documents uploaded without one can end up with that named vector
missing or empty on all of their points. This is a read-only audit — no data
is modified.

With --theme, the same pass also logs theme coverage: points and documents
with/without a theme and the count per theme. It reads only source_id and theme
(never text), and needs no theme payload index, so it works on a collection the
service has not created that index on yet (e.g. before and after a backfill).

Talks directly to Qdrant via its own client, independent of the vectorization
service. It does not import app.core.clients.qdrant (which pulls in the
SentenceTransformer embedding model) or start the FastAPI app — only
app.config (host/port/collection settings) and the qdrant-client SDK.

Usage (from the vectorization-service root):
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py --collection-name documents
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py --batch-size 512
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py --theme
"""
import argparse
import logging
from collections import defaultdict

from qdrant_client import QdrantClient

from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
# qdrant-client logs every HTTP call via httpx at INFO; keep this script's output to
# just the summary.
logging.getLogger("httpx").setLevel(logging.WARNING)

AUDITED_VECTORS = ("title", "summary", "tags")
# Payload read for --theme: enough to count documents and themes, never the chunk text
THEME_PAYLOAD_FIELDS = ["source_id", "theme"]


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


def pct(part: int, whole: int) -> float:
    return (part / whole * 100) if whole else 0.0


class ThemeCoverage:
    """Tallies theme coverage from scrolled payloads; no Qdrant filter, so no index needed"""

    def __init__(self):
        self.points = 0
        self.points_with_theme = 0
        self.documents = set()
        self.themed_documents = set()
        self.chunks_per_theme = defaultdict(int)
        self.documents_per_theme = defaultdict(set)

    def add(self, payload: dict):
        payload = payload or {}
        source_id = payload.get("source_id")
        # A point written before the theme field existed has no key at all: count it as
        # without a theme, the same as a stored null.
        theme = payload.get("theme")

        self.points += 1
        self.documents.add(source_id)
        if theme is None:
            return
        self.points_with_theme += 1
        self.themed_documents.add(source_id)
        self.chunks_per_theme[theme] += 1
        self.documents_per_theme[theme].add(source_id)


def theme_index_status(client, collection_name: str) -> str:
    """Describe the theme payload index for the log; informational only, never raises"""
    try:
        schema = client.get_collection(collection_name).payload_schema or {}
    except Exception as exc:
        return f"unknown ({exc})"

    index = schema.get("theme")
    if index is None:
        return "missing (created on the next service startup; coverage does not need it)"
    # payload_schema values carry the index type as data_type (an enum or a plain string)
    data_type = getattr(index, "data_type", index)
    return str(getattr(data_type, "value", data_type))


def log_theme_coverage(coverage: ThemeCoverage, index_status: str):
    points_without = coverage.points - coverage.points_with_theme
    total_docs = len(coverage.documents)
    docs_with = len(coverage.themed_documents)
    docs_without = total_docs - docs_with

    logger.info(f"theme index: {index_status}")
    logger.info(
        f"theme points: with={coverage.points_with_theme} ({pct(coverage.points_with_theme, coverage.points):.2f}%) "
        f"without={points_without} ({pct(points_without, coverage.points):.2f}%)"
    )
    logger.info(
        f"theme documents: total={total_docs} with={docs_with} ({pct(docs_with, total_docs):.2f}%) "
        f"without={docs_without} ({pct(docs_without, total_docs):.2f}%)"
    )
    # Most-used theme first, so a skewed backfill shows at the top
    for theme in sorted(coverage.chunks_per_theme, key=lambda t: (-len(coverage.documents_per_theme[t]), t)):
        logger.info(
            f'theme "{theme}": documents={len(coverage.documents_per_theme[theme])} '
            f"chunks={coverage.chunks_per_theme[theme]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--collection-name",
        default=settings.COLLECTION_NAME,
        help="Qdrant collection to audit (default: COLLECTION_NAME from env/settings)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--theme",
        action="store_true",
        help="also log theme coverage (works without the theme payload index)",
    )
    args = parser.parse_args()

    qdrant_client = build_qdrant_client()

    total_points = 0
    missing_counts = {name: 0 for name in AUDITED_VECTORS}
    coverage = ThemeCoverage() if args.theme else None

    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=args.collection_name,
            limit=args.batch_size,
            offset=offset,
            # Payload is read only for --theme; without it the scroll is unchanged
            with_payload=THEME_PAYLOAD_FIELDS if args.theme else False,
            with_vectors=list(AUDITED_VECTORS),
        )
        for point in points:
            total_points += 1
            vector = getattr(point, "vector", None)
            for name in AUDITED_VECTORS:
                if is_missing(vector, name):
                    missing_counts[name] += 1
            if coverage is not None:
                coverage.add(getattr(point, "payload", None))
        if offset is None:
            break

    logger.info(f"collection={args.collection_name} total_points={total_points}")
    for name in AUDITED_VECTORS:
        missing = missing_counts[name]
        logger.info(f"{name}: missing={missing} ({pct(missing, total_points):.2f}%)")

    # Theme lines come after the vector lines, so the default output stays as it was
    if coverage is not None:
        log_theme_coverage(coverage, theme_index_status(qdrant_client, args.collection_name))


if __name__ == "__main__":
    main()
