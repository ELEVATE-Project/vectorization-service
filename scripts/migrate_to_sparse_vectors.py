"""One-time migration script: add BM25 sparse vectors to existing Qdrant documents.

Run ONCE after upgrading to qdrant-client[fastembed]>=1.9.0 and setting
SPARSE_SEARCH_ENABLED=true.

Usage:
    python scripts/migrate_to_sparse_vectors.py [--dry-run] [--batch-size N]

The script scrolls through every point in the ``documents`` collection.
For each point that is missing the BM25 sparse vector it:
  1. Generates a BM25 sparse vector from ``payload["text"]``.
  2. Upserts only the sparse vector for that point (dense vectors are untouched).

Running the script a second time is safe (idempotent) — points that already
have the sparse vector are skipped.

Requirements:
    - QDRANT_HOST / QDRANT_PORT env vars (or defaults 127.0.0.1 / 6333)
    - COLLECTION_NAME env var (or default "documents")
    - SPARSE_VECTOR_NAME env var (or default "bm25")
    - qdrant-client[fastembed]>=1.9.0 installed in this environment
"""
import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate documents collection to include BM25 sparse vectors.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing any changes.")
    parser.add_argument("--batch-size", type=int, default=100, help="Points to upsert per Qdrant request (default 100).")
    parser.add_argument("--scroll-limit", type=int, default=1000, help="Points to fetch per scroll page (default 1000).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import after arg parsing so --help works without env/deps
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import SparseVector, PointVectors
    except ImportError as exc:
        logger.error(f"qdrant-client import failed: {exc}. Install qdrant-client[fastembed]>=1.9.0.")
        sys.exit(1)

    try:
        from app.core.clients.sparse_encoder import generate_sparse_vector
    except ImportError as exc:
        logger.error(f"sparse_encoder import failed: {exc}. Run from the vectorization-service root.")
        sys.exit(1)

    host = os.getenv("QDRANT_HOST", "127.0.0.1")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("COLLECTION_NAME", "documents")
    sparse_name = os.getenv("SPARSE_VECTOR_NAME", "bm25")

    client = QdrantClient(host=host, port=port)
    logger.info(f"Connected to Qdrant at {host}:{port}, collection='{collection}'")

    if args.dry_run:
        logger.info("DRY RUN — no changes will be written.")

    total_scanned = 0
    total_skipped = 0   # already have sparse vector
    total_migrated = 0
    total_errors = 0
    start_time = time.monotonic()

    offset = None
    pending_upserts: list = []

    def flush_upserts() -> None:
        nonlocal total_migrated, total_errors
        if not pending_upserts or args.dry_run:
            if args.dry_run and pending_upserts:
                logger.info(f"[DRY RUN] Would upsert {len(pending_upserts)} points")
                del pending_upserts[:]
            return
        try:
            client.upsert(collection_name=collection, points=pending_upserts)
            total_migrated += len(pending_upserts)
            logger.info(f"Upserted {len(pending_upserts)} points (total migrated: {total_migrated})")
        except Exception as exc:
            total_errors += len(pending_upserts)
            logger.error(f"Upsert batch failed: {exc}")
        del pending_upserts[:]

    logger.info("Starting scroll...")
    while True:
        scroll_kwargs: dict = dict(
            collection_name=collection,
            limit=args.scroll_limit,
            with_payload=["text"],
            with_vectors=[sparse_name],  # only fetch the sparse field to check existence
        )
        if offset is not None:
            scroll_kwargs["offset"] = offset

        try:
            points, next_offset = client.scroll(**scroll_kwargs)
        except Exception as exc:
            logger.error(f"Scroll failed: {exc}")
            break

        for point in points:
            total_scanned += 1

            # Check if sparse vector already present (non-empty indices list)
            existing_vectors = getattr(point, "vector", {}) or {}
            sparse_existing = existing_vectors.get(sparse_name)
            if sparse_existing and getattr(sparse_existing, "indices", None):
                total_skipped += 1
                continue

            # Generate sparse vector from text payload
            text = (point.payload or {}).get("text", "")
            if not text:
                logger.debug(f"Point {point.id} has no text payload — skipping")
                total_skipped += 1
                continue

            try:
                indices, values = generate_sparse_vector(text)
            except Exception as exc:
                logger.warning(f"Sparse encoding failed for point {point.id}: {exc}")
                total_errors += 1
                continue

            if not indices:
                total_skipped += 1
                continue

            pending_upserts.append(
                PointVectors(
                    id=point.id,
                    vector={sparse_name: SparseVector(indices=indices, values=values)},
                )
            )

            if len(pending_upserts) >= args.batch_size:
                flush_upserts()

        if total_scanned % 5000 == 0 and total_scanned > 0:
            elapsed = time.monotonic() - start_time
            logger.info(
                f"Progress: scanned={total_scanned}, migrated={total_migrated}, "
                f"skipped={total_skipped}, errors={total_errors}, elapsed={elapsed:.1f}s"
            )

        if not next_offset:
            break
        offset = next_offset

    flush_upserts()

    elapsed = time.monotonic() - start_time
    logger.info(
        f"Migration complete in {elapsed:.1f}s — "
        f"scanned={total_scanned}, migrated={total_migrated}, "
        f"skipped={total_skipped}, errors={total_errors}"
    )
    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
