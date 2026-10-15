"""Remove title/summary/tags copies from payload.metadata on existing points.

Uploads now store title/summary/tags only at the payload top level (search responses
return them there too). This brings existing points in line; metadata.TITLE is kept.

For each point:
  - if a top-level field is empty but metadata holds a value, it is backfilled to
    the top level first (legacy points), so nothing is lost;
  - title/summary/tags are then removed from metadata.
Vectors are not touched.

Usage (from the vectorization-service root):
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py --dry-run
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py
"""
import argparse
import logging

from app.config import settings
from app.core.clients.qdrant import qdrant_client

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_update(payload: dict):
    """Return the payload keys to overwrite for one point, or None if already clean"""
    metadata = dict(payload.get("metadata") or {})
    if not any(field in metadata for field in settings.OMITTED_FIELDS_FROM_METADATA):
        return None

    update = {}
    for field in settings.OMITTED_FIELDS_FROM_METADATA:
        value = metadata.pop(field, None)
        # Backfill legacy points whose only copy lives in metadata
        if value and not payload.get(field):
            update[field] = value
    update["metadata"] = metadata
    return update


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="count affected points without writing")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    scanned = cleaned = backfilled = 0
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=args.batch_size,
            offset=offset,
            with_payload=["title", "summary", "tags", "metadata"],
            with_vectors=False,
        )
        for point in points:
            scanned += 1
            update = build_update(point.payload or {})
            if update is None:
                continue
            cleaned += 1
            backfilled += len(update) > 1

            # set_payload overwrites only the given top-level keys (metadata as a whole)
            if not args.dry_run:
                qdrant_client.set_payload(
                    collection_name=settings.COLLECTION_NAME,
                    payload=update,
                    points=[point.id],
                )
        if offset is None:
            break

    mode = "DRY RUN - " if args.dry_run else ""
    logger.info(
        f"{mode}collection={settings.COLLECTION_NAME} scanned={scanned} "
        f"cleaned={cleaned} backfilled_top_level={backfilled}"
    )


if __name__ == "__main__":
    main()
