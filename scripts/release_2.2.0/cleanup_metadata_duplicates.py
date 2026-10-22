"""Remove title/summary/tags/theme copies from payload.metadata on existing points.

Uploads now store title/summary/tags/theme only at the payload top level (search responses
return them there too). This brings existing points in line; metadata.TITLE is kept.

For each point:
  - if a top-level field is empty but metadata holds a value, it is backfilled to
    the top level first (legacy points), so nothing is lost;
  - title/summary/tags are then removed from metadata, and so is every theme key in
    any case (theme, THEME, ...), as uploads do;
  - if a metadata theme differs from the top-level one, the top-level value is kept
    and the mismatch is logged.
Vectors are not touched. Re-running is safe: clean points are skipped and nothing is written.

Usage (from the vectorization-service root):
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py --dry-run
    PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py
"""
import argparse
import logging
import sys
from collections import Counter

from app.config import settings
from app.core.clients.qdrant import qdrant_client
from app.utils.generic import normalize_theme

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# httpx logs every Qdrant call at INFO (one line per point write); keep only the script's own lines
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

THEME_FIELD = "theme"
# Top-level fields that can be backfilled from metadata, and the payload keys the scroll
# reads (text and vectors are skipped to keep each page small)
BACKFILL_FIELDS = [*settings.OMITTED_FIELDS_FROM_METADATA, THEME_FIELD]
SCROLL_FIELDS = [*BACKFILL_FIELDS, "metadata"]


def pick_theme_backfill(point_id, top_level_theme, metadata_values: list):
    """Return the theme to backfill to the top level, or None if the top level is set or no usable value exists.
    Inputs: point_id (for logs), the point's top-level theme, and the values removed from metadata theme keys."""
    top_level = normalize_theme(top_level_theme)
    backfill = None
    for value in metadata_values:
        # None/blank copies are dropped quietly; a non-string one is dropped and reported
        if value is None:
            continue
        if not isinstance(value, str):
            logger.warning(f"point={point_id}: metadata theme {value!r} is not a string; removed, not backfilled")
            continue
        value = normalize_theme(value)
        if value is None:
            continue

        # The top-level theme always wins; a different metadata copy is only reported
        if top_level is not None:
            if value != top_level:
                logger.warning(
                    f"point={point_id}: metadata theme {value!r} differs from top-level theme "
                    f"{top_level!r}; kept top-level"
                )
        elif backfill is None:
            backfill = value
        elif value != backfill:
            logger.warning(
                f"point={point_id}: metadata themes differ ({backfill!r} vs {value!r}); backfilled {backfill!r}"
            )
    return backfill


def build_update(point_id, payload: dict):
    """Return the payload keys to overwrite for one point, or None if its metadata is already clean.
    Inputs: point_id (for logs) and the point's payload (title, summary, tags, theme, metadata) from scroll."""
    metadata = payload.get("metadata")
    # No metadata means nothing to clean; a non-object one is left untouched and reported
    if not metadata:
        return None
    if not isinstance(metadata, dict):
        logger.warning(f"point={point_id}: metadata is a {type(metadata).__name__}, not an object; skipped")
        return None

    # theme is matched in any key case (theme, THEME, ...), the same as validate_theme on upload
    theme_keys = [key for key in metadata if str(key).lower() == THEME_FIELD]
    if not theme_keys and not any(field in metadata for field in settings.OMITTED_FIELDS_FROM_METADATA):
        return None

    metadata = dict(metadata)
    update = {}
    # title/summary/tags match exact keys only (metadata.TITLE is caller data and stays).
    # A value held only in metadata is backfilled, so legacy points lose nothing
    for field in settings.OMITTED_FIELDS_FROM_METADATA:
        value = metadata.pop(field, None)
        if value and not payload.get(field):
            update[field] = value

    # Every theme copy is removed; one is backfilled only when the top-level theme is empty
    theme = pick_theme_backfill(point_id, payload.get(THEME_FIELD), [metadata.pop(key) for key in theme_keys])
    if theme:
        update[THEME_FIELD] = theme

    update["metadata"] = metadata
    return update


def main():
    """Scroll the collection page by page and write a cleaned payload for every point that needs it.
    Inputs (CLI): --dry-run logs what would change without writing; --batch-size is points per scroll page."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="count affected points without writing")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    collection = settings.COLLECTION_NAME
    mode = "DRY RUN" if args.dry_run else "WRITE"
    clean_label = "would_clean" if args.dry_run else "cleaned"
    logger.info(
        f"Starting cleanup: collection={collection} mode={mode} batch_size={args.batch_size} "
        f"fields={', '.join(BACKFILL_FIELDS)}"
    )

    scanned = cleaned = failed = page = 0
    backfilled = Counter()
    offset = None
    while True:
        # Reads are paged: one scroll per batch, following offset until Qdrant returns None.
        # A failed read stops the run; points already written stay clean, so a re-run continues
        try:
            points, offset = qdrant_client.scroll(
                collection_name=collection,
                limit=args.batch_size,
                offset=offset,
                with_payload=SCROLL_FIELDS,
                with_vectors=False,
            )
        except Exception as e:
            logger.error(
                f"Scroll failed on page {page + 1} after scanning {scanned} points: {e}. "
                f"Points cleaned so far stay cleaned; re-run the script to continue."
            )
            sys.exit(1)
        page += 1

        for point in points:
            scanned += 1
            update = build_update(point.id, point.payload or {})
            if update is None:
                continue

            # set_payload overwrites only the keys in update (metadata as a whole). One call per
            # point, since each point's cleaned metadata differs; a failure is logged and skipped
            if not args.dry_run:
                try:
                    qdrant_client.set_payload(
                        collection_name=collection,
                        payload=update,
                        points=[point.id],
                    )
                except Exception as e:
                    failed += 1
                    logger.warning(f"point={point.id}: update failed: {e}")
                    continue
            cleaned += 1
            backfilled.update(field for field in update if field != "metadata")

        logger.info(f"Page {page}: scanned={scanned} {clean_label}={cleaned} failed={failed}")
        if offset is None:
            break

    already_clean = scanned - cleaned - failed
    logger.info(
        f"Done: collection={collection} mode={mode} scanned={scanned} {clean_label}={cleaned} "
        f"already_clean={already_clean} failed={failed} backfilled "
        + " ".join(f"{field}={backfilled[field]}" for field in BACKFILL_FIELDS)
    )

    # One closing line telling the operator what the run means and whether to act
    if failed:
        logger.error(f"{failed} points failed (see warnings above). Re-run the script to retry them.")
        sys.exit(1)
    if scanned == 0:
        logger.info(f"Collection {collection} is empty; nothing to do.")
    elif cleaned == 0:
        logger.info(f"Nothing to clean: all {scanned} points are already clean. Re-running is safe.")
    elif args.dry_run:
        logger.info(f"DRY RUN: nothing was written; {cleaned} points would be cleaned. Run without --dry-run to apply.")
    else:
        logger.info(f"Cleanup complete: {cleaned} points cleaned. A re-run will find nothing to clean.")


if __name__ == "__main__":
    main()
