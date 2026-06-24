"""Migrate existing Qdrant documents to include BM25 sparse vectors.

TWO MODES depending on whether --new-collection is provided:

  IN-PLACE MODE (no --new-collection):
    Adds BM25 sparse vectors to points in the EXISTING collection.
    Requires the collection to already have the 'bm25' sparse vector field declared.
    Use this if the collection was created with SPARSE_SEARCH_ENABLED=true,
    or after manually adding the sparse field to a fresh/empty collection.

    Usage:
        PYTHONPATH=. COLLECTION_NAME=documents1 SPARSE_SEARCH_ENABLED=true \\
          .venv/bin/python3 scripts/migrate_to_sparse_vectors.py

  BLUE-GREEN MODE (with --new-collection):
    Use for production collections that do NOT yet have the sparse vector field.
    Qdrant's vector schema is fixed at collection creation — sparse fields cannot
    be added to an existing collection. This mode works around that by:
      1. Creating a NEW collection with the full schema (dense + BM25 sparse).
      2. Copying all points (dense vectors + payload) from the source collection.
      3. Generating and writing BM25 sparse vectors for every point.
      4. Verifying point count and BM25 spot-check before touching .env.
      5. Auto-updating COLLECTION_NAME in .env only if verification passes.
    The source collection is untouched until Step 5. Rollback at any point
    before that: reset COLLECTION_NAME in .env + restart (< 30 seconds).

    Usage:
        PYTHONPATH=. \\
        COLLECTION_NAME=documents1 \\
        QDRANT_HOST=<prod-host> \\
        QDRANT_PORT=6333 \\
          .venv/bin/python3 scripts/migrate_to_sparse_vectors.py \\
            --new-collection documents1_v2

Requirements:
    - qdrant-client[fastembed] >= 1.9.0
    - Run from the vectorization-service root (PYTHONPATH=.)
    - QDRANT_HOST / QDRANT_PORT env vars (defaults: 127.0.0.1 / 6333)
    - COLLECTION_NAME env var (default: "documents")
    - SPARSE_VECTOR_NAME env var (default: "bm25")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKGROUND — WHY THIS MIGRATION EXISTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY BM25 SPARSE VECTORS?
─────────────────────────
Dense (semantic) embeddings understand meaning well but blur exact keywords —
abbreviations, proper nouns, document codes, and IDs get lost in the vector
space. BM25 ranks by term frequency and rarity, excelling at exact matches.

    Dense Vector (Semantic)           BM25 Sparse Vector (Keyword)
    ────────────────────────          ──────────────────────────────
    "lesson plan" ≈ "study guide"    "NCERT" = exactly "NCERT"
    Good: synonyms, concepts          Good: codes, names, IDs, acronyms
    Bad:  abbreviations, exact IDs    Bad:  paraphrasing, synonyms

HYBRID SEARCH (RRF FUSION)
───────────────────────────
Both searches run simultaneously. Reciprocal Rank Fusion (RRF) merges results,
promoting documents that rank well in both — the industry-standard approach used
by Elasticsearch, Weaviate, Pinecone, and Azure AI Search.

    User Query: "NCERT Class 5 Maths lesson"
          │
          ├── Dense Search  → Finds conceptually related docs
          ├── BM25 Search   → Finds docs containing exact words "NCERT", "Class 5"
          └── RRF Fusion    → Promotes docs that rank highly in BOTH ✅

WHY A MIGRATION IS NEEDED (not just a schema update)
──────────────────────────────────────────────────────
    ⚠️  Qdrant does NOT allow adding a vector field to an existing collection.
        This is a hard architectural constraint, confirmed on Qdrant 1.18.2:

        {"status": {"error": "Wrong input: Not existing vector name error: bm25"}}

    Unlike SQL (ALTER TABLE ADD COLUMN), Qdrant's vector schema is fixed at
    collection creation time. A new collection with the correct schema must be
    created and data migrated into it — hence the blue-green mode above.
"""
import argparse
import logging
import os
import re
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EMBED_SIZE = 384          # all-MiniLM-L6-v2 output dimension
DENSE_FIELDS = ["text", "title", "summary", "tags", "metadata"]

# Allowed characters for a collection name. This is intentionally strict:
# the value is written verbatim into the .env file, which start_mac.sh sources
# as a shell script (`set -a; source .env`). Anything outside this set — newlines,
# quotes, shell metacharacters, command substitution — could inject variables or
# execute arbitrary commands when .env is sourced. The pattern also matches
# Qdrant's own collection naming constraints.
_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_collection_name(name: str) -> str:
    """Return name if it is a safe collection identifier, else raise ValueError.

    Guards the value before it is ever written to the .env file (see
    _update_env_file) — see _COLLECTION_NAME_RE for why this must be strict.
    """
    if not name or not _COLLECTION_NAME_RE.match(name):
        raise ValueError(
            f"Invalid collection name {name!r}: only letters, digits, '_' and '-' "
            "are allowed (must be non-empty, no whitespace/quotes/shell metacharacters)."
        )
    return name


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant documents to include BM25 sparse vectors. See module docstring for full background.",
    )
    parser.add_argument(
        "--new-collection",
        metavar="NAME",
        default=None,
        help=(
            "Target collection name for blue-green migration. "
            "When omitted, performs an in-place update on the existing collection."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing any changes.")
    parser.add_argument("--batch-size", type=int, default=100, help="Points per upsert/update batch (default 100).")
    parser.add_argument("--scroll-limit", type=int, default=500, help="Points per scroll page (default 500).")
    # Blue-green only
    parser.add_argument("--skip-copy", action="store_true", help="[Blue-green] Skip copy step (target already populated).")
    parser.add_argument("--skip-bm25", action="store_true", help="[Blue-green] Skip BM25 encoding step.")
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        default=".env",
        help="Path to the .env file to auto-update after successful blue-green migration (default: .env).",
    )
    return parser.parse_args()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _flush_update_vectors(client, collection_name: str, pending: list, dry_run: bool) -> tuple[int, int]:
    """Flush a batch of PointVectors via update_vectors. Returns (migrated, errors)."""
    if not pending:
        return 0, 0
    if dry_run:
        logger.info(f"  [DRY RUN] Would update {len(pending)} points")
        del pending[:]
        return 0, 0
    migrated = errors = 0
    try:
        client.update_vectors(collection_name=collection_name, points=pending)
        migrated = len(pending)
    except Exception as exc:
        logger.error(f"update_vectors batch failed: {exc}")
        errors = len(pending)
    del pending[:]
    return migrated, errors


def _encode_and_queue(generate_sparse_vector, SparseVector, PointVectors,
                      point, sparse_name: str, pending: list) -> str:
    """
    Generate BM25 vector for a point and append to pending list.
    Returns 'queued', 'skipped', or 'error'.
    """
    text = (point.payload or {}).get("text", "")
    if not text:
        return "skipped"
    try:
        indices, values = generate_sparse_vector(text)
    except Exception as exc:
        logger.warning(f"BM25 encoding failed for point {point.id}: {exc}")
        return "error"
    if not indices:
        return "skipped"
    pending.append(
        PointVectors(
            id=point.id,
            vector={sparse_name: SparseVector(indices=indices, values=values)},
        )
    )
    return "queued"


# ── In-place mode ─────────────────────────────────────────────────────────────

def run_inplace(client, collection_name: str, sparse_name: str, args,
                generate_sparse_vector, SparseVector, PointVectors) -> None:
    """Add BM25 sparse vectors to points in an existing collection (in-place)."""
    logger.info("MODE: In-place — adding BM25 sparse vectors to existing collection.")
    logger.info(f"  Collection : {collection_name}")
    logger.info(f"  Sparse name: {sparse_name}")
    if args.dry_run:
        logger.info("  DRY RUN — no changes will be written.")

    total_points = client.count(collection_name).count
    logger.info(f"  Total points: {total_points}")

    scanned = migrated = skipped = errors = 0
    offset = None
    pending: list = []
    start = time.monotonic()

    while True:
        scroll_kwargs = dict(
            collection_name=collection_name,
            limit=args.scroll_limit,
            with_payload=["text"],
            with_vectors=[sparse_name],
        )
        if offset is not None:
            scroll_kwargs["offset"] = offset

        try:
            points, next_offset = client.scroll(**scroll_kwargs)
        except Exception as exc:
            logger.error(f"Scroll failed: {exc}")
            break

        for point in points:
            scanned += 1

            # Skip points that already have the sparse vector
            existing = (getattr(point, "vector", {}) or {}).get(sparse_name)
            if existing and getattr(existing, "indices", None):
                skipped += 1
                continue

            result = _encode_and_queue(
                generate_sparse_vector, SparseVector, PointVectors,
                point, sparse_name, pending,
            )
            if result == "error":
                errors += 1
            elif result == "skipped":
                skipped += 1

            if len(pending) >= args.batch_size:
                m, e = _flush_update_vectors(client, collection_name, pending, args.dry_run)
                migrated += m
                errors += e

        if scanned % 1000 == 0 and scanned > 0:
            elapsed = time.monotonic() - start
            logger.info(
                f"  Progress: scanned={scanned}/{total_points}, migrated={migrated}, "
                f"skipped={skipped}, errors={errors}, elapsed={elapsed:.1f}s"
            )

        if not next_offset:
            break
        offset = next_offset

    m, e = _flush_update_vectors(client, collection_name, pending, args.dry_run)
    migrated += m
    errors += e

    elapsed = time.monotonic() - start
    logger.info(
        f"In-place migration complete in {elapsed:.1f}s — "
        f"scanned={scanned}, migrated={migrated}, skipped={skipped}, errors={errors}"
    )
    if errors:
        logger.warning(f"{errors} encoding errors. Re-run to retry (script is idempotent).")
        sys.exit(1)


# ── Blue-green mode ───────────────────────────────────────────────────────────

def _verify_migration(client, old_col: str, new_col: str, sparse_name: str) -> bool:
    """Deep verification of the blue-green migration.

    Checks:
      1. Point count in new collection matches old collection.
      2. A sample of points in new collection have a non-empty BM25 sparse vector.

    Returns True if all checks pass, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Verification")
    passed = True

    # Check 1: Point count
    try:
        old_count = client.count(old_col).count
        new_count = client.count(new_col).count
        logger.info(f"  Point count — source '{old_col}': {old_count}, target '{new_col}': {new_count}")
        if new_count == old_count:
            logger.info(f"  ✅ Count check passed: {new_count} == {old_count}")
        else:
            logger.error(f"  ❌ Count mismatch: expected {old_count}, got {new_count}")
            passed = False
    except Exception as exc:
        logger.error(f"  ❌ Could not verify point count: {exc}")
        passed = False

    # Check 2: BM25 spot-check — sample up to 5 points
    logger.info("  Spot-checking BM25 sparse vectors on sample points...")
    try:
        sample_points, _ = client.scroll(
            collection_name=new_col,
            limit=5,
            with_payload=["text"],
            with_vectors=[sparse_name],
        )
        bm25_ok = 0
        bm25_missing = 0
        bm25_no_text = 0
        for p in sample_points:
            text = (p.payload or {}).get("text", "")
            if not text:
                bm25_no_text += 1
                continue
            sv = (getattr(p, "vector", {}) or {}).get(sparse_name)
            if sv and getattr(sv, "indices", None):
                bm25_ok += 1
            else:
                bm25_missing += 1
                logger.warning(f"  ⚠️  Point {p.id} has text but missing BM25 vector")
        logger.info(
            f"  Sample BM25 check ({len(sample_points)} points): "
            f"ok={bm25_ok}, missing={bm25_missing}, no_text={bm25_no_text}"
        )
        if bm25_missing > 0:
            logger.error("  ❌ BM25 spot-check failed — some points are missing sparse vectors.")
            passed = False
        else:
            logger.info("  ✅ BM25 spot-check passed")
    except Exception as exc:
        logger.error(f"  ❌ BM25 spot-check error: {exc}")
        passed = False

    return passed


def _update_env_file(env_path: str, old_col: str, new_col: str) -> bool:
    """Rewrite COLLECTION_NAME in the .env file to new_col.

    Returns True on success, False on any error.
    """
    # Defense-in-depth: never write an unvalidated value into .env, regardless of
    # caller. .env is sourced by start_mac.sh, so an unsafe value is code execution.
    try:
        _validate_collection_name(new_col)
    except ValueError as exc:
        logger.error(f"  ❌ Refusing to update .env: {exc}")
        return False
    if not os.path.isfile(env_path):
        logger.warning(f"  .env file not found at '{env_path}' — skipping auto-update.")
        return False
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
        updated = False
        new_lines = []
        for line in lines:
            if line.startswith("COLLECTION_NAME="):
                new_lines.append(f"COLLECTION_NAME={new_col}\n")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            # Key not present — append it
            new_lines.append(f"COLLECTION_NAME={new_col}\n")
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        logger.info(f"  ✅ .env updated: COLLECTION_NAME={new_col} (was {old_col})")
        return True
    except Exception as exc:
        logger.error(f"  ❌ Failed to update .env: {exc}")
        return False


def run_bluegreen(client, old_col: str, new_col: str, sparse_name: str, args,
                  generate_sparse_vector, SparseVector, PointVectors, PointStruct,
                  VectorParams, Distance, SparseVectorParams, Modifier,
                  env_path: str = ".env") -> None:
    """Copy collection to a new name, then add BM25 sparse vectors (blue-green)."""
    logger.info("MODE: Blue-green — migrating to a new collection with BM25 sparse vectors.")
    logger.info(f"  Source : {old_col}")
    logger.info(f"  Target : {new_col}")
    logger.info(f"  Sparse : {sparse_name}")
    if args.dry_run:
        logger.info("  DRY RUN — no changes will be written.")

    try:
        total_points = client.count(old_col).count
        logger.info(f"  Source has {total_points} points.")
    except Exception as exc:
        logger.error(f"Cannot read source collection '{old_col}': {exc}")
        sys.exit(1)

    # Step 1: Create target collection
    existing_cols = [c.name for c in client.get_collections().collections]
    if new_col in existing_cols:
        logger.info(f"Target '{new_col}' already exists — skipping creation.")
    elif not args.dry_run:
        logger.info(f"Creating target collection '{new_col}'...")
        client.create_collection(
            collection_name=new_col,
            vectors_config={
                field: VectorParams(size=EMBED_SIZE, distance=Distance.COSINE)
                for field in DENSE_FIELDS
            },
            sparse_vectors_config={
                sparse_name: SparseVectorParams(modifier=Modifier.IDF)
            },
        )
        logger.info(f"Created '{new_col}' with dense={DENSE_FIELDS} + sparse=[{sparse_name}]")
    else:
        logger.info(f"[DRY RUN] Would create target collection '{new_col}'")

    # Step 2: Copy dense vectors + payload
    if not args.skip_copy:
        logger.info("=" * 60)
        logger.info("STEP 2: Copying dense vectors + payload")
        copied = 0
        offset = None
        start = time.monotonic()

        while True:
            scroll_kwargs = dict(
                collection_name=old_col,
                limit=args.scroll_limit,
                with_payload=True,
                with_vectors=True,
            )
            if offset is not None:
                scroll_kwargs["offset"] = offset

            try:
                points, next_offset = client.scroll(**scroll_kwargs)
            except Exception as exc:
                logger.error(f"Scroll failed: {exc}")
                sys.exit(1)

            if not points:
                break

            if not args.dry_run:
                structs = [
                    PointStruct(id=p.id, payload=p.payload, vector=p.vector)
                    for p in points
                ]
                try:
                    client.upsert(collection_name=new_col, points=structs)
                    copied += len(structs)
                except Exception as exc:
                    logger.error(f"Upsert batch failed: {exc}")
            else:
                copied += len(points)

            pct = copied / max(total_points, 1) * 100
            elapsed = time.monotonic() - start
            logger.info(f"  Copied {copied}/{total_points} ({pct:.1f}%) — {elapsed:.1f}s")

            if not next_offset:
                break
            offset = next_offset

        logger.info(f"Copy complete: {copied} points transferred.")
    else:
        logger.info("Skipping copy step (--skip-copy).")

    # Step 3: Generate and write BM25 sparse vectors
    if not args.skip_bm25:
        logger.info("=" * 60)
        logger.info("STEP 3: Generating BM25 sparse vectors on target collection")
        migrated = skipped = errors = 0
        offset = None
        pending: list = []
        start = time.monotonic()

        while True:
            scroll_kwargs = dict(
                collection_name=new_col,
                limit=args.scroll_limit,
                with_payload=["text"],
                with_vectors=[sparse_name],
            )
            if offset is not None:
                scroll_kwargs["offset"] = offset

            try:
                points, next_offset = client.scroll(**scroll_kwargs)
            except Exception as exc:
                logger.error(f"Scroll on target failed: {exc}")
                break

            for point in points:
                # Skip already-encoded points (idempotent)
                existing = (getattr(point, "vector", {}) or {}).get(sparse_name)
                if existing and getattr(existing, "indices", None):
                    skipped += 1
                    continue

                result = _encode_and_queue(
                    generate_sparse_vector, SparseVector, PointVectors,
                    point, sparse_name, pending,
                )
                if result == "error":
                    errors += 1
                elif result == "skipped":
                    skipped += 1

                if len(pending) >= args.batch_size:
                    m, e = _flush_update_vectors(client, new_col, pending, args.dry_run)
                    migrated += m
                    errors += e

            total_done = migrated + skipped + errors
            pct = total_done / max(total_points, 1) * 100
            elapsed = time.monotonic() - start
            logger.info(
                f"  BM25: {total_done}/{total_points} ({pct:.1f}%) | "
                f"migrated={migrated} skipped={skipped} errors={errors} — {elapsed:.1f}s"
            )

            if not next_offset:
                break
            offset = next_offset

        m, e = _flush_update_vectors(client, new_col, pending, args.dry_run)
        migrated += m
        errors += e

        elapsed = time.monotonic() - start
        logger.info(
            f"BM25 encoding complete in {elapsed:.1f}s — "
            f"migrated={migrated}, skipped={skipped}, errors={errors}"
        )
        if errors:
            logger.warning(f"{errors} errors. Re-run with --skip-copy to retry BM25 only (idempotent).")
    else:
        logger.info("Skipping BM25 step (--skip-bm25).")

    # Step 4: Verify + auto-update .env
    if args.dry_run:
        logger.info("[DRY RUN] Skipping verification and .env update.")
        return

    migration_ok = _verify_migration(client, old_col, new_col, sparse_name)

    host = os.getenv("QDRANT_HOST", "127.0.0.1")
    port = os.getenv("QDRANT_PORT", "6333")

    if migration_ok:
        logger.info("=" * 60)
        logger.info("✅ MIGRATION SUCCESSFUL")
        logger.info("")
        # Auto-update .env
        logger.info(f"Updating .env file at '{env_path}'...")
        env_updated = _update_env_file(env_path, old_col, new_col)

        logger.info("=" * 60)
        logger.warning("⚠️  ACTION REQUIRED — PLEASE READ:")
        logger.warning(f"  The new collection '{new_col}' is ready and contains all migrated data.")
        if env_updated:
            logger.warning(f"  ✅ .env has been automatically updated: COLLECTION_NAME={new_col}")
        else:
            logger.warning(f"  ❌ .env could NOT be updated automatically.")
            logger.warning(f"     Manually set COLLECTION_NAME={new_col} in your .env / prod config.")
        logger.warning("  ⚠️  YOU MUST RESTART the vectorization-service for changes to take effect.")
        logger.warning(f"  After confirming search works, delete the old collection with:")
        logger.warning(f"     curl -X DELETE http://{host}:{port}/collections/{old_col}")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.error("❌ MIGRATION FAILED VERIFICATION")
        logger.error("   The new collection has issues. Do NOT update your .env yet.")
        logger.error("   Re-run the script with --skip-copy to retry the BM25 encoding step:")
        logger.error(f"     PYTHONPATH=. COLLECTION_NAME={old_col} .venv/bin/python3 scripts/migrate_to_sparse_vectors.py --new-collection {new_col} --skip-copy")
        logger.info("=" * 60)
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Validate the target collection name before doing anything: it is written
    # verbatim into the .env file on success, which start_mac.sh sources as shell.
    if args.new_collection is not None:
        try:
            _validate_collection_name(args.new_collection)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(2)

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams,
            SparseVectorParams, Modifier,
            SparseVector, PointVectors, PointStruct,
        )
    except ImportError as exc:
        logger.error(f"qdrant-client import failed: {exc}. Install qdrant-client[fastembed]>=1.9.0.")
        sys.exit(1)

    try:
        from app.core.clients.sparse_encoder import generate_sparse_vector
    except ImportError as exc:
        logger.error(f"sparse_encoder import failed: {exc}. Run with PYTHONPATH=. from the vectorization-service root.")
        sys.exit(1)

    host = os.getenv("QDRANT_HOST", "127.0.0.1")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("COLLECTION_NAME", "documents")
    sparse_name = os.getenv("SPARSE_VECTOR_NAME", "bm25")

    client = QdrantClient(host=host, port=port)
    logger.info(f"Connected to Qdrant at {host}:{port}")

    if args.new_collection:
        run_bluegreen(
            client=client,
            old_col=collection,
            new_col=args.new_collection,
            sparse_name=sparse_name,
            args=args,
            generate_sparse_vector=generate_sparse_vector,
            SparseVector=SparseVector,
            PointVectors=PointVectors,
            PointStruct=PointStruct,
            VectorParams=VectorParams,
            Distance=Distance,
            SparseVectorParams=SparseVectorParams,
            Modifier=Modifier,
            env_path=args.env_file,
        )
    else:
        run_inplace(
            client=client,
            collection_name=collection,
            sparse_name=sparse_name,
            args=args,
            generate_sparse_vector=generate_sparse_vector,
            SparseVector=SparseVector,
            PointVectors=PointVectors,
        )


if __name__ == "__main__":
    main()
