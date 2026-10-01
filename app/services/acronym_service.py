import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.constants import (
    ACRONYM_CACHE_KEY_PREFIX,
    ACRONYM_CSV_COLUMN_ACRONYM,
    ACRONYM_CSV_COLUMN_DESCRIPTION,
    ACRONYM_CSV_COLUMN_EXPANSIONS,
    ACRONYM_CSV_COLUMN_IS_ACTIVE,
    ACRONYM_KEY_WORD_PATTERN,
)
from app.core.clients import cache_client
from app.core.database import SessionLocal
from app.models.db_models import AcronymMapping

logger = logging.getLogger(__name__)


def _cache_key(acronym: str) -> str:
    return f"{ACRONYM_CACHE_KEY_PREFIX}:{acronym}"


def _valid_expansions(expansions) -> bool:
    """True for a non-empty list of non-empty strings, the only usable expansions value."""
    return isinstance(expansions, list) and bool(expansions) and all(
        isinstance(e, str) and e.strip() for e in expansions
    )


def get_expansions_batch(acronyms: List[str]) -> Dict[str, List[str]]:
    """Cache-aside lookup for many acronyms at once: one Redis read, Postgres for misses.

    Input is deduped; acronyms not found are absent from the result. See design notes, Cache.
    """
    normalized = list(dict.fromkeys(a.strip().upper() for a in acronyms if a and a.strip()))
    if not normalized:
        return {}

    if not settings.CACHE_ENABLED:
        cached_values = [None] * len(normalized)
    else:
        keys = [_cache_key(a) for a in normalized]
        try:
            cached_values = cache_client.get_many(keys)
        except Exception as e:
            logger.warning(
                f"Acronym batch cache read failed for {len(normalized)} acronym(s), "
                f"falling back to Postgres: {e}"
            )
            cached_values = [None] * len(normalized)

    result: Dict[str, List[str]] = {}
    missing: List[str] = []
    # Keys holding an unusable value: overwritten below, since SET NX would keep them.
    broken: set = set()
    for acronym, cached in zip(normalized, cached_values):
        if cached is None:
            missing.append(acronym)
            continue

        # An unparseable value is a miss, not a 500; Postgres answers instead.
        try:
            decoded = json.loads(cached)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Acronym {acronym!r} has an unparseable cached value, "
                f"falling back to Postgres: {e}"
            )
            missing.append(acronym)
            broken.add(acronym)
            continue

        # Cached null = "not an acronym": skip it, don't send it to Postgres.
        if decoded is None:
            continue

        # Same shape check as DB rows: a cached bare string would make
        # expansions[0] its first character.
        if not _valid_expansions(decoded):
            logger.warning(
                f"Acronym {acronym!r} has a malformed cached value "
                f"({decoded!r}), falling back to Postgres"
            )
            missing.append(acronym)
            broken.add(acronym)
            continue

        result[acronym] = decoded

    if not missing:
        return result

    db = SessionLocal()
    try:
        mappings = (
            db.query(AcronymMapping)
            .filter(AcronymMapping.acronym.in_(missing), AcronymMapping.is_active.is_(True))
            .all()
        )
    except Exception as e:
        # A Postgres outage degrades to "not found", cached briefly so it isn't
        # retried on every request.
        logger.warning(
            f"Acronym batch DB lookup failed for {len(missing)} acronym(s), "
            f"treating as not found: {e}"
        )
        if settings.CACHE_ENABLED:
            try:
                # SET NX: an absence inferred from a failed lookup must not
                # overwrite a real cached value.
                cache_client.set_many_if_absent({
                    _cache_key(a): (json.dumps(None), settings.REDIS_DB_ERROR_CACHE_TTL)
                    for a in missing
                })
            except Exception as cache_e:
                logger.warning(f"Acronym batch DB-error negative-cache write failed: {cache_e}")
        return result
    finally:
        db.close()

    found_by_acronym = {}
    for mapping in mappings:
        if _valid_expansions(mapping.expansions):
            found_by_acronym[mapping.acronym] = mapping.expansions
        else:
            logger.warning(
                f"Acronym {mapping.acronym!r} has malformed expansions in DB "
                f"({mapping.expansions!r}), treating as not found"
            )
    # SET NX so an upload that committed after our read wins; broken keys are
    # overwritten. See release-doc/acronym-design-notes.md, Cache.
    if_absent_writes: Dict[str, Tuple[str, int]] = {}
    overwrite_writes: Dict[str, Tuple[str, int]] = {}
    for acronym in missing:
        if acronym in found_by_acronym:
            result[acronym] = found_by_acronym[acronym]
            entry = (json.dumps(found_by_acronym[acronym]), settings.REDIS_CACHE_TTL)
        else:
            entry = (json.dumps(None), settings.REDIS_NEGATIVE_CACHE_TTL)
        target = overwrite_writes if acronym in broken else if_absent_writes
        target[_cache_key(acronym)] = entry

    if settings.CACHE_ENABLED:
        try:
            cache_client.set_many_if_absent(if_absent_writes)
            cache_client.set_many(overwrite_writes)
        except Exception as e:
            logger.warning(
                "Acronym batch cache write-through failed for "
                f"{len(if_absent_writes) + len(overwrite_writes)} acronym(s): {e}"
            )

    return result


def invalidate_cache(acronyms: List[str]) -> bool:
    """Delete the cached entries for `acronyms` in one round-trip.

    Swallows Redis errors; returns False if stale entries may still be served.
    """
    if not acronyms or not settings.CACHE_ENABLED:
        return True
    keys = [_cache_key(a.strip().upper()) for a in acronyms]
    try:
        cache_client.delete_many(keys)
        return True
    except Exception as e:
        logger.warning(f"Acronym cache invalidation failed for {len(keys)} key(s): {e}")
        return False


# See mark_deactivated_in_cache: must outlast an in-flight lookup, and nothing
# more.
_DEACTIVATION_MARKER_TTL = 30


def _active_acronyms(acronyms: List[str]) -> set:
    """Which of `acronyms` are active in Postgres right now. Raises on a DB
    error; the caller decides what that means."""
    db = SessionLocal()
    try:
        rows = (
            db.query(AcronymMapping.acronym)
            .filter(AcronymMapping.acronym.in_(acronyms), AcronymMapping.is_active.is_(True))
            .all()
        )
        return {row.acronym for row in rows}
    finally:
        db.close()


def mark_deactivated_in_cache(acronyms: List[str]) -> bool:
    """Write a short-lived "not an acronym" marker for acronyms Postgres shows inactive.

    Blocks a late lookup from restoring them; deletes the keys if the re-read fails.
    Returns False on a Redis error. See design notes, Cache.
    """
    if not acronyms or not settings.CACHE_ENABLED:
        return True
    normalized = list(dict.fromkeys(a.strip().upper() for a in acronyms))
    try:
        still_active = _active_acronyms(normalized)
    except Exception as e:
        logger.warning(
            f"Acronym deactivation re-check failed for {len(normalized)} acronym(s), "
            f"invalidating instead: {e}"
        )
        return invalidate_cache(normalized)
    writes = {
        _cache_key(a): (json.dumps(None), _DEACTIVATION_MARKER_TTL)
        for a in normalized if a not in still_active
    }
    try:
        cache_client.set_many(writes)
        return True
    except Exception as e:
        logger.warning(f"Acronym cache deactivation marker failed for {len(writes)} key(s): {e}")
        return False


def refresh_cache(acronyms: List[str]) -> None:
    """Re-cache the given active acronyms in one pipelined write (after a bulk upload).

    Blocking: call via run_in_threadpool. Raises on Redis errors; the caller handles them.
    """
    if not acronyms or not settings.CACHE_ENABLED:
        return

    db = SessionLocal()
    try:
        mappings = (
            db.query(AcronymMapping)
            .filter(AcronymMapping.acronym.in_(acronyms), AcronymMapping.is_active.is_(True))
            .all()
        )
    finally:
        db.close()

    # One pipelined write instead of one round-trip per acronym.
    writes: Dict[str, Tuple[str, int]] = {}
    for mapping in mappings:
        if not _valid_expansions(mapping.expansions):
            logger.warning(
                f"Skipping cache refresh for {mapping.acronym!r}: malformed "
                f"expansions ({mapping.expansions!r})"
            )
            continue
        writes[_cache_key(mapping.acronym)] = (
            json.dumps(mapping.expansions), settings.REDIS_CACHE_TTL
        )

    # Not wrapped on purpose: the upload endpoint handles a refresh failure.
    cache_client.set_many(writes)


def load_acronym_cache() -> int:
    """Warm the cache with every active acronym at startup, in one pipelined write.

    Blocking: call via run_in_threadpool. Returns the count, or 0 if the warm-up failed.
    """
    if not settings.CACHE_ENABLED:
        logger.info("Acronym cache warm-up skipped (CACHE_ENABLED=false)")
        return 0

    db = SessionLocal()
    try:
        mappings = db.query(AcronymMapping).filter(AcronymMapping.is_active.is_(True)).all()
    finally:
        db.close()

    writes: Dict[str, Tuple[str, int]] = {}
    for mapping in mappings:
        if not _valid_expansions(mapping.expansions):
            logger.warning(
                f"Skipping cache warm for {mapping.acronym!r}: malformed "
                f"expansions ({mapping.expansions!r})"
            )
            continue
        writes[_cache_key(mapping.acronym)] = (
            json.dumps(mapping.expansions), settings.REDIS_CACHE_TTL
        )

    # One pipelined write, all or nothing. A failure must not stop the boot: a cold
    # cache only costs latency, since lookups fall back to Postgres.
    try:
        cache_client.set_many(writes)
    except Exception as e:
        logger.warning(
            f"Acronym cache warm-up failed for {len(writes)} acronym(s), "
            f"continuing with a cold cache: {e}"
        )
        return 0

    logger.info(f"Acronym cache warmed: {len(writes)} active acronym(s)")
    return len(writes)


def _split_expansions(raw: str) -> List[str]:
    """Pipe-separated text to a deduped list, order kept.

    Duplicated in migration f13a664a31b6 on purpose: migrations stay frozen snapshots.
    """
    seen = set()
    result = []
    for part in raw.split("|"):
        expansion = part.strip()
        if expansion and expansion not in seen:
            seen.add(expansion)
            result.append(expansion)
    return result


# Read off the model rather than hardcoded, so this can't silently drift out
# of sync if the column length is ever changed there.
_ACRONYM_MAX_LENGTH = AcronymMapping.__table__.c.acronym.type.length

# Letters only, up to ACRONYM_MAX_PHRASE_WORDS words: the only shape detection can
# find, so anything else ("WI-FI", "4G") is rejected at upload.
_ACRONYM_KEY_RE = re.compile(
    rf"^{ACRONYM_KEY_WORD_PATTERN}(?: {ACRONYM_KEY_WORD_PATTERN}){{0,{settings.ACRONYM_MAX_PHRASE_WORDS - 1}}}$"
)


def _is_valid_acronym_key(acronym: str) -> bool:
    if not _ACRONYM_KEY_RE.match(acronym):
        return False
    words = acronym.split(" ")
    # Detection skips single letters, so "A" alone can never match; a one-letter
    # word inside a phrase ("RBI GRADE B") is fine.
    if len(words) == 1 and len(words[0]) < 2:
        return False
    return True


def bulk_upsert(rows: List[dict]) -> Tuple[List[str], List[str], List[str], List[dict]]:
    """Validate and upsert CSV rows in one transaction; bad rows become errors, the rest commit.

    Returns (created, updated, deactivated, errors); deactivated is a subset of created + updated.
    Does not touch the cache. See design notes, Bulk upload.
    """
    errors: List[dict] = []
    valid_by_acronym: dict = {}

    for index, row in enumerate(rows):
        raw_acronym = (row.get(ACRONYM_CSV_COLUMN_ACRONYM) or "").strip()
        acronym = raw_acronym.upper()
        expansions = _split_expansions(row.get(ACRONYM_CSV_COLUMN_EXPANSIONS) or "")
        description = (row.get(ACRONYM_CSV_COLUMN_DESCRIPTION) or "").strip() or None
        status = (row.get(ACRONYM_CSV_COLUMN_IS_ACTIVE) or "").strip().lower()

        if not acronym or not expansions:
            errors.append({
                "index": index,
                "acronym": raw_acronym or None,
                "reason": "acronym and expansions must be non-empty after trimming whitespace",
            })
            continue

        if not _is_valid_acronym_key(acronym):
            errors.append({
                "index": index,
                "acronym": acronym,
                "reason": (
                    "acronym must be letters only (1-4 space-separated words; "
                    "a lone word needs at least 2 letters) — this can never be "
                    "detected in a search query otherwise"
                ),
            })
            continue

        # Optional: missing means active; anything but true/false is a row error.
        if not status:
            is_active = True
        elif status == "true":
            is_active = True
        elif status == "false":
            is_active = False
        else:
            errors.append({
                "index": index,
                "acronym": acronym,
                "reason": f"is_active must be 'true' or 'false' if present, got {status!r}",
            })
            continue

        if len(acronym) > _ACRONYM_MAX_LENGTH:
            # Reject here: an over-length value would fail the whole multi-row INSERT.
            errors.append({
                "index": index,
                "acronym": acronym,
                "reason": f"acronym exceeds max length of {_ACRONYM_MAX_LENGTH} characters",
            })
            continue

        if acronym in valid_by_acronym:
            errors.append({
                "index": valid_by_acronym[acronym]["index"],
                "acronym": acronym,
                "reason": f"duplicate acronym in batch, superseded by row {index}",
            })

        valid_by_acronym[acronym] = {
            "index": index,
            "expansions": expansions,
            "description": description,
            "is_active": is_active,
        }

    if not valid_by_acronym:
        return [], [], [], errors

    acronym_table = sa.table(
        "acronym_mapping",
        sa.column("acronym", sa.String),
        sa.column("expansions", JSONB),
        sa.column("description", sa.Text),
        sa.column("is_active", sa.Boolean),
        sa.column("created_at", sa.DateTime),
        sa.column("updated_at", sa.DateTime),
    )
    now = datetime.now(timezone.utc)
    incoming_acronyms = list(valid_by_acronym.keys())

    deactivated = [a for a in incoming_acronyms if not valid_by_acronym[a]["is_active"]]

    db = SessionLocal()
    try:
        values = [
            {
                "acronym": acronym,
                "expansions": v["expansions"],
                "description": v["description"],
                "is_active": v["is_active"],
                "created_at": now,
                "updated_at": now,
            }
            for acronym, v in valid_by_acronym.items()
        ]
        stmt = pg_insert(acronym_table).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["acronym"],
            set_={
                "expansions": stmt.excluded.expansions,
                "description": stmt.excluded.description,
                "is_active": stmt.excluded.is_active,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        # One call: RETURNING says which rows were new. An inserted row has xmax = 0;
        # a row updated by ON CONFLICT carries this transaction's id. Timestamps
        # can't tell: two uploads may share the same `now`.
        stmt = stmt.returning(
            acronym_table.c.acronym,
            sa.literal_column("xmax = 0").label("inserted"),
        )
        inserted = {row.acronym: row.inserted for row in db.execute(stmt)}
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    created = [a for a in incoming_acronyms if inserted.get(a)]
    updated = [a for a in incoming_acronyms if a in inserted and not inserted[a]]
    return created, updated, deactivated, errors


def list_acronyms(
    prefix: Optional[str] = None,
    is_active: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[AcronymMapping], int]:
    """Rows for GET /api/acronyms from Postgres (the cache holds only active rows).

    `prefix` matches the start of the key; is_active=None returns all. Returns (rows, total).
    """
    db = SessionLocal()
    try:
        query = db.query(AcronymMapping)
        if is_active is not None:
            query = query.filter(AcronymMapping.is_active.is_(is_active))
        if prefix and prefix.strip():
            # Escape LIKE's own wildcards (%, _) so a prefix containing them is
            # matched as a literal string, not as a broader pattern.
            escaped_prefix = (
                prefix.strip().upper()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            query = query.filter(
                AcronymMapping.acronym.like(f"{escaped_prefix}%", escape="\\")
            )
        total = query.count()
        rows = (
            query.order_by(AcronymMapping.acronym)
            .offset(offset)
            .limit(limit)
            .all()
        )
        return rows, total
    finally:
        db.close()
