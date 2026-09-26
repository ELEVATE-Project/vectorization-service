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
)
from app.core.clients import cache_client
from app.core.database import SessionLocal
from app.models.db_models import AcronymMapping

logger = logging.getLogger(__name__)


def _cache_key(acronym: str) -> str:
    return f"{ACRONYM_CACHE_KEY_PREFIX}:{acronym}"


def _valid_expansions(expansions) -> bool:
    """A DB/cache expansions value must be a non-empty list of non-empty
    strings. Guards against a malformed row (null, wrong type, empty
    strings) written outside bulk_upsert's validation — e.g. raw SQL —
    from being cached and crashing the acronym query-building code in
    prioritized_search_service.search() downstream."""
    return isinstance(expansions, list) and bool(expansions) and all(
        isinstance(e, str) and e.strip() for e in expansions
    )


def get_expansions_batch(acronyms: List[str]) -> Dict[str, List[str]]:
    """Cache-aside lookup for a list of acronyms in one call — Redis hits
    return immediately, misses fall back to Postgres and write through to
    the cache.

    Batched, not one round-trip per acronym: detect_acronyms() calls this
    once per query with every candidate token, which also collapses N
    independent connection-timeout exposures into one on a Redis outage.

    Input is deduped internally; acronyms not found (or empty) are simply
    absent from the returned dict."""
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
    for acronym, cached in zip(normalized, cached_values):
        if cached is None:
            missing.append(acronym)
            continue

        # Parse inside its own guard: the try above covers only the Redis READ,
        # so an unparseable value used to raise JSONDecodeError straight out of
        # search() as a 500 on an ordinary query. A cache entry is never worth
        # failing a request over — treat a bad one exactly like a cache miss and
        # let Postgres, the source of truth, answer instead.
        try:
            decoded = json.loads(cached)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Acronym {acronym!r} has an unparseable cached value, "
                f"falling back to Postgres: {e}"
            )
            missing.append(acronym)
            continue

        # None is the NEGATIVE cache entry ("looked this up, it isn't an
        # acronym"). Skipped WITHOUT going into `missing` — routing it to
        # Postgres would defeat the negative cache and re-query every ordinary
        # English word on every request.
        if decoded is None:
            continue

        # The same shape check the DB rows already get. Without it a cached bare
        # string survives as-is and expansions[0] downstream yields its first
        # CHARACTER, silently searching for "D" instead of the expansion; a
        # cached dict or list-of-ints fails later as a RuntimeError instead.
        if not _valid_expansions(decoded):
            logger.warning(
                f"Acronym {acronym!r} has a malformed cached value "
                f"({decoded!r}), falling back to Postgres"
            )
            missing.append(acronym)
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
        # A Postgres outage (or the acronym table not existing yet) must
        # degrade to "not found" rather than propagating — search never
        # depended on Postgres before this feature. Negative-cache with the
        # short DB-error TTL so a Postgres outage doesn't get re-attempted
        # on every request for its full duration.
        logger.warning(
            f"Acronym batch DB lookup failed for {len(missing)} acronym(s), "
            f"treating as not found: {e}"
        )
        if settings.CACHE_ENABLED:
            try:
                # if-absent for the same reason as the normal write-back below:
                # this is an absence we inferred from a FAILED lookup, which is
                # even weaker evidence than a successful "not found", so it must
                # never overwrite a real value someone else cached.
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
    # Split by what we actually learned, because the two need different write
    # rules. A FOUND value came from Postgres, the source of truth, so writing
    # it unconditionally is safe. A NOT-FOUND is only an absence, and an absence
    # can be filled in while we were querying: a bulk upload committing in that
    # window calls refresh_cache and writes the real value, and an unconditional
    # write-back then buried it under `null` for the whole negative TTL — an hour
    # in which a freshly uploaded acronym was in the database but invisible to
    # search. Negative entries therefore go through SET..NX, which makes the
    # "is it still absent" check and the write one atomic step inside Redis.
    found_writes: Dict[str, Tuple[str, int]] = {}
    absent_writes: Dict[str, Tuple[str, int]] = {}
    for acronym in missing:
        if acronym in found_by_acronym:
            result[acronym] = found_by_acronym[acronym]
            found_writes[_cache_key(acronym)] = (
                json.dumps(found_by_acronym[acronym]), settings.REDIS_CACHE_TTL
            )
        else:
            absent_writes[_cache_key(acronym)] = (
                json.dumps(None), settings.REDIS_NEGATIVE_CACHE_TTL
            )

    if settings.CACHE_ENABLED:
        try:
            cache_client.set_many(found_writes)
            cache_client.set_many_if_absent(absent_writes)
        except Exception as e:
            logger.warning(
                "Acronym batch cache write-through failed for "
                f"{len(found_writes) + len(absent_writes)} acronym(s): {e}"
            )

    return result


def invalidate_cache(acronyms: List[str]) -> bool:
    """Drop cached entries for the given acronyms in one batched round-trip.
    Call after any write to those rows so stale expansions aren't served
    until TTL expiry.

    Swallows Redis errors (logged) rather than raising — this is already the
    degraded-mode fallback when the cache is having problems (e.g. the bulk
    upload endpoint calls this when refresh_cache() itself failed), so letting
    it raise would turn an already-committed, successful write into a 500.

    Returns True when the cache is consistent with the database afterwards
    (including the nothing-to-do and cache-disabled cases), False when the
    delete failed and stale entries may still be served for up to
    REDIS_CACHE_TTL. Swallowing the error is right; staying SILENT about it was
    not — a bulk upload that deactivated 30 acronyms returned 200 with no hint
    that all 30 were still live in search for the next 24 hours."""
    if not acronyms or not settings.CACHE_ENABLED:
        return True
    keys = [_cache_key(a.strip().upper()) for a in acronyms]
    try:
        cache_client.delete_many(keys)
        return True
    except Exception as e:
        logger.warning(f"Acronym cache invalidation failed for {len(keys)} key(s): {e}")
        return False


def refresh_cache(acronyms: List[str]) -> None:
    """Re-cache expansions for exactly the given acronyms (e.g. after a bulk
    upload), instead of re-warming the entire cache via load_acronym_cache().

    Deliberately synchronous, same reasoning as load_acronym_cache() —
    callers must run this via run_in_threadpool rather than awaiting it
    directly."""
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

    # Collect first, write once. One pipelined round-trip instead of one SETEX
    # per acronym — the same batching the read side (get_expansions_batch) has
    # always used. A 500-row upload was 500 sequential round-trips.
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

    # Deliberately NOT wrapped: the bulk-upload caller catches a refresh failure
    # and invalidates the whole batch instead, so swallowing it here would hide
    # the failure and leave stale expansions cached.
    cache_client.set_many(writes)


def load_acronym_cache() -> int:
    """Pre-populate the cache-aside store with every active acronym at
    startup, so first-touch queries after boot are already cache hits.
    Not a substitute for get_expansions_batch()'s DB fallback — acronyms
    added after startup, or evicted via TTL, are still served by that path.

    Deliberately synchronous — every call inside is blocking, so async
    callers must run this via run_in_threadpool or the ~600 sequential
    Redis round-trips stall the whole event loop."""
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

    # One pipelined round-trip rather than ~550 sequential ones at boot. The
    # per-acronym try/except this replaces could only ever have salvaged a
    # partial warm-up from a mid-flight Redis failure; a pipeline rides one
    # connection either way, so the batch is all-or-nothing. The failure is
    # logged and swallowed because a cold cache must never stop the service
    # booting — get_expansions_batch still falls back to Postgres per query,
    # so a cold cache costs latency on first touch, not correctness.
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
    """Pipe-separated -> deduped list, order preserved (spec §5/§7).

    Deliberately duplicated from the identical helper in migration
    f13a664a31b6 rather than imported: Alembic migrations must stay
    self-contained snapshots, frozen at the point they were written, so a
    future change to this live-app helper can never silently alter what an
    already-applied historical migration does on re-run."""
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

# Letters only, 1-4 space-separated words — matches exactly what
# detect_acronyms() can ever actually find (single tokens and the up-to-4-word
# phrase windows in acronym_query_service.py). Anything outside this shape
# (symbols like "WI-FI", digits like "4G", a stray "/" or "-") can never be
# detected, so bulk_upsert rejects it up front instead of silently accepting
# a row that will sit in the table forever unreachable.
_ACRONYM_KEY_RE = re.compile(r"^[A-Z]+(?: [A-Z]+){0,3}$")


def _is_valid_acronym_key(acronym: str) -> bool:
    if not _ACRONYM_KEY_RE.match(acronym):
        return False
    words = acronym.split(" ")
    # A single-letter WHOLE key ("A") can never be a candidate on its own —
    # detect_acronyms drops any token under 2 letters. A single-letter word
    # INSIDE a phrase is fine ("RBI GRADE B" is a real, working entry): the
    # phrase-window path only requires each word to be non-empty.
    if len(words) == 1 and len(words[0]) < 2:
        return False
    return True


def bulk_upsert(rows: List[dict]) -> Tuple[List[str], List[str], List[str], List[dict]]:
    """Validate and upsert a batch of CSV rows (spec §7) in a single
    transaction. A row missing acronym/expansions, or an in-batch duplicate
    (Postgres' ON CONFLICT can't affect the same row twice), is recorded as
    an error and skipped rather than aborting the batch — last occurrence
    wins. Does not touch the cache — the caller owns refresh-vs-invalidate.
    Returns (created_acronyms, updated_acronyms, deactivated_acronyms, errors).
    deactivated_acronyms is a subset of created+updated (whichever rows had
    is_active=false in this batch), not a separate category.
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

        # Optional column — missing/empty defaults to active (backward-compatible
        # with every CSV that predates this column). Only "true"/"false" are
        # accepted; anything else is a per-row error rather than a silent guess.
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
            # Must be caught here, before the batch insert: an over-length
            # value reaching Postgres raises StringDataRightTruncation on the
            # single multi-row INSERT, which fails the ENTIRE batch (including
            # every otherwise-valid row) rather than just this one row.
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

    db = SessionLocal()
    try:
        # Determine create vs. update (spec §7 counts them separately) before
        # the upsert — ON CONFLICT itself doesn't tell us which branch fired
        # per row.
        existing = {
            row.acronym
            for row in db.query(AcronymMapping.acronym)
            .filter(AcronymMapping.acronym.in_(incoming_acronyms))
            .all()
        }
        created = [a for a in incoming_acronyms if a not in existing]
        updated = [a for a in incoming_acronyms if a in existing]
        deactivated = [a for a in incoming_acronyms if not valid_by_acronym[a]["is_active"]]

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
        db.execute(stmt)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return created, updated, deactivated, errors


def list_acronyms(
    prefix: Optional[str] = None,
    is_active: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[AcronymMapping], int]:
    """Read path for GET /api/acronyms (AC-16: there was no way to inspect
    the dictionary via the API at all, active or not). Queries Postgres
    directly rather than the Redis cache, which only ever holds active rows
    and was never meant to support listing/pagination — this is an
    admin/inspection endpoint, not the search hot path, so a plain query per
    call is fine.

    prefix: case-insensitive match against the START of the acronym key, so
    "RTE" matches both "RTE" and "RTE ACT". is_active=None (the default)
    returns both active and inactive rows on purpose — omitting inactive
    rows entirely was the original gap.

    Returns (rows, total) — total is the full match count ignoring
    limit/offset, for the caller to build pagination from.
    """
    db = SessionLocal()
    try:
        query = db.query(AcronymMapping)
        if is_active is not None:
            query = query.filter(AcronymMapping.is_active.is_(is_active))
        if prefix and prefix.strip():
            query = query.filter(AcronymMapping.acronym.like(f"{prefix.strip().upper()}%"))
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
