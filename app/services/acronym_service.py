import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.constants import constants as const
from app.constants import messages as msg
from app.core.clients import cache_client
from app.core.database import SessionLocal
from app.models.db_models import AcronymMapping
from app.validations.acronym import validate_acronym_row

logger = logging.getLogger(__name__)


def _cache_key(acronym: str) -> str:
    return f"{const.ACRONYM_CACHE_KEY_PREFIX}:{acronym}"


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

    cache_enabled = settings.CACHE_ENABLED
    if not cache_enabled:
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
    # Keys holding an unusable value: deleted below, since SET NX would keep them.
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

    # Ask Postgres about every word Redis did not know, in one query; only active
    # acronyms count. On success the code below caches each answer (a meaning, or
    # "not an acronym"). If Postgres fails, search goes on without these acronyms and
    # the failure is remembered briefly (see except). The connection is always closed.
    db = SessionLocal()
    try:
        mappings = (
            db.query(AcronymMapping)
            .filter(AcronymMapping.acronym.in_(missing), AcronymMapping.is_active.is_(True))
            .all()
        )
    except Exception as e:
        # Postgres is down: the search goes on, treating the missing words as "not an acronym".
        logger.warning(
            f"Acronym batch DB lookup failed for {len(missing)} acronym(s), "
            f"treating as not found: {e}"
        )
        if cache_enabled:
            try:
                # Remember "not an acronym" for only REDIS_DB_ERROR_CACHE_TTL (30 s): searches
                # stop asking the broken database, but real acronyms come back soon after it
                # recovers. SET NX: never replace a real meaning already in Redis.
                cache_client.set_many_if_absent({
                    _cache_key(a): (json.dumps(None), settings.REDIS_DB_ERROR_CACHE_TTL)
                    for a in missing
                })
            except Exception as cache_e:
                # Redis failed too: nothing is saved, so the next search asks Postgres again.
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
    # SET NX: a value an upload wrote after our read is kept. Broken keys are
    # deleted, and the next lookup fills them again. See design notes, Cache.
    writes: Dict[str, Tuple[str, int]] = {}
    for acronym in missing:
        if acronym in found_by_acronym:
            result[acronym] = found_by_acronym[acronym]
            entry = (json.dumps(found_by_acronym[acronym]), settings.REDIS_CACHE_TTL)
        else:
            entry = (json.dumps(None), settings.REDIS_NEGATIVE_CACHE_TTL)
        if acronym not in broken:
            writes[_cache_key(acronym)] = entry

    if cache_enabled:
        try:
            cache_client.set_many_if_absent(writes)
            cache_client.delete_many([_cache_key(a) for a in broken])
        except Exception as e:
            logger.warning(
                "Acronym batch cache write-through failed for "
                f"{len(writes) + len(broken)} acronym(s): {e}"
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


def load_acronym_cache() -> bool:
    """Write every active acronym to the cache in one pipelined write.

    Called at startup (warm-up) and after each bulk upload. Overwrites on purpose, so it
    also repairs keys changed outside the upload API (migrations, SQL fixes).
    Blocking: call via run_in_threadpool. Returns False if it failed; not fatal, since
    lookups fall back to Postgres.
    """
    if not settings.CACHE_ENABLED:
        logger.info("Acronym cache load skipped (CACHE_ENABLED=false)")
        return True

    try:
        db = SessionLocal()
        try:
            mappings = db.query(AcronymMapping).filter(AcronymMapping.is_active.is_(True)).all()
        finally:
            db.close()

        writes: Dict[str, Tuple[str, int]] = {}
        for mapping in mappings:
            if not _valid_expansions(mapping.expansions):
                logger.warning(
                    f"Skipping cache load for {mapping.acronym!r}: malformed "
                    f"expansions ({mapping.expansions!r})"
                )
                continue
            writes[_cache_key(mapping.acronym)] = (
                json.dumps(mapping.expansions), settings.REDIS_CACHE_TTL
            )

        cache_client.set_many(writes)
    except Exception as e:
        logger.warning(f"Acronym cache load failed, lookups fall back to Postgres: {e}")
        return False

    logger.info(f"Acronym cache loaded: {len(writes)} active acronym(s)")
    return True


def bulk_upsert(rows: List[dict]) -> Tuple[List[str], List[str], List[str], List[dict]]:
    """Validate and upsert CSV rows in one transaction; bad rows become errors, the rest commit.

    Returns (created, updated, deactivated, errors); deactivated is a subset of updated.
    Does not touch the cache. See design notes, Bulk upload.
    """
    errors: List[dict] = []
    valid_by_acronym: dict = {}

    for index, row in enumerate(rows):
        acronym, entry, error = validate_acronym_row(index, row)
        if error:
            errors.append(error)
            continue

        if acronym in valid_by_acronym:
            errors.append({
                "index": valid_by_acronym[acronym]["index"],
                "acronym": acronym,
                "reason": msg.ACRONYM_ROW_DUPLICATE.format(index=index),
            })

        valid_by_acronym[acronym] = entry

    if not valid_by_acronym:
        return [], [], [], errors

    acronym_table = AcronymMapping.__table__
    now = datetime.now(timezone.utc)
    incoming_acronyms = list(valid_by_acronym.keys())
    to_upsert = [a for a in incoming_acronyms if valid_by_acronym[a]["is_active"]]
    to_deactivate = [a for a in incoming_acronyms if not valid_by_acronym[a]["is_active"]]
    # Batches in one transaction: Postgres allows at most 65,535 values per statement.
    batch_size = settings.ACRONYM_BULK_UPLOAD_BATCH_SIZE

    db = SessionLocal()
    try:
        # Acronym -> True if inserted, False if an existing row was changed.
        inserted: Dict[str, bool] = {}

        # Active rows: insert new acronyms, overwrite existing ones.
        for start in range(0, len(to_upsert), batch_size):
            stmt = pg_insert(acronym_table).values([
                {
                    "acronym": acronym,
                    "expansions": valid_by_acronym[acronym]["expansions"],
                    "description": valid_by_acronym[acronym]["description"],
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
                for acronym in to_upsert[start:start + batch_size]
            ])
            stmt = stmt.on_conflict_do_update(
                index_elements=["acronym"],
                set_={
                    "expansions": stmt.excluded.expansions,
                    "description": stmt.excluded.description,
                    "is_active": stmt.excluded.is_active,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            # xmax = 0 marks an inserted row; an ON CONFLICT update has this transaction's id.
            stmt = stmt.returning(
                acronym_table.c.acronym,
                sa.literal_column("xmax = 0").label("inserted"),
            )
            inserted.update({row.acronym: row.inserted for row in db.execute(stmt)})

        # Inactive rows only turn off a stored acronym; its expansions and description stay.
        for start in range(0, len(to_deactivate), batch_size):
            stmt = (
                sa.update(acronym_table)
                .where(acronym_table.c.acronym.in_(to_deactivate[start:start + batch_size]))
                .values(is_active=False, updated_at=now)
                .returning(acronym_table.c.acronym)
            )
            inserted.update({row.acronym: False for row in db.execute(stmt)})
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # An acronym the UPDATE did not find does not exist, so it cannot be deactivated.
    for acronym in to_deactivate:
        if acronym not in inserted:
            errors.append({
                "index": valid_by_acronym[acronym]["index"],
                "acronym": acronym,
                "reason": msg.ACRONYM_ROW_NOT_FOUND,
            })

    created = [a for a in incoming_acronyms if inserted.get(a)]
    updated = [a for a in incoming_acronyms if a in inserted and not inserted[a]]
    deactivated = [a for a in to_deactivate if a in inserted]
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
