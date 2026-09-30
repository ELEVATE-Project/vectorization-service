import csv
import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import verify_admin_token, verify_internal_token
from app.config import settings
from app.constants import ACRONYM_CSV_COLUMN_ACRONYM, ACRONYM_CSV_COLUMN_EXPANSIONS
from app.models.api_models import AcronymBulkUploadResponse, AcronymItem, AcronymListResponse
from app.services.acronym_service import (
    bulk_upsert,
    invalidate_cache,
    list_acronyms,
    mark_deactivated_in_cache,
    refresh_cache,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=AcronymListResponse,
    dependencies=[Depends(verify_internal_token)],
)
async def get_acronyms(
    prefix: Optional[str] = Query(None, description="Case-insensitive match against the start of the acronym key"),
    is_active: Optional[bool] = Query(None, description="Filter by active status; omit to include both"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Read-only: list the acronym dictionary, active and inactive rows both
    by default (AC-16 — there was previously no way to inspect it via the
    API at all). Gated by internal_access_token only, not admin_auth_token
    too — reading the dictionary is lower-risk than writing to it.
    """
    rows, total = await run_in_threadpool(
        list_acronyms, prefix=prefix, is_active=is_active, limit=limit, offset=offset
    )
    return AcronymListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[
            AcronymItem(
                acronym=row.acronym,
                expansions=row.expansions,
                description=row.description,
                is_active=row.is_active,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ],
    )


@router.post(
    "/bulk",
    response_model=AcronymBulkUploadResponse,
    dependencies=[Depends(verify_internal_token), Depends(verify_admin_token)],
)
async def bulk_upload_acronyms(file: UploadFile = File(...)):
    """Internal-only: upsert acronym -> expansions rows from a CSV upload (spec
    §7). Columns: acronym, expansions (pipe-separated), description (optional),
    is_active (optional, "true"/"false" — defaults to active if omitted).
    A bad row is reported in `errors`, not a batch failure — the rest commits.
    """
    max_bytes = settings.ACRONYM_BULK_UPLOAD_MAX_SIZE_MB * 1024 * 1024
    # One byte past the limit is enough to know it's too big, without pulling
    # an oversized upload into worker memory first.
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.ACRONYM_BULK_UPLOAD_MAX_SIZE_MB}MB",
        )

    try:
        # utf-8-sig strips a leading BOM if present (common in CSVs exported
        # from Excel/Sheets) and is otherwise identical to plain utf-8 — a
        # BOM left in place would silently become part of the first header
        # name ("﻿acronym"), making every row fail acronym validation.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(text))
    required_columns = {ACRONYM_CSV_COLUMN_ACRONYM, ACRONYM_CSV_COLUMN_EXPANSIONS}
    if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have columns {sorted(required_columns)}, found {reader.fieldnames}",
        )

    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file has no data rows")

    # bulk_upsert/refresh_cache/invalidate_cache are all synchronous, blocking
    # Postgres/Redis I/O — run_in_threadpool keeps them off the event loop.
    created, updated, deactivated, errors = await run_in_threadpool(bulk_upsert, rows)

    # Spec §7: commit first (bulk_upsert already did), then refresh the cache.
    # deactivated rows go straight to mark_deactivated_in_cache (no DB round-trip
    # needed, bulk_upsert already knows their status) — refresh_cache's own query
    # filters to is_active=true, so it would silently skip them and leave their
    # old cached expansion in place. A `null` marker rather than a delete, so a
    # search that read the old row just before this commit can't write it back
    # (see mark_deactivated_in_cache). If anything in this block fails, invalidate the
    # whole batch instead so the next lookup reloads from Postgres rather than
    # serving stale data.
    # Tracked so the caller learns whether the cache actually caught up. False
    # means the write committed but stale entries may still be served — for a
    # deactivated acronym that is up to REDIS_CACHE_TTL (24h) of it still
    # working in search. The request deliberately still succeeds: the database
    # change is done, and re-uploading would not fix a cache problem.
    cache_refreshed = True
    if created or updated:
        active_batch = [a for a in (created + updated) if a not in deactivated]
        try:
            if active_batch:
                await run_in_threadpool(refresh_cache, active_batch)
            if deactivated:
                cache_refreshed = await run_in_threadpool(mark_deactivated_in_cache, deactivated)
        except Exception as e:
            logger.warning(f"Cache refresh failed after bulk upload, invalidating instead: {e}")
            # Active keys are deleted so the next lookup reloads them. Deactivated
            # keys still get their marker, not a delete: an empty key would let a
            # lookup that read the old active row write it back for the full TTL.
            invalidated = await run_in_threadpool(invalidate_cache, active_batch)
            marked = await run_in_threadpool(mark_deactivated_in_cache, deactivated)
            cache_refreshed = invalidated and marked

    if not cache_refreshed:
        logger.warning(
            "Acronym bulk upload committed but the cache could not be updated; "
            "stale entries may be served until their TTL expires"
        )

    logger.info(
        f"Acronym bulk upload: {len(created)} created, {len(updated)} updated, "
        f"{len(deactivated)} deactivated, {len(errors)} error(s)"
    )
    return AcronymBulkUploadResponse(
        received=len(rows),
        created=len(created),
        updated=len(updated),
        # Already computed by bulk_upsert and used above to invalidate the
        # right cache keys; it was simply never surfaced to the caller.
        deactivated=len(deactivated),
        cache_refreshed=cache_refreshed,
        errors=errors,
    )
