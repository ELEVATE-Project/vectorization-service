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
    """List the acronym dictionary, active and inactive rows by default.

    Needs internal_access_token only; reading is lower risk than writing.
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
    """Upsert acronyms from a CSV: acronym, expansions (pipe-separated), description, is_active.

    Needs both tokens. Bad rows are reported in `errors`; the rest commit.
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
        # utf-8-sig drops an Excel/Sheets BOM, which would otherwise join the
        # first header name and fail every row.
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

    # Committed already; now refresh active rows and mark deactivated ones.
    # cache_refreshed=False means stale entries may be served. See design notes.
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
            # Delete active keys (reload on next lookup); deactivated keys still get
            # their marker, since an empty key could be refilled with the old value.
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
