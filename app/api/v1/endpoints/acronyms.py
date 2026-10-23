import csv
import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.api.deps import verify_internal_token
from app.config import settings
from app.constants import messages as msg
from app.models.api_models import AcronymBulkUploadResponse, AcronymItem, AcronymListResponse
from app.services.acronym_service import (
    bulk_upsert,
    invalidate_cache,
    list_acronyms,
    load_acronym_cache,
)
from app.validations.acronym import AcronymCsvError, validate_csv_header, validate_csv_rows_present

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
                created_by=row.created_by,
                updated_by=row.updated_by,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ],
    )


@router.post(
    "/bulkUpload",
    response_model=AcronymBulkUploadResponse,
    dependencies=[Depends(verify_internal_token)],
)
async def bulk_upload_acronyms(file: UploadFile = File(...)):
    """Upsert acronyms from a CSV: acronym, expansions (pipe-separated), description, is_active.

    Needs internal_access_token. Bad rows are reported in `errors`; the rest commit.
    """
    max_bytes = settings.ACRONYM_BULK_UPLOAD_MAX_SIZE_MB * 1024 * 1024
    # One byte past the limit is enough to know it's too big, without pulling
    # an oversized upload into worker memory first.
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=msg.ACRONYM_CSV_TOO_LARGE.format(max_size_mb=settings.ACRONYM_BULK_UPLOAD_MAX_SIZE_MB),
        )

    try:
        # utf-8-sig drops an Excel/Sheets BOM, which would otherwise join the
        # first header name and fail every row.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail=msg.ACRONYM_CSV_NOT_UTF8)

    reader = csv.DictReader(io.StringIO(text))
    try:
        reader.fieldnames = validate_csv_header(reader.fieldnames or [])
        rows = list(reader)
        validate_csv_rows_present(rows)
    except AcronymCsvError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # bulk_upsert/invalidate_cache/load_acronym_cache are all synchronous, blocking
    # Postgres/Redis I/O — run_in_threadpool keeps them off the event loop.
    try:
        created, updated, deactivated, errors = await run_in_threadpool(bulk_upsert, rows)
    except SQLAlchemyError as e:
        # A lost connection during commit can still have saved it; a retry is safe (upsert).
        logger.error(f"Acronym bulk upload failed in the database, outcome unconfirmed: {e}")
        raise HTTPException(
            status_code=503 if isinstance(e, OperationalError) else 500,
            detail=msg.ACRONYM_UPLOAD_DB_ERROR,
        ) from e

    # Committed already; delete the uploaded keys, then reload every active acronym.
    # Deactivated keys stay deleted, since the reload writes active ones only.
    # cache_refreshed=False means stale entries may be served. See design notes.
    cache_refreshed = True
    if created or updated:
        deleted = await run_in_threadpool(invalidate_cache, created + updated)
        reloaded = await run_in_threadpool(load_acronym_cache)
        cache_refreshed = deleted and reloaded

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
        created_acronyms=created,
        updated_acronyms=updated,
        deactivated_acronyms=deactivated,
    )
