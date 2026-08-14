import csv
import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.v1.deps import verify_internal_token
from app.models.api_models import AcronymBulkUploadResponse
from app.services.acronym_service import bulk_upsert, invalidate_cache, load_acronym_cache

router = APIRouter()
logger = logging.getLogger(__name__)


def _invalidate_all(acronyms):
    for acronym in acronyms:
        invalidate_cache(acronym)


@router.post("/bulk", response_model=AcronymBulkUploadResponse, dependencies=[Depends(verify_internal_token)])
async def bulk_upload_acronyms(file: UploadFile = File(...)):
    """Internal-only: upsert a batch of acronym -> expansions rows from a CSV
    upload (spec §7). Columns: acronym, expansions (pipe-separated if more
    than one), description (optional).

    One invalid or duplicate row doesn't fail the batch — it's reported in
    `errors` while the rest of the batch still commits.
    """
    raw = await file.read()
    try:
        # utf-8-sig strips a leading BOM if present (common in CSVs exported
        # from Excel/Sheets) and is otherwise identical to plain utf-8 — a
        # BOM left in place would silently become part of the first header
        # name ("﻿acronym"), making every row fail acronym validation.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    rows = list(csv.DictReader(io.StringIO(text)))
    # bulk_upsert/load_acronym_cache/invalidate_cache are all synchronous, blocking
    # Postgres/Redis I/O — run_in_threadpool keeps them off the event loop,
    # so an upload (up to ~600 sequential Redis writes during the cache
    # refresh) doesn't stall every other in-flight request (search, health
    # checks) for its duration.
    created, updated, errors = await run_in_threadpool(bulk_upsert, rows)

    # Spec §7: commit first (bulk_upsert already did), then refresh the
    # cache; if the refresh itself fails, invalidate the upserted keys
    # instead so the next lookup reloads from Postgres rather than serving
    # stale data.
    if created or updated:
        try:
            await run_in_threadpool(load_acronym_cache)
        except Exception as e:
            logger.warning(f"Cache refresh failed after bulk upload, invalidating instead: {e}")
            await run_in_threadpool(_invalidate_all, created + updated)

    logger.info(
        f"Acronym bulk upload: {len(created)} created, {len(updated)} updated, "
        f"{len(errors)} error(s)"
    )
    return AcronymBulkUploadResponse(
        received=len(rows),
        created=len(created),
        updated=len(updated),
        errors=errors,
    )
