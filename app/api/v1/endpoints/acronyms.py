import csv
import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.v1.deps import verify_internal_token
from app.models.api_models import AcronymBulkUploadResponse
from app.services.acronym_service import bulk_upsert, invalidate_cache, warm_cache

router = APIRouter()
logger = logging.getLogger(__name__)


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
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    rows = list(csv.DictReader(io.StringIO(text)))
    created, updated, errors = bulk_upsert(rows)

    # Spec §7: commit first (bulk_upsert already did), then refresh the
    # cache; if the refresh itself fails, invalidate the upserted keys
    # instead so the next lookup reloads from Postgres rather than serving
    # stale data.
    if created or updated:
        try:
            await warm_cache()
        except Exception as e:
            logger.warning(f"Cache refresh failed after bulk upload, invalidating instead: {e}")
            for acronym in created + updated:
                invalidate_cache(acronym)

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
