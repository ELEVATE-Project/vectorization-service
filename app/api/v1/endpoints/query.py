from fastapi import APIRouter, HTTPException
from app.models.api_models import MultilingualQueryRequest, MultilingualQueryResponse
from app.services.query_service import QueryService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
query_service = QueryService()

@router.post("/")
async def query_documents(request: MultilingualQueryRequest) -> MultilingualQueryResponse:
    """Query documents with multilingual support"""
    return query_service.process_query(request)
    