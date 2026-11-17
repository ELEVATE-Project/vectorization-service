from fastapi import APIRouter, Depends
from app.api.deps import verify_internal_token_if_enforced
from app.api.v1.endpoints import documents, query, cache, acronyms

# Applies to every route under /api; /api/health is defined in main.py and stays open.
api_router = APIRouter(dependencies=[Depends(verify_internal_token_if_enforced)])

api_router.include_router(documents.router, prefix="", tags=["documents"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(cache.router, prefix="/cache", tags=["cache"])
api_router.include_router(acronyms.router, prefix="/acronyms", tags=["acronyms"])
