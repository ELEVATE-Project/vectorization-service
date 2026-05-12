# app/services/translation_service.py
import logging
from app.utils.language_utils import detect_language, translate_text
from app.config import settings
from app.core.database import SessionLocal
from app.models.db_models import TranslationRecord

logger = logging.getLogger(__name__)

def process_chunk(chunk_text: str, chunk_id: str) -> tuple:
    """Process a single chunk of text - translation disabled"""
    # Translation disabled - return original text as-is
    return chunk_text, False
