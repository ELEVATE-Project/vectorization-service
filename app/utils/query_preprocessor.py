"""
Query preprocessing utility using spaCy for text normalization.

This module provides query preprocessing functionality to improve search quality by:
- Removing pronouns (I, my, you, etc.)
- Removing stop words (the, a, an, etc.)
- Applying lemmatization (running → run, algorithms → algorithm)
- Normalizing whitespace
"""
import logging
from typing import Optional
import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# Global spaCy model instance (singleton pattern for efficiency)
_nlp_model: Optional[Language] = None


def _load_spacy_model() -> Language:
    """
    Load spaCy English model with singleton pattern.
    
    Returns:
        Loaded spaCy Language model
        
    Raises:
        RuntimeError: If model loading fails
    """
    global _nlp_model
    
    if _nlp_model is not None:
        return _nlp_model
    
    try:
        logger.info("Loading spaCy English model (en_core_web_sm)...")
        _nlp_model = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
        return _nlp_model
    except OSError as e:
        error_msg = (
            "Failed to load spaCy model 'en_core_web_sm'. "
            "Please install it using: python -m spacy download en_core_web_sm"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        logger.error(f"Unexpected error loading spaCy model: {str(e)}")
        raise RuntimeError(f"Failed to load spaCy model: {str(e)}") from e


def preprocess_query(query: str) -> str:
    # Handle empty or whitespace-only queries
    if not query or not query.strip():
        logger.debug("Empty query provided, returning empty string")
        return ""

    stripped = query.strip()

    # Short queries (fewer words than threshold or fewer than 20 chars) bypass stop-word
    # removal entirely — stripping "the" from "the AI" or "RTE" would destroy meaning.
    try:
        from app.config import settings
        threshold = settings.SHORT_QUERY_THRESHOLD
    except Exception:
        threshold = 3

    word_count = len(stripped.split())
    if word_count < threshold or len(stripped) < 20:
        logger.debug(
            f"Short query ({word_count} words, {len(stripped)} chars) — "
            "skipping spaCy preprocessing"
        )
        return stripped.lower()

    try:
        # Load spaCy model
        nlp = _load_spacy_model()
        
        # Log original query
        logger.debug(f"Preprocessing query: '{query}'")
        
        # Process query through spaCy pipeline
        doc = nlp(query)
        
        # Filter tokens (no lemmatization)
        processed_tokens = []
        for token in doc:
            # Skip pronouns (I, my, you, etc.)
            if token.pos_ == "PRON":
                logger.debug(f"Removing pronoun: '{token.text}'")
                continue
            
            # Skip stop words (the, a, an, is, etc.)
            if token.is_stop:
                logger.debug(f"Removing stop word: '{token.text}'")
                continue
            
            # Skip punctuation
            if token.is_punct:
                continue
            
            # Skip whitespace tokens
            if token.is_space:
                continue
            
            # Add original token text (lowercase)
            token_text = token.text.lower().strip()
            if token_text:  # Only add non-empty tokens
                processed_tokens.append(token_text)
                logger.debug(f"Keeping token: '{token.text}'")
        
        # Join tokens and normalize whitespace
        preprocessed = " ".join(processed_tokens)
        preprocessed = " ".join(preprocessed.split())

        # If preprocessing wiped everything out, fall back to the original so the
        # embedding step always has something to work with.
        if not preprocessed:
            logger.debug("Preprocessing produced empty result — falling back to original query")
            return stripped.lower()

        logger.debug(f"Preprocessed result: '{preprocessed}'")
        return preprocessed

    except RuntimeError:
        # Re-raise model loading errors
        raise
    except Exception as e:
        logger.error(f"Error preprocessing query '{query}': {str(e)}", exc_info=True)
        logger.warning("Returning original query due to preprocessing error")
        return stripped


def is_spacy_model_available() -> bool:
    """
    Check if spaCy model is available without loading it.
    
    Returns:
        True if model is available, False otherwise
    """
    try:
        import spacy.util
        return spacy.util.is_package("en_core_web_sm")
    except Exception:
        return False
