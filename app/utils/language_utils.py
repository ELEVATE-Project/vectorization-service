import logging
import requests
import time
from fastapi import HTTPException
import datetime
from typing import Dict, Any, Optional
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """Custom exception for translation errors"""
    def __init__(self, message: str, status_code: int, should_retry: bool = False):
        self.message = message
        self.status_code = status_code
        self.should_retry = should_retry
        super().__init__(self.message)

def exponential_backoff(retry_count: int, base_delay: float = 1.0) -> float:
    """Calculate delay with exponential backoff"""
    return min(base_delay * (2 ** retry_count), 60)  # Cap at 60 seconds

def _handle_rate_limit(response, retry_count: int) -> int:
    """Handle rate limit response"""
    retry_after = int(response.headers.get('Retry-After', 60))
    logger.warning(f"Rate limited. Waiting {retry_after} seconds before retry.")
    time.sleep(retry_after)
    return retry_count + 1

def _handle_server_error(response, retry_count: int, max_retries: int) -> int:
    """Handle server error response"""
    if retry_count < max_retries:
        delay = exponential_backoff(retry_count)
        logger.warning(f"Server error. Retrying in {delay} seconds...")
        time.sleep(delay)
        return retry_count + 1
    else:
        raise TranslationError(
            f"Server error after {max_retries} retries",
            status_code=response.status_code
        )

def _handle_timeout(retry_count: int, max_retries: int) -> int:
    """Handle request timeout"""
    if retry_count < max_retries:
        delay = exponential_backoff(retry_count)
        logger.warning(f"Request timeout. Retrying in {delay} seconds...")
        time.sleep(delay)
        return retry_count + 1
    raise TranslationError("Translation timeout", status_code=408)

def translate_text(
    text: str,
    source_language: str,
    target_language: str,
    api_key: str,
    max_retries: int = 3
) -> str:
    """
    Translate text with robust error handling and retry logic
    """
    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        try:
            payload = {
                "controlConfig": {"dataTracking": True},
                "input": [{"source": text}],
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language,
                    },
                },
            }

            headers = {
                'accept': '/',
                'content-type': 'application/json',
                'origin': 'https://models.ai4bharat.org',
                'referer': 'https://models.ai4bharat.org/',
                'ulcaApiKey': api_key
            }

            response = requests.post(
                settings.TRANSLATION_API_URL,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()['output'][0]['target']

            elif response.status_code == 429:
                retry_count = _handle_rate_limit(response, retry_count)
                continue

            elif response.status_code >= 500:
                retry_count = _handle_server_error(response, retry_count, max_retries)
                continue

            else:
                raise TranslationError(
                    f"Translation API error: {response.status_code}",
                    status_code=response.status_code
                )

        except requests.exceptions.Timeout:
            retry_count = _handle_timeout(retry_count, max_retries)
            continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {str(e)}")
            raise TranslationError(f"Network error: {str(e)}", status_code=503)

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise TranslationError(f"Unexpected error: {str(e)}", status_code=500)

    raise HTTPException(
        status_code=500,
        detail=f"Translation failed after {max_retries} retries. Last error: {last_error}"
    )

def detect_language(text: str) -> str:
    """Detect if text contains Hindi characters"""
    hindi_chars = set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
    text_chars = set(text)
    return 'hi' if text_chars & hindi_chars else 'en'