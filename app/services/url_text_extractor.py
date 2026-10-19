import logging
import httpx
from typing import Optional
from bs4 import BeautifulSoup
from fastapi import HTTPException
from app.config import settings

logger = logging.getLogger(__name__)


class URLTextExtractor:
    """Service to extract text content from URLs"""
    
    def __init__(self):
        self.timeout = settings.URL_REQUEST_TIMEOUT
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def extract_text(self, url: str) -> str:
        """
        Extract text content from a URL.
        
        Args:
            url: The URL to extract text from
            
        Returns:
            Extracted text content
            
        Raises:
            HTTPException: If URL is invalid or extraction fails
        """
        if not url or not url.strip():
            raise HTTPException(
                status_code=400,
                detail="URL cannot be empty"
            )
        
        # Validate URL format
        # NOSONAR - Service must accept both HTTP and HTTPS URLs from user input
        if not url.startswith(('http://', 'https://')):  # NOSONAR
            raise HTTPException(
                status_code=400,
                detail=f"Invalid URL format: {url}. URL must start with http:// or https://"
            )
        
        try:
            logger.info(f"Fetching content from URL: {url}")
            
            # Fetch content from URL using async HTTP client
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=self.headers)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Get content type
            content_type = response.headers.get('Content-Type', '').lower()
            
            logger.info(f"Content-Type: {content_type}")
            
            # Extract text based on content type
            if 'text/html' in content_type:
                text = self._extract_from_html(response.text, url)
            elif 'text/plain' in content_type or 'text/markdown' in content_type:
                text = response.text
            else:
                # Try to extract as HTML anyway
                logger.warning(f"Unknown content type: {content_type}, attempting HTML extraction")
                text = self._extract_from_html(response.text, url)
            
            if not text or not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"No text content could be extracted from URL: {url}"
                )
            
            logger.info(f"Successfully extracted {len(text)} characters from URL")
            return text.strip()
            
        except httpx.TimeoutException:
            logger.error(f"Timeout while fetching URL: {url}")
            raise HTTPException(
                status_code=408,
                detail=f"Request timeout while fetching URL: {url}"
            )
        except httpx.ConnectError:
            logger.error(f"Connection error while fetching URL: {url}")
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to URL: {url}"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while fetching URL: {url}, Status: {e.response.status_code}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"HTTP error {e.response.status_code} while fetching URL: {url}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting text from URL {url}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from URL: {str(e)}"
            )
    
    def _extract_from_html(self, html_content: str, url: str) -> str:
        """
        Extract main text content from HTML.
        
        Args:
            html_content: HTML content as string
            url: Original URL (for logging)
            
        Returns:
            Extracted text content
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                script.decompose()
            
            # Try to find main content area
            main_content = None
            
            # Common main content selectors
            main_selectors = [
                'main',
                'article',
                '[role="main"]',
                '.main-content',
                '#main-content',
                '.content',
                '#content',
                '.post-content',
                '.article-content',
                '.entry-content'
            ]
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    logger.info(f"Found main content using selector: {selector}")
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body')
                logger.info("Using body tag for content extraction")
            
            if not main_content:
                # Fallback to entire document
                main_content = soup
                logger.warning("No body tag found, using entire document")
            
            # Extract text
            text = main_content.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            return text
            
        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {str(e)}")
            # Fallback to simple text extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
