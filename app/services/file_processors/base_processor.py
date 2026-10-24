from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from app.config import settings
from app.constants import messages as msg

logger = logging.getLogger(__name__)


class BaseFileProcessor(ABC):
    """Abstract base class for file processors"""

    def __init__(self):
        self.supported_extensions = self.get_supported_extensions()

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        pass

    @abstractmethod
    async def process(self, file_content: bytes, filename: str, priority: str) -> List[Dict[str, Any]]:
        """Process file and return chunks"""
        pass

    def can_process(self, file_extension: str) -> bool:
        """Check if this processor can handle the file type"""
        return file_extension.lower() in self.supported_extensions

    def _validate_file_content(self, file_content: bytes, filename: str):
        """Common validation logic"""
        if not file_content:
            raise ValueError(msg.PROCESSOR_EMPTY_FILE_CONTENT.format(filename=filename))

        max_file_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if len(file_content) > max_file_size_bytes:
            raise ValueError(
                msg.PROCESSOR_FILE_TOO_LARGE.format(
                    filename=filename, size_mb=file_size_mb, max_size_mb=settings.MAX_FILE_SIZE_MB
                )
            )
