"""
LLM Client Base Class
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from http.client import RemoteDisconnected
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
    after_log
)
from ..schemas.types import LLMRequest, LLMResponse
from ...config import get_settings
from ...logging_config import get_logger

logger = get_logger(__name__)


def is_retryable_error(exception: Exception) -> bool:
    """Determine if exception is retryable
    
    Retryable errors include:
    - ConnectionError (connection interrupted)
    - TimeoutError (timeout)
    - RemoteDisconnected (remote end disconnected)
    - HTTP 429 (rate limited)
    - HTTP 500/502/503 (temporary server errors)
    """
    error_msg = str(exception).lower()
    error_type = type(exception).__name__
    
    # Network connection related errors
    if isinstance(exception, (ConnectionError, TimeoutError, RemoteDisconnected)):
        logger.warning(f"Detected retryable network error: {error_type} - {error_msg}")
        return True
    
    # HTTP status code related errors
    retryable_status_codes = ['429', '500', '502', '503']
    for code in retryable_status_codes:
        if code in error_msg:
            logger.warning(f"Detected retryable HTTP error ({code}): {error_msg}")
            return True
    
    # Specific error message patterns
    retryable_patterns = [
        'connection aborted',
        'connection reset',
        'connection refused',
        'timeout',
        'timed out',
        'remote end closed',
        'remotedisconnected'
    ]
    
    for pattern in retryable_patterns:
        if pattern in error_msg:
            logger.warning(f"Detected retryable error pattern '{pattern}': {error_msg}")
            return True
    
    # Non-retryable errors (authentication, permissions, configuration, etc.)
    non_retryable_patterns = ['401', '403', '404', 'unauthorized', 'forbidden', 'not found']
    for pattern in non_retryable_patterns:
        if pattern in error_msg:
            logger.info(f"Detected non-retryable error: {error_msg}")
            return False
    
    # Default to not retry unknown errors
    logger.info(f"Unknown error type, not retrying: {error_type} - {error_msg}")
    return False


class BaseLLMClient(ABC):
    """Generic LLM Client Base Class"""
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.config = kwargs
        self.settings = get_settings()
    
    @abstractmethod
    async def _call_impl(self, request: LLMRequest) -> LLMResponse:
        """Unified LLM call interface implementation (implemented by subclasses)"""
        pass
    
    async def call(self, request: LLMRequest) -> LLMResponse:
        """LLM call interface with retry mechanism"""
        settings = get_settings()
        
        # Create call function with retry
        @retry(
            stop=stop_after_attempt(settings.llm_max_retries),
            wait=wait_exponential(
                min=settings.llm_retry_min_wait,
                max=settings.llm_retry_max_wait
            ),
            retry=retry_if_exception(is_retryable_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO)
        )
        async def call_with_retry():
            try:
                return await self._call_impl(request)
            except Exception as e:
                # Log error information
                logger.error(f"LLM API call error: {type(e).__name__} - {str(e)}")
                raise
        
        # Execute call with retry
        try:
            return await call_with_retry()
        except Exception as e:
            # After all retries failed, log final error
            logger.error(f"LLM API call failed, max retries reached: {str(e)}")
            raise
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass
    
    async def _process_attachments(self, attachments: Optional[List[str]]) -> str:
        """Process attachment content (completed within LLM module)"""
        if not attachments:
            return ""
        
        # Implement attachment reading and processing logic
        attachment_content = ""
        for attachment_path in attachments:
            # Process attachment based on file type
            content = await self._read_attachment(attachment_path)
            attachment_content += f"\nAttachment content: {content}\n"
        
        return attachment_content
    
    async def _read_attachment(self, file_path: str) -> str:
        """Read single attachment content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Failed to read attachment: {str(e)}"
