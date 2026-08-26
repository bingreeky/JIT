"""
LLM Client Factory
"""
from typing import Type
from .base import BaseLLMClient
from .gemini import GeminiClient
from .chatgpt import ChatGPTClient
from .openrouter import OpenRouterClient
from ...config import get_settings
from ..utils.attachment_parser import AttachmentParser
from ..schemas.types import LLMRequest, LLMResponse


class LLMClientFactory:
    """LLM Client Factory"""

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if not model_name:
            return model_name
        if model_name.startswith(("openrouter/", "gemini-", "gpt-", "qwen/")):
            return model_name
        # Route anything else (e.g. bare "deepseek-v4-flash") through the
        # OpenRouter-style OpenAI-compatible client: the prefix is stripped
        # again in create_client, and the configured openrouter_base_url /
        # api_key (mapped from judge_api_base/key) do the real targeting.
        return f"openrouter/{model_name}"
    
    @classmethod
    def create_client(cls, model_name: str) -> BaseLLMClient:
        """Create LLM client instance
        
        Auto-detect client type based on prefix:
        - gemini-* -> GeminiClient (uses official model name directly)
        - gpt-* -> ChatGPTClient (uses official model name directly)
        - qwen/* -> OpenRouterClient
        - openrouter/* -> OpenRouterClient (removes 'openrouter/' prefix when calling)
        """
        settings = get_settings()
        model_name = cls._normalize_model_name(model_name)
        
        # Auto-detect client type based on prefix
        if model_name.startswith("openrouter/"):
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API Key not configured")
            # Remove 'openrouter/' prefix before passing to client
            openrouter_model = model_name[len("openrouter/"):]
            return OpenRouterClient(
                api_key=settings.openrouter_api_key,
                model=openrouter_model,
                base_url=settings.openrouter_base_url
            )
        elif model_name.startswith("qwen/"):
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API Key not configured")
            return OpenRouterClient(
                api_key=settings.openrouter_api_key,
                model=model_name,
                base_url=settings.openrouter_base_url
            )
        elif model_name.startswith("gemini-"):
            if not settings.gemini_api_key:
                raise ValueError("Gemini API Key not configured")
            # Use official model name directly
            return GeminiClient(
                api_key=settings.gemini_api_key, 
                model=model_name,
                base_url=settings.gemini_base_url
            )
        elif model_name.startswith("gpt-"):
            if not settings.chatgpt_api_key:
                raise ValueError("ChatGPT API Key not configured")
            # Use official model name directly
            return ChatGPTClient(
                api_key=settings.chatgpt_api_key,
                model=model_name,
                base_url=settings.chatgpt_base_url
            )
        else:
            raise ValueError(
                f"Unsupported model prefix: {model_name}. "
                "Model name must start with 'gemini-', 'gpt-', 'qwen/', or 'openrouter/'"
            )
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available models (empty list as models are now prefix-based)"""
        return []
    
    @classmethod
    async def call_with_attachments(cls, request: LLMRequest) -> LLMResponse:
        """
        LLM call with attachment parsing
        
        Args:
            request: LLM request containing attachment information
            
        Returns:
            LLM response
        """
        # Create client
        request.model_name = cls._normalize_model_name(request.model_name)
        client = cls.create_client(request.model_name)
        
        # If attachment parsing is needed and not a Gemini model (Gemini has its own multimodal handling)
        if (request.parse_attachments and 
            request.attachments and
            not request.model_name.startswith("gemini-")):
            # Create attachment parser
            parser = AttachmentParser()
            
            # Parse attachment content
            attachment_content = await parser.parse_attachments(request.attachments, request.html_extract_mode)
            
            # Add attachment content to the last user message
            if attachment_content and request.messages:
                last_user_msg = None
                for i in range(len(request.messages) - 1, -1, -1):
                    if request.messages[i].role == "user":
                        last_user_msg = request.messages[i]
                        break
                
                if last_user_msg:
                    last_user_msg.content += f"\n\n{attachment_content}"
                else:
                    from ..schemas.types import LLMMessage
                    request.messages.append(LLMMessage(
                        role="user", 
                        content=f"Please analyze the following attachment content:\n\n{attachment_content}"
                    ))
        
        # Call LLM
        return await client.call(request)
