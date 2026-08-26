"""
OpenRouter Client Implementation - Using OpenAI Compatible API
Supports multiple models (Gemini, Claude, GPT, etc.), web search and multimodal input
"""
import base64
import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from .base import BaseLLMClient
from ..schemas.types import LLMRequest, LLMResponse
from ..utils.attachment_parser import AttachmentParser
from ...config import get_settings
from ...logging_config import get_logger

logger = get_logger(__name__)


class PayloadTooLargeError(Exception):
    """Exception raised when API returns 413 Payload Too Large"""
    pass


class ContextLengthExceededError(Exception):
    """Exception raised when request exceeds model's context length"""
    pass


class OpenRouterClient(BaseLLMClient):
    """OpenRouter Client Implementation, using OpenAI Compatible API"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        settings = get_settings()
        self.base_url = kwargs.get("base_url", settings.openrouter_base_url)
        self.model = kwargs.get("model", "google/gemini-2.0-flash-exp")
        
        # Initialize OpenAI SDK client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Initialize attachment parser
        self.attachment_parser = AttachmentParser()
        
        logger.info(f"Initialized OpenRouter client: model={self.model}, base_url={self.base_url}")
    
    async def _call_impl(self, request: LLMRequest) -> LLMResponse:
        """Call OpenRouter API, with multimodal and web search support"""
        # Build message list
        messages = await self._build_messages(request)
        
        # Determine whether to use web search
        use_web_search = request.use_grounding
        if use_web_search is None:
            settings = get_settings()
            use_web_search = settings.enable_google_search_grounding
        
        # Build model name (keep original model name, don't add :online suffix)
        model_name = self.model
        
        # Build base payload
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": request.temperature,
        }
        
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        
        # Add web search via extra_body if enabled
        if use_web_search:
            settings = get_settings()
            payload["extra_body"] = {
                "plugins": [
                    {
                        "id": "web",
                        "max_results": settings.web_search_max_results
                    }
                ]
            }
            logger.info(f"Web search enabled with max_results={settings.web_search_max_results}, using model: {model_name}")
        
        # Execute API call
        settings = get_settings()
        timeout = max(settings.llm_timeout, 180)
        
        try:
            logger.info(f"Preparing to call OpenRouter API:")
            logger.info(f"  - model: {model_name}")
            logger.info(f"  - web_search: {use_web_search}")
            if use_web_search:
                logger.info(f"  - web_search_max_results: {settings.web_search_max_results}")
            
            response = await self.client.chat.completions.create(
                **payload,
                timeout=timeout
            )
            
            logger.info(f"API call successful, parsing response")
            return self._parse_response(response, use_web_search)
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"OpenRouter API call failed [{error_type}]: {error_msg}")
            
            # Check for 413 Payload Too Large error
            if "413" in error_msg or "payload too large" in error_msg.lower():
                raise PayloadTooLargeError(f"Request payload too large: {error_msg}")
            # Check for 400 error with context length exceeded (especially when output token is 1000000)
            elif "400" in error_msg and ("context length" in error_msg.lower() or "maximum context" in error_msg.lower()):
                # Check if it mentions large output token count (e.g., 1000000)
                if "1000000 in the output" in error_msg or "1000000" in error_msg:
                    raise ContextLengthExceededError(f"Context length exceeded with large output token request: {error_msg}")
                else:
                    raise Exception(f"OpenRouter API call failed [{error_type}]: {error_msg}")
            elif "401" in error_msg or "Unauthorized" in error_msg:
                raise Exception(f"OpenRouter API authentication failed, please check OPENROUTER_API_KEY configuration: {error_msg}")
            elif "404" in error_msg or "Not Found" in error_msg:
                raise Exception(f"OpenRouter API endpoint not found, please check OPENROUTER_BASE_URL configuration ({self.base_url}): {error_msg}")
            else:
                raise Exception(f"OpenRouter API call failed [{error_type}]: {error_msg}")
    
    def _parse_response(self, response: Any, use_web_search: bool = False) -> LLMResponse:
        """Parse OpenRouter API response"""
        try:
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            finish_reason = choice.finish_reason or "stop"
            
            # Parse web search metadata
            grounding_metadata = None
            if use_web_search and hasattr(message, 'annotations') and message.annotations:
                grounding_metadata = {
                    'sources': [],
                    'annotations': []
                }
                
                for annotation in message.annotations:
                    if hasattr(annotation, 'type') and annotation.type == 'url_citation':
                        url_citation = annotation.url_citation
                        source_info = {}
                        if hasattr(url_citation, 'url'):
                            source_info['uri'] = url_citation.url
                        if hasattr(url_citation, 'title'):
                            source_info['title'] = url_citation.title
                        if hasattr(url_citation, 'content'):
                            source_info['content'] = url_citation.content
                        
                        grounding_metadata['sources'].append(source_info)
                
                if not grounding_metadata['sources']:
                    grounding_metadata = None
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=finish_reason,
                grounding_metadata=grounding_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}")
            raise Exception(f"Failed to parse response: {str(e)}")
    
    async def _build_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """Build message list with multimodal content support"""
        messages = []
        
        for msg in request.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Check if we need multimodal content (file attachments or image attachments)
        has_file_attachments = request.attachments or request.attachment_infos
        has_image_attachments = request.image_attachments and len(request.image_attachments) > 0
        
        if has_file_attachments or has_image_attachments:
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx is not None:
                original_content = messages[last_user_idx]["content"]
                content_parts = [{"type": "text", "text": original_content}]
                
                # Handle file attachments (original logic)
                if has_file_attachments:
                    if request.attachment_infos:
                        for attachment_info in request.attachment_infos:
                            await self._add_attachment_to_content(
                                content_parts,
                                attachment_info.path,
                                attachment_info.type,
                                attachment_info.extract_mode,
                                attachment_info.render_mode
                            )
                    elif request.attachments:
                        for attachment_path in request.attachments:
                            attachment_info = await self.attachment_parser.get_attachment_info(attachment_path)
                            
                            if attachment_info['file_category'] == 'image':
                                await self._add_attachment_to_content(content_parts, attachment_path, "image")
                            elif attachment_info['file_category'] == 'html':
                                await self._add_attachment_to_content(
                                    content_parts,
                                    attachment_path,
                                    "html",
                                    request.html_extract_mode
                                )
                            else:
                                await self._add_attachment_to_content(content_parts, attachment_path, "document")
                
                # Handle image attachments (screenshots from HTML rendering)
                if has_image_attachments:
                    for img_attachment in request.image_attachments:
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\n=== Screenshot: {img_attachment.filename} ==="
                        })
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img_attachment.mime_type};base64,{img_attachment.base64_data}"
                            }
                        })
                        logger.info(f"Added image attachment: {img_attachment.filename}")
                
                messages[last_user_idx]["content"] = content_parts
            else:
                content_parts = [{"type": "text", "text": "Please analyze the following attachment content:"}]
                
                if has_file_attachments:
                    if request.attachment_infos:
                        for attachment_info in request.attachment_infos:
                            await self._add_attachment_to_content(
                                content_parts,
                                attachment_info.path,
                                attachment_info.type,
                                attachment_info.extract_mode,
                                attachment_info.render_mode
                            )
                    elif request.attachments:
                        for attachment_path in request.attachments:
                            attachment_info = await self.attachment_parser.get_attachment_info(attachment_path)
                            
                            if attachment_info['file_category'] == 'image':
                                await self._add_attachment_to_content(content_parts, attachment_path, "image")
                            else:
                                await self._add_attachment_to_content(content_parts, attachment_path, "document")
                
                # Handle image attachments
                if has_image_attachments:
                    for img_attachment in request.image_attachments:
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\n=== Screenshot: {img_attachment.filename} ==="
                        })
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img_attachment.mime_type};base64,{img_attachment.base64_data}"
                            }
                        })
                        logger.info(f"Added image attachment: {img_attachment.filename}")
                
                messages.append({
                    "role": "user",
                    "content": content_parts
                })
        
        return messages
    
    async def _add_attachment_to_content(
        self,
        content_parts: List[Dict[str, Any]],
        file_path: str,
        file_type: str,
        extract_mode: str = "text",
        render_mode: Optional[str] = None
    ):
        """Add attachment to content parts"""
        try:
            if not os.path.exists(file_path):
                content_parts.append({
                    "type": "text",
                    "text": f"\n\nAttachment file does not exist: {file_path}"
                })
                return
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            if file_type == "image":
                try:
                    mime_type = self.attachment_parser.image_parser.get_mime_type(file_path)
                    base64_image = base64.b64encode(file_data).decode('utf-8')
                    
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                    logger.info(f"Added image: {file_path}, mime_type={mime_type}")
                except Exception as e:
                    content_parts.append({
                        "type": "text",
                        "text": f"\n\nImage processing failed: {file_path} - {str(e)}"
                    })
            
            elif file_type == "html":
                try:
                    html_content = await self.attachment_parser._parse_html(
                        file_data,
                        file_path,
                        extract_mode
                    )
                    if html_content:
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\n=== HTML file content ({file_path}) ===\n{html_content}"
                        })
                        logger.info(f"Added HTML file: {file_path} (mode: {extract_mode})")
                except Exception as e:
                    content_parts.append({
                        "type": "text",
                        "text": f"\n\nHTML file processing failed: {file_path} - {str(e)}"
                    })
            
            else:
                try:
                    attachment_info = await self.attachment_parser.get_attachment_info(file_path)
                    settings = get_settings()
                    effective_render_mode = render_mode if render_mode is not None else settings.pptx_render_mode
                    
                    if (attachment_info['file_category'] == 'office' and 
                        file_path.lower().endswith('.pptx') and 
                        effective_render_mode == 'image'):
                        try:
                            image_bytes_list = await self.attachment_parser.parse_pptx_for_multimodal(
                                file_data,
                                file_path,
                                dpi=settings.pptx_render_dpi,
                                max_slides=settings.pptx_render_max_slides
                            )
                            
                            for i, img_bytes in enumerate(image_bytes_list, 1):
                                content_parts.append({
                                    "type": "text",
                                    "text": f"\nSlide {i}:"
                                })
                                
                                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                })
                            
                            logger.info(f"Added PPTX rendered images: {file_path} ({len(image_bytes_list)} slides)")
                        except Exception as e:
                            logger.warning(f"PPTX image rendering failed, falling back to text extraction: {str(e)}")
                            content = await self.attachment_parser._parse_office(file_data, file_path)
                            content_parts.append({
                                "type": "text",
                                "text": f"\n\n=== Document content ({file_path}) [rendering failed, using text mode] ===\n{content}"
                            })
                    
                    elif attachment_info['file_category'] == 'office':
                        content = await self.attachment_parser._parse_office(file_data, file_path)
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\n=== Document content ({file_path}) ===\n{content}"
                        })
                        logger.info(f"Added document: {file_path}")
                    else:
                        content = await self.attachment_parser._parse_text(file_data, file_path)
                        content_parts.append({
                            "type": "text",
                            "text": f"\n\n=== Document content ({file_path}) ===\n{content}"
                        })
                        logger.info(f"Added document: {file_path}")
                        
                except Exception as e:
                    content_parts.append({
                        "type": "text",
                        "text": f"\n\nDocument processing failed: {file_path} - {str(e)}"
                    })
        
        except Exception as e:
            content_parts.append({
                "type": "text",
                "text": f"\n\nAttachment processing failed: {file_path} - {str(e)}"
            })
    
    def get_model_name(self) -> str:
        return f"openrouter/{self.model}"
