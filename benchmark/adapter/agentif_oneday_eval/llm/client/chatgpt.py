"""
ChatGPT Client Implementation - Using official openai SDK
Supports multimodal input (simplified version, without function calling)
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


class ChatGPTClient(BaseLLMClient):
    """ChatGPT Client Implementation, using official openai SDK, with multimodal support"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        settings = get_settings()
        self.base_url = kwargs.get("base_url", settings.chatgpt_base_url)
        self.model = kwargs.get("model", "gpt-4o")
        
        # Initialize official openai SDK client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Initialize attachment parser
        self.attachment_parser = AttachmentParser()
        
        logger.info(f"Initialized ChatGPT official SDK client: model={self.model}, base_url={self.base_url}")
    
    async def _call_impl(self, request: LLMRequest) -> LLMResponse:
        """Call ChatGPT API, with multimodal support (internal implementation)"""
        # Build message list
        messages = await self._build_messages(request)
        
        # Build base payload
        payload = {
            "model": self.model,
            "messages": messages,
        }
        
        # Some models (like o1 series) don't support temperature parameter
        models_without_temperature = ["o1-preview", "o1-mini", "o1"]
        if not any(model_name in self.model for model_name in models_without_temperature):
            payload["temperature"] = request.temperature
        else:
            logger.info(f"Model {self.model} does not support temperature parameter, using default value")
        
        # Use max_completion_tokens
        if request.max_tokens:
            payload["max_completion_tokens"] = request.max_tokens
        
        # Execute API call
        settings = get_settings()
        timeout = max(settings.llm_timeout, 180)
        
        try:
            response = await self.client.chat.completions.create(
                **payload,
                timeout=timeout
            )
            
            # Parse response
            choice = response.choices[0]
            message = choice.message
            
            # Extract usage information
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return LLMResponse(
                content=message.content or "",
                usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason
            )
            
        except Exception as e:
            logger.error(f"ChatGPT API call failed: {str(e)}")
            raise Exception(f"ChatGPT API call failed: {str(e)}")
    
    async def _build_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """Build message list with multimodal content support"""
        messages = []
        
        # Process regular messages
        for msg in request.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Process attachments (multimodal support)
        if request.attachments or request.attachment_infos:
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx is not None:
                original_content = messages[last_user_idx]["content"]
                content_parts = [{"type": "text", "text": original_content}]
                
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
                
                messages[last_user_idx]["content"] = content_parts
            else:
                content_parts = [{"type": "text", "text": "Please analyze the following attachment content:"}]
                
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
        return self.model
