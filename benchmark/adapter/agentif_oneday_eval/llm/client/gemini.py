"""
Gemini Client Implementation - Using official google.genai SDK, without function calling
"""
import os
from pathlib import Path
from typing import Any, Optional
from google import genai
from google.genai import types
from .base import BaseLLMClient
from ..schemas.types import LLMRequest, LLMResponse
from ..utils.attachment_parser import AttachmentParser
from ...config import get_settings
from ...logging_config import get_logger

logger = get_logger(__name__)


class GeminiClient(BaseLLMClient):
    """Gemini Client Implementation - Using official SDK, without function calling"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        # Get base_url from config, support custom configuration
        settings = get_settings()
        self.base_url = kwargs.get("base_url", settings.gemini_base_url)
        self.model = kwargs.get("model", "gemini-2.5-flash")
        
        # Initialize official SDK client
        if self.base_url:
            http_options = types.HttpOptions(base_url=self.base_url)
            self.client = genai.Client(api_key=self.api_key, http_options=http_options)
            logger.info(f"Initialized Gemini official SDK client: model={self.model}, base_url={self.base_url}")
        else:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Gemini official SDK client: model={self.model}, using default endpoint")
        
        # Initialize attachment parser
        self.attachment_parser = AttachmentParser()
    
    async def _call_impl(self, request: LLMRequest) -> LLMResponse:
        """Call Gemini API, supporting multimodal input and Google Search grounding (internal implementation)"""
        # Build contents list (supports mixed text and Part objects)
        contents = []
        
        # Process message content
        user_text = ""
        system_instruction = None
        
        for msg in request.messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                user_text += msg.content + "\n"
            elif msg.role == "assistant":
                user_text += f"Assistant previously said: {msg.content}\n"
        
        # Add text content
        if user_text.strip():
            contents.append(user_text.strip())
        
        # Process attachments (images use Part, others converted to text)
        if request.attachments or request.attachment_infos:
            await self._process_multimodal_attachments(contents, request)
        
        # Ensure at least one content item
        if not contents:
            contents.append("Please analyze the provided content.")
        
        # Build configuration
        config_params = {
            "temperature": request.temperature,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": request.max_tokens or 8000,
        }
        
        # Check if Google Search grounding is enabled
        settings = get_settings()
        use_grounding = request.use_grounding
        if use_grounding is None:
            use_grounding = settings.enable_google_search_grounding
        
        # Prepare tools list
        tools = None
        if use_grounding:
            tools = [types.Tool(google_search=types.GoogleSearch())]
            logger.info("Google Search grounding enabled")
        
        # Create generation config
        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=config_params["temperature"],
            top_k=config_params["top_k"],
            top_p=config_params["top_p"],
            max_output_tokens=config_params["max_output_tokens"],
            tools=tools
        )
        
        # Call official SDK
        try:
            grounding_mode = "with Google Search grounding" if use_grounding else "without tools"
            logger.info(f"Preparing to call Gemini API ({grounding_mode}):")
            logger.info(f"  - model: {self.model}")
            logger.info(f"  - base_url: {self.base_url}")
            logger.info(f"  - contents types: {[type(c).__name__ for c in contents]}")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            logger.info(f"API call successful, parsing response")
            return self._parse_response(response, use_grounding)
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"Gemini API call failed [{error_type}]: {error_msg}")
            
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise Exception(f"Gemini API authentication failed, please check GEMINI_API_KEY configuration: {error_msg}")
            elif "404" in error_msg or "Not Found" in error_msg:
                raise Exception(f"Gemini API endpoint not found, please check GEMINI_BASE_URL configuration ({self.base_url}): {error_msg}")
            else:
                raise Exception(f"Gemini API call failed [{error_type}]: {error_msg}")
    
    def _parse_response(self, response: Any, use_grounding: bool = False) -> LLMResponse:
        """Parse SDK response"""
        try:
            # Get text content
            content = None
            if hasattr(response, 'text'):
                content = response.text
            
            # If text is None, try to extract from candidates
            if content is None and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content'):
                    candidate_content = candidate.content
                    if hasattr(candidate_content, 'parts') and candidate_content.parts is not None:
                        parts_text = []
                        for part in candidate_content.parts:
                            if hasattr(part, 'text') and part.text:
                                parts_text.append(part.text)
                        if parts_text:
                            content = ''.join(parts_text)
            
            if content is None:
                logger.warning("Unable to extract text content from response, using empty string")
                content = ""
            
            # Parse usage statistics
            usage = None
            if hasattr(response, 'usage_metadata'):
                usage_metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage_metadata, 'total_token_count', 0)
                }
            
            # Parse finish reason
            finish_reason = "stop"
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    reason = str(candidate.finish_reason)
                    if "STOP" in reason:
                        finish_reason = "stop"
                    else:
                        finish_reason = reason.lower()
            
            # Parse grounding metadata
            grounding_metadata = None
            if use_grounding and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    gm = candidate.grounding_metadata
                    grounding_metadata = {}
                    
                    if hasattr(gm, 'search_entry_point'):
                        search_entry = gm.search_entry_point
                        if hasattr(search_entry, 'rendered_content'):
                            grounding_metadata['search_query'] = search_entry.rendered_content
                    
                    if hasattr(gm, 'grounding_supports') and gm.grounding_supports is not None:
                        supports = []
                        for support in gm.grounding_supports:
                            support_info = {}
                            if hasattr(support, 'segment'):
                                segment = support.segment
                                if hasattr(segment, 'text'):
                                    support_info['text'] = segment.text
                            if hasattr(support, 'grounding_chunk_indices') and support.grounding_chunk_indices is not None:
                                support_info['chunk_indices'] = list(support.grounding_chunk_indices)
                            if hasattr(support, 'confidence_scores') and support.confidence_scores is not None:
                                support_info['confidence_scores'] = list(support.confidence_scores)
                            supports.append(support_info)
                        grounding_metadata['supports'] = supports
                    
                    if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks is not None:
                        chunks = []
                        for chunk in gm.grounding_chunks:
                            chunk_info = {}
                            if hasattr(chunk, 'web'):
                                web = chunk.web
                                if hasattr(web, 'uri'):
                                    chunk_info['uri'] = web.uri
                                if hasattr(web, 'title'):
                                    chunk_info['title'] = web.title
                            chunks.append(chunk_info)
                        grounding_metadata['sources'] = chunks
                    
                    if hasattr(gm, 'web_search_queries') and gm.web_search_queries is not None:
                        grounding_metadata['web_search_queries'] = list(gm.web_search_queries)
                    
                    if not grounding_metadata:
                        grounding_metadata = None
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                finish_reason=finish_reason,
                grounding_metadata=grounding_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse response: {str(e)}")
            raise Exception(f"Failed to parse response: {str(e)}")
    
    async def _process_multimodal_attachments(self, contents: list, request: LLMRequest):
        """Process multimodal attachments, add to contents"""
        if request.attachment_infos:
            for attachment_info in request.attachment_infos:
                await self._add_attachment_to_contents(
                    contents, 
                    attachment_info.path, 
                    attachment_info.type, 
                    attachment_info.extract_mode,
                    attachment_info.render_mode
                )
        elif request.attachments:
            for attachment_path in request.attachments:
                attachment_info = await self.attachment_parser.get_attachment_info(attachment_path)
                
                if attachment_info['file_category'] == 'image':
                    await self._add_attachment_to_contents(contents, attachment_path, "image")
                elif attachment_info['file_category'] == 'html':
                    await self._add_attachment_to_contents(
                        contents, 
                        attachment_path, 
                        "html", 
                        request.html_extract_mode
                    )
                else:
                    await self._add_attachment_to_contents(contents, attachment_path, "document")
    
    async def _add_attachment_to_contents(
        self, 
        contents: list, 
        file_path: str, 
        file_type: str, 
        extract_mode: str = "text",
        render_mode: Optional[str] = None
    ):
        """Add attachment to contents"""
        try:
            if not os.path.exists(file_path):
                error_msg = f"\n\nAttachment file does not exist: {file_path}"
                if contents and isinstance(contents[-1], str):
                    contents[-1] += error_msg
                else:
                    contents.append(error_msg)
                return
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            if file_type == "image":
                try:
                    mime_type = self.attachment_parser.image_parser.get_mime_type(file_path)
                    image_part = types.Part.from_bytes(
                        data=file_data,
                        mime_type=mime_type
                    )
                    contents.append(image_part)
                    logger.info(f"Added image Part: {file_path}, mime_type={mime_type}")
                except Exception as e:
                    error_msg = f"\n\nImage processing failed: {file_path} - {str(e)}"
                    if contents and isinstance(contents[-1], str):
                        contents[-1] += error_msg
                    else:
                        contents.append(error_msg)
            
            elif file_type == "html":
                try:
                    html_content = await self.attachment_parser._parse_html(
                        file_data, 
                        file_path, 
                        extract_mode
                    )
                    if html_content:
                        text_content = f"\n\n=== HTML file content ({file_path}) ===\n{html_content}"
                        if contents and isinstance(contents[-1], str):
                            contents[-1] += text_content
                        else:
                            contents.append(text_content)
                        logger.info(f"Added HTML file: {file_path} (mode: {extract_mode})")
                except Exception as e:
                    error_msg = f"\n\nHTML file processing failed: {file_path} - {str(e)}"
                    if contents and isinstance(contents[-1], str):
                        contents[-1] += error_msg
                    else:
                        contents.append(error_msg)
            
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
                                page_label = f"\nSlide {i}:"
                                if contents and isinstance(contents[-1], str):
                                    contents[-1] += page_label
                                else:
                                    contents.append(page_label)
                                
                                image_part = types.Part.from_bytes(
                                    data=img_bytes,
                                    mime_type='image/png'
                                )
                                contents.append(image_part)
                            
                            logger.info(f"Added PPTX rendered images: {file_path} ({len(image_bytes_list)} slides)")
                        except Exception as e:
                            logger.warning(f"PPTX image rendering failed, falling back to text extraction: {str(e)}")
                            content = await self.attachment_parser._parse_office(file_data, file_path)
                            text_content = f"\n\n=== Document content ({file_path}) [rendering failed, using text mode] ===\n{content}"
                            if contents and isinstance(contents[-1], str):
                                contents[-1] += text_content
                            else:
                                contents.append(text_content)
                    
                    elif attachment_info['file_category'] == 'office':
                        content = await self.attachment_parser._parse_office(file_data, file_path)
                        text_content = f"\n\n=== Document content ({file_path}) ===\n{content}"
                        if contents and isinstance(contents[-1], str):
                            contents[-1] += text_content
                        else:
                            contents.append(text_content)
                        logger.info(f"Added document: {file_path}")
                    else:
                        content = await self.attachment_parser._parse_text(file_data, file_path)
                        text_content = f"\n\n=== Document content ({file_path}) ===\n{content}"
                        if contents and isinstance(contents[-1], str):
                            contents[-1] += text_content
                        else:
                            contents.append(text_content)
                        logger.info(f"Added document: {file_path}")
                        
                except Exception as e:
                    error_msg = f"\n\nDocument processing failed: {file_path} - {str(e)}"
                    if contents and isinstance(contents[-1], str):
                        contents[-1] += error_msg
                    else:
                        contents.append(error_msg)
        
        except Exception as e:
            error_msg = f"\n\nAttachment processing failed: {file_path} - {str(e)}"
            if contents and isinstance(contents[-1], str):
                contents[-1] += error_msg
            else:
                contents.append(error_msg)
    
    def get_model_name(self) -> str:
        return self.model
