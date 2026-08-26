"""
LLM Related Data Type Definitions
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """LLM Message"""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")


class AttachmentInfo(BaseModel):
    """Attachment Information"""
    path: str = Field(..., description="Attachment path")
    type: str = Field(..., description="Attachment type: image, html, document, code, etc.")
    extract_mode: Optional[str] = Field("text", description="HTML extraction mode: text, structured, summary")
    render_mode: Optional[str] = Field(None, description="PPTX rendering mode: text (text extraction), image (render as images)")


class ImageAttachment(BaseModel):
    """Image Attachment for Multimodal Models"""
    filename: str = Field(..., description="Original filename")
    mime_type: str = Field("image/png", description="Image MIME type")
    base64_data: str = Field(..., description="Base64 encoded image data")


class LLMRequest(BaseModel):
    """LLM Request"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., description="Model name, e.g., Gemini, ChatGPT, etc.")
    messages: List[LLMMessage] = Field(..., description="Message list")
    attachments: Optional[List[str]] = Field(None, description="Attachment path list, supports local paths")
    attachment_infos: Optional[List[AttachmentInfo]] = Field(None, description="Detailed attachment information list")
    image_attachments: Optional[List[ImageAttachment]] = Field(None, description="Image attachments for multimodal models (e.g., HTML screenshots)")
    temperature: float = Field(0.1, description="Temperature parameter")
    max_tokens: Optional[int] = Field(None, description="Maximum token count")
    parse_attachments: bool = Field(True, description="Whether to parse attachment content as context")
    html_extract_mode: str = Field("text", description="HTML content extraction mode: text, structured, summary")
    use_grounding: Optional[bool] = Field(None, description="Whether to enable Google Search grounding (Gemini models only), None means use global config")


class LLMResponse(BaseModel):
    """LLM Standard Response"""
    content: str = Field(..., description="Response content")
    usage: Optional[Dict[str, Any]] = Field(None, description="Usage statistics")
    model: str = Field(..., description="Actual model used")
    finish_reason: str = Field("stop", description="Finish reason")
    grounding_metadata: Optional[Dict[str, Any]] = Field(None, description="Google Search grounding metadata, including search queries, source links, etc.")
