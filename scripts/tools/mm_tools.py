"""
Multimodal tools for inspecting files as text or images.

Adapted from Flash-Searcher's mm_tools.py.
- TextInspectorTool: reads documents (PDF, XLSX, PPTX, HTML, etc.) as text
- VisualInspectorTool: analyzes images via a vision-capable LLM

Both tools follow the same Tool base class as our existing tools.
"""

import base64
import logging
import mimetypes
import os
from typing import Optional

import requests
from PIL import Image

from .base import Tool
from .mm_tools_utils import MarkdownConverter

logger = logging.getLogger(__name__)


# ── Text Inspector ────────────────────────────────────────────────────

class TextInspectorTool(Tool):
    """Read and inspect text-based files (PDF, XLSX, PPTX, DOCX, HTML, CSV, etc.).

    For simple reads (no question), returns the raw text content.
    For question-based reads, uses the LLM to extract specific information.
    """

    name = "inspect_file_as_text"
    description = (
        "Read and inspect the content of a text-based file. "
        "Supports PDF, Excel (.xlsx/.xls), PowerPoint (.pptx), Word (.docx), "
        "HTML, CSV, JSON, XML, plain text, code files, and ZIP archives. "
        "If a question is provided, the content is summarized to answer it. "
        "For images, use 'inspect_file_as_image' instead."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to inspect.",
        },
        "question": {
            "type": "string",
            "description": "Optional question to answer about the file content.",
            "nullable": True,
        },
    }
    output_type = "string"

    # Class-level converter (shared across instances)
    _md_converter = MarkdownConverter()

    def __init__(self, model=None, text_limit: int = 100000):
        """
        Args:
            model: LLM callable for summarization (messages -> ChatMessage).
            text_limit: Maximum text length before truncation.
        """
        super().__init__()
        self._model = model
        self._text_limit = text_limit
        self._workspace = ""

    @property
    def workspace(self) -> str:
        """Return the current workspace directory path, if configured."""
        return self._workspace

    def set_workspace(self, path: str) -> None:
        """Override the workspace directory for resolving relative file paths."""
        self._workspace = path

    def _resolve_path(self, file_path: str) -> str:
        """Resolve file_path against the configured workspace when needed."""
        if os.path.isabs(file_path) or not self._workspace:
            return file_path
        return os.path.join(self._workspace, file_path)

    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        """Read file content, optionally answering a question about it."""
        resolved_path = self._resolve_path(file_path)

        if not os.path.exists(resolved_path):
            return f"Error: file not found: {file_path}"

        ext = os.path.splitext(resolved_path)[1].lower()

        # Redirect image files to visual inspector
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            return (
                f"Error: '{file_path}' is an image file. "
                "Use the 'inspect_file_as_image' tool instead."
            )

        # Convert file to text
        try:
            result = self._md_converter.convert(resolved_path)
            if result is None or not result.text_content:
                return f"Error: could not extract text from {file_path}"
            content = result.text_content
        except Exception as e:
            return f"Error reading file: {e}"

        # Truncate if too long
        if len(content) > self._text_limit:
            content = content[: self._text_limit] + "\n\n[... content truncated ...]"

        # If no question, return raw content
        if not question:
            return content

        # If question provided and model available, use LLM to answer
        if self._model is None:
            return content

        if len(content) < 4000:
            return content

        # Use LLM to extract relevant information
        try:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert document analyst. "
                                "Answer the user's question based on the file content provided."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"File: {os.path.basename(resolved_path)}\n\n"
                                f"Content:\n{content}\n\n"
                                f"Question: {question}\n\n"
                                "Please provide a detailed answer with:\n"
                                "1. Key findings\n"
                                "2. Relevant details from the document\n"
                                "3. Direct quotes or data if applicable"
                            ),
                        }
                    ],
                },
            ]
            response = self._model(messages)
            return response.content
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return content


# ── Visual Inspector ──────────────────────────────────────────────────

class VisualInspectorTool(Tool):
    """Analyze images using a vision-capable LLM.

    Sends the image (base64-encoded) to the model for visual analysis.
    """

    name = "inspect_file_as_image"
    description = (
        "Analyze an image file using a vision-capable AI model. "
        "Supports JPG, JPEG, PNG, GIF, BMP, and WebP formats. "
        "Use this to describe images, extract text from screenshots, "
        "analyze charts/diagrams, or answer questions about visual content."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the image file to analyze.",
        },
        "question": {
            "type": "string",
            "description": "Optional question about the image. Default: describe the image.",
            "nullable": True,
        },
    }
    output_type = "string"

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    def __init__(self, model=None, api_key: str = "", api_base: str = ""):
        """
        Args:
            model: Not used directly (kept for interface compatibility).
            api_key: API key for the vision model.
            api_base: API base URL (defaults to OpenRouter).
        """
        super().__init__()
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._api_base = (
            api_base
            or os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        )
        self._model_id = os.getenv(
            "VISION_MODEL", "google/gemini-3-flash-preview"
        )
        self._workspace = ""

    @property
    def workspace(self) -> str:
        """Return the current workspace directory path, if configured."""
        return self._workspace

    def set_workspace(self, path: str) -> None:
        """Override the workspace directory for resolving relative file paths."""
        self._workspace = path

    def _resolve_path(self, file_path: str) -> str:
        """Resolve file_path against the configured workspace when needed."""
        if os.path.isabs(file_path) or not self._workspace:
            return file_path
        return os.path.join(self._workspace, file_path)

    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        """Analyze image and return description or answer."""
        resolved_path = self._resolve_path(file_path)

        # Validate file type
        ext = os.path.splitext(resolved_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            return (
                f"Error: unsupported image format '{ext}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        if not os.path.exists(resolved_path):
            return f"Error: file not found: {file_path}"

        # Encode image
        try:
            image_data = self._encode_image(resolved_path)
        except Exception as e:
            return f"Error encoding image: {e}"

        # Build prompt
        prompt = question or "Please write a detailed caption for this image."

        # Determine MIME type
        mime_type = mimetypes.guess_type(resolved_path)[0] or "image/png"

        # Call vision model via OpenAI-compatible API
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key, base_url=self._api_base)
            response = client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling vision model: {e}"

    @staticmethod
    def _encode_image(file_path: str) -> str:
        """Base64-encode an image file."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
