"""
Attachment Parser Utility Class
Supports parsing Office files, images, and code files (local mode only)
"""
import os
import base64
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import asyncio
import aiofiles


@dataclass
class ParsedAttachmentResult:
    """Result of parsing an attachment, may include both text and images"""
    filename: str
    text_content: str = ""
    # List of screenshots: each is {"filename": str, "mime_type": str, "base64_data": str}
    screenshots: List[Dict[str, str]] = field(default_factory=list)

# Import standalone parser modules
from .image_parser import ImageParser
from .html_parser import HTMLParser
from .office_parser import OfficeParser
from .text_parser import TextParser
from .notebook_parser import NotebookParser
from .archive_parser import ArchiveParser
from .subtitle_parser import SubtitleParser


class AttachmentParser:
    """Attachment Parser - Unified attachment parsing entry (local mode only)"""
    
    def __init__(self):
        """Initialize attachment parser (no MinIO required)"""
        # Initialize individual parsers
        self.image_parser = ImageParser()
        self.html_parser = HTMLParser()
        self.office_parser = OfficeParser()
        self.text_parser = TextParser()
        self.notebook_parser = NotebookParser()
        self.archive_parser = ArchiveParser(attachment_parser=self)
        self.subtitle_parser = SubtitleParser()
        
        # Build supported extension mapping
        self.supported_extensions = {}
        
        # Add text/code formats (added first, lowest priority)
        for ext in TextParser.SUPPORTED_FORMATS:
            self.supported_extensions[ext] = self._parse_text
        
        # Add image formats
        for ext in ImageParser.SUPPORTED_FORMATS:
            self.supported_extensions[ext] = self._parse_image
        
        # Add Office formats
        for ext in OfficeParser.SUPPORTED_FORMATS:
            self.supported_extensions[ext] = self._parse_office
        
        # Add Notebook formats
        for ext in NotebookParser.SUPPORTED_FORMATS:
            self.supported_extensions[ext] = self._parse_notebook
        
        # Add archive formats
        for ext in ArchiveParser.SUPPORTED_FORMATS:
            self.supported_extensions[ext] = self._parse_archive
        
        # Add subtitle formats
        for ext in SubtitleParser.SUPPORTED_FORMATS:
            self.supported_extensions[ext] = self._parse_subtitle
        
        # Add HTML formats (added last, highest priority)
        for ext in HTMLParser.SUPPORTED_FORMATS:
            self.supported_extensions[ext] = self._parse_html
    
    async def parse_attachments(self, attachment_paths: List[str], html_extract_mode: str = "text") -> str:
        """
        Parse multiple attachments and return merged text content
        
        Args:
            attachment_paths: List of attachment paths (local paths)
            html_extract_mode: HTML file extraction mode
            
        Returns:
            Merged text content
        """
        if not attachment_paths:
            return ""
        
        parsed_contents = []
        
        for path in attachment_paths:
            try:
                content = await self.parse_single_attachment(path, html_extract_mode)
                if content:
                    parsed_contents.append(f"=== Attachment: {Path(path).name} ===\n{content}\n")
            except Exception as e:
                parsed_contents.append(f"=== Attachment: {Path(path).name} ===\nParsing failed: {str(e)}\n")
        
        return "\n".join(parsed_contents)
    
    async def parse_single_attachment(self, path: str, html_extract_mode: str = "text") -> str:
        """
        Parse single attachment (local files only)
        
        Args:
            path: Attachment path
            html_extract_mode: HTML file extraction mode
            
        Returns:
            Parsed text content
        """
        return await self._parse_local_file(path, html_extract_mode)
    
    async def _parse_local_file(self, path: str, html_extract_mode: str = "text") -> str:
        """Parse local file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")
        
        file_ext = Path(path).suffix.lower()
        
        # Read file content
        async with aiofiles.open(path, 'rb') as f:
            file_data = await f.read()
        
        return await self._parse_file_content(file_data, file_ext, path, html_extract_mode)
    
    async def _parse_file_content(self, file_data: bytes, file_ext: str, file_path: str, extract_mode: str = "text") -> str:
        """Parse content based on file type"""
        # Special handling for .tar.gz files
        if file_path.lower().endswith('.tar.gz') or file_path.lower().endswith('.tgz'):
            return await self._parse_archive(file_data, file_path)
        
        if file_ext not in self.supported_extensions:
            return f"Unsupported file type: {file_ext}"
        
        parser_func = self.supported_extensions[file_ext]
        
        # If HTML file, pass extract_mode parameter
        if file_ext in HTMLParser.SUPPORTED_FORMATS:
            return await parser_func(file_data, file_path, extract_mode)
        else:
            return await parser_func(file_data, file_path)
    
    # Parser method implementations
    async def _parse_image(self, file_data: bytes, file_path: str) -> str:
        """Parse image file"""
        return await self.image_parser.parse_image_for_text(file_data, file_path)
    
    async def _parse_html(self, file_data: bytes, file_path: str, extract_mode: str = "screenshot") -> str:
        """Parse HTML file"""
        return await self.html_parser.parse_html(file_data, file_path, extract_mode)
    
    async def parse_html_for_multimodal(self, file_data: bytes, file_path: str, css_files: Optional[List[str]] = None, js_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse HTML for multimodal models"""
        return await self.html_parser.parse_html_for_multimodal(file_data, file_path, css_files, js_files)
    
    async def _parse_office(self, file_data: bytes, file_path: str) -> str:
        """Parse Office document"""
        return await self.office_parser.parse_document(file_data, file_path)
    
    async def _parse_text(self, file_data: bytes, file_path: str) -> str:
        """Parse text/code file"""
        return await self.text_parser.parse_text_file(file_data, file_path, include_metadata=True)
    
    async def _parse_notebook(self, file_data: bytes, file_path: str) -> str:
        """Parse Jupyter Notebook file"""
        return await self.notebook_parser.parse_notebook(file_data, file_path)
    
    async def _parse_archive(self, file_data: bytes, file_path: str) -> str:
        """Parse archive file"""
        return await self.archive_parser.parse_archive(file_data, file_path, recursion_depth=0)
    
    async def _parse_subtitle(self, file_data: bytes, file_path: str) -> str:
        """Parse subtitle file"""
        return await self.subtitle_parser.parse_subtitle(file_data, file_path)
    
    # Special methods for multimodal models
    async def parse_image_for_multimodal(self, file_data: bytes, file_path: str) -> Dict[str, Any]:
        """Parse image for multimodal models"""
        return await self.image_parser.parse_image_for_multimodal(file_data, file_path)
    
    async def parse_pptx_for_multimodal(self, file_data: bytes, file_path: str, dpi: int = 150, max_slides: int = 50) -> List[bytes]:
        """Parse PPTX for multimodal models"""
        return await self.office_parser.parse_pptx_for_images(file_data, file_path, dpi, max_slides)
    
    async def get_attachment_info(self, file_path: str) -> Dict[str, Any]:
        """Get attachment basic information"""
        file_ext = Path(file_path).suffix.lower()
        
        info = {
            'file_name': Path(file_path).name,
            'file_extension': file_ext,
            'is_supported': file_ext in self.supported_extensions
        }
        
        # Determine file type
        if file_path.lower().endswith('.tar.gz') or file_path.lower().endswith('.tgz'):
            info['file_category'] = 'archive'
            info['parser'] = 'ArchiveParser'
        elif file_ext in ImageParser.SUPPORTED_FORMATS:
            info['file_category'] = 'image'
            info['parser'] = 'ImageParser'
        elif file_ext in HTMLParser.SUPPORTED_FORMATS:
            info['file_category'] = 'html'
            info['parser'] = 'HTMLParser'
        elif file_ext in OfficeParser.SUPPORTED_FORMATS:
            info['file_category'] = 'office'
            info['parser'] = 'OfficeParser'
            info['supports_image_rendering'] = file_ext == '.pptx'
        elif file_ext in NotebookParser.SUPPORTED_FORMATS:
            info['file_category'] = 'notebook'
            info['parser'] = 'NotebookParser'
        elif file_ext in ArchiveParser.SUPPORTED_FORMATS:
            info['file_category'] = 'archive'
            info['parser'] = 'ArchiveParser'
        elif file_ext in TextParser.SUPPORTED_FORMATS:
            info['file_category'] = 'text'
            info['parser'] = 'TextParser'
        elif file_ext in SubtitleParser.SUPPORTED_FORMATS:
            info['file_category'] = 'subtitle'
            info['parser'] = 'SubtitleParser'
        else:
            info['file_category'] = 'unknown'
            info['parser'] = None
        
        return info
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.supported_extensions.keys())
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file is supported for parsing"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_extensions
    
    async def parse_attachment_with_screenshots(self, path: str, html_extract_mode: str = "screenshot") -> ParsedAttachmentResult:
        """
        Parse attachment and return both text content and any screenshots
        
        This method is designed for multimodal LLM scoring:
        - For HTML files: renders as screenshot and returns image data
        - For ZIP/archive files: renders contained HTML files as screenshots
        - For other files: returns text content only
        
        Args:
            path: Attachment file path
            html_extract_mode: HTML extraction mode (default "screenshot")
            
        Returns:
            ParsedAttachmentResult with text_content and screenshots list
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")
        
        filename = Path(path).name
        file_ext = Path(path).suffix.lower()
        
        # Read file content
        async with aiofiles.open(path, 'rb') as f:
            file_data = await f.read()
        
        result = ParsedAttachmentResult(filename=filename)
        
        # Handle HTML files - render as screenshot
        if file_ext in HTMLParser.SUPPORTED_FORMATS:
            try:
                # Try to get screenshot for multimodal
                multimodal_data = await self.html_parser.parse_html_for_multimodal(file_data, path)
                if multimodal_data and 'inline_data' in multimodal_data:
                    result.screenshots.append({
                        "filename": filename,
                        "mime_type": multimodal_data['inline_data']['mime_type'],
                        "base64_data": multimodal_data['inline_data']['data']
                    })
                    result.text_content = f"HTML file rendered as screenshot: {filename}"
                else:
                    # Fallback to text
                    result.text_content = await self._parse_html(file_data, path, html_extract_mode)
            except Exception as e:
                # Fallback to text parsing
                result.text_content = await self._parse_html(file_data, path, "text")
        
        # Handle archive files - may contain HTML projects
        elif file_ext in ArchiveParser.SUPPORTED_FORMATS or path.lower().endswith('.tar.gz') or path.lower().endswith('.tgz'):
            # Parse archive (this will also render HTML files inside)
            full_text_content = await self._parse_archive(file_data, path)
            
            # Get screenshots from the archive parser
            screenshots = self.archive_parser.get_last_screenshots()
            for screenshot in screenshots:
                screenshot_data = screenshot.get('data', b'')
                if screenshot_data:
                    result.screenshots.append({
                        "filename": screenshot.get('filename', 'screenshot.png'),
                        "mime_type": "image/png",
                        "base64_data": base64.b64encode(screenshot_data).decode('utf-8')
                    })
            
            # If we have screenshots, don't include the full text content (it's too large)
            # Just include a summary to reduce payload size
            if result.screenshots:
                # Extract just the summary part from the parsed content
                lines = full_text_content.split('\n')
                summary_lines = []
                for line in lines[:30]:  # First 30 lines contain archive info and HTML render results
                    summary_lines.append(line)
                    if "Parsing complete:" in line:
                        break
                result.text_content = '\n'.join(summary_lines)
                result.text_content += f"\n\n[Note: Full archive content omitted. {len(result.screenshots)} HTML screenshot(s) provided for visual evaluation.]"
            else:
                # No screenshots, include full text content
                result.text_content = full_text_content
        
        # Handle other file types - text only
        else:
            result.text_content = await self._parse_file_content(file_data, file_ext, path, html_extract_mode)
        
        return result
