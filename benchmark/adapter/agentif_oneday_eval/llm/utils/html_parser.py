"""
HTML File Parser
Uses html2text to extract structured content or uses Playwright to render screenshots
"""
import re
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from html2text import HTML2Text
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False

try:
    from .web_renderer import WebRenderer
    RENDERER_SUPPORT = True
except ImportError:
    RENDERER_SUPPORT = False

def _debug_log_html(msg, data=None):
    """No-op hook kept to avoid changing upstream control flow."""
    return None


class HTMLParser:
    """HTML File Parser"""
    
    SUPPORTED_FORMATS = {'.html', '.htm', '.xhtml'}
    
    EXTRACT_MODES = {
        'structured': 'Extract structured content (preserve tag information)',
        'screenshot': 'Use Playwright to render as screenshot'
    }
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a supported HTML format"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def get_supported_modes(cls) -> Dict[str, str]:
        """Get supported extraction modes"""
        return cls.EXTRACT_MODES.copy()
    
    @classmethod
    async def parse_html(cls, file_data: bytes, file_path: str, extract_mode: str = "screenshot") -> str:
        """
        Parse HTML file and extract structured content or render as screenshot
        
        Args:
            file_data: HTML file data
            file_path: File path
            extract_mode: Extraction mode ("structured" or "screenshot")
        
        Returns:
            Parsed HTML content or screenshot description
        """
        # #region agent log
        _debug_log_html("parse_html called", {"file_path": file_path, "extract_mode": extract_mode, "file_data_len": len(file_data), "RENDERER_SUPPORT": RENDERER_SUPPORT})
        # #endregion
        
        # If screenshot mode is selected
        if extract_mode == "screenshot":
            try:
                result = await cls._render_as_screenshot(file_data, file_path)
                # #region agent log
                _debug_log_html("Screenshot rendering successful", {"file_path": file_path, "result_len": len(result)})
                # #endregion
                return result
            except Exception as e:
                # If rendering fails, fall back to text mode
                # #region agent log
                _debug_log_html("Screenshot rendering failed, falling back to text mode", {"file_path": file_path, "error": str(e)})
                # #endregion
                return f"Playwright rendering failed ({str(e)}), falling back to text mode:\n\n" + await cls._parse_as_text(file_data)
        
        # Text mode
        # #region agent log
        _debug_log_html("Using text mode parsing", {"file_path": file_path, "extract_mode": extract_mode})
        # #endregion
        return await cls._parse_as_text(file_data)
    
    @classmethod
    async def _parse_as_text(cls, file_data: bytes) -> str:
        """
        Parse HTML in text mode
        
        Args:
            file_data: HTML file data
            
        Returns:
            Parsed text content
        """
        if not HTML_SUPPORT:
            return "html2text library required to parse HTML files"
        
        try:
            # Try different encodings to decode HTML content
            html_content = cls._decode_html_content(file_data)
            if html_content is None:
                return "HTML file encoding not supported, unable to parse"
            
            # Use html2text to extract structured content
            return cls._extract_structured_content(html_content)
                
        except Exception as e:
            return f"Error parsing HTML file: {str(e)}"
    
    @classmethod
    async def _render_as_screenshot(cls, file_data: bytes, file_path: str) -> str:
        """
        Use Playwright to render HTML as screenshot
        
        Args:
            file_data: HTML file data
            file_path: File path
            
        Returns:
            Screenshot description
        """
        if not RENDERER_SUPPORT:
            raise Exception("WebRenderer not installed")
        
        if not WebRenderer.is_supported():
            raise Exception("Playwright not installed")
        
        # Create temporary file for rendering
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp(prefix='llm_judge_html_')
        temp_html_path = os.path.join(temp_dir, Path(file_path).name)
        
        try:
            # Write HTML content
            with open(temp_html_path, 'wb') as f:
                f.write(file_data)
            
            # Render
            renderer = WebRenderer()
            screenshot_data = await renderer.render_html_file(temp_html_path)
            
            # Convert to base64
            base64_data = base64.b64encode(screenshot_data).decode('utf-8')
            
            return f"HTML file rendered as screenshot (size: {len(screenshot_data)} bytes)\n[Screenshot data: {len(base64_data)} characters base64 encoded]"
            
        finally:
            # Clean up temporary files
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    @classmethod
    def _decode_html_content(cls, file_data: bytes) -> Optional[str]:
        """Try to decode HTML content using different encodings"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return file_data.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        return None
    
    @classmethod
    def _extract_structured_content(cls, html_content: str) -> str:
        """Use html2text to extract structured content"""
        try:
            # Remove base64 encoded data URIs to avoid overly long context
            # Match format: data:[mediatype][;base64],<base64data>
            html_content = re.sub(
                r'data:([^;,]+)(;base64)?,([A-Za-z0-9+/=\s]{100,})',
                r'data:\1 [base64 data omitted]',
                html_content
            )
            
            # Initialize HTML2Text converter
            text_maker = HTML2Text()
            
            # Configure converter parameters
            text_maker.ignore_links = False  # Preserve link information
            text_maker.bypass_tables = False  # Preserve table structure
            text_maker.ignore_images = False  # Preserve image information
            text_maker.body_width = 0  # No line width limit
            text_maker.unicode_snob = True  # Preserve Unicode characters
            text_maker.mark_code = True  # Mark code blocks
            
            # Convert HTML to structured text
            structured_text = text_maker.handle(html_content)
            
            # Clean and optimize output
            # Remove excessive blank lines
            structured_text = re.sub(r'\n{3,}', '\n\n', structured_text)
            
            # Remove leading/trailing whitespace from lines
            lines = [line.strip() for line in structured_text.split('\n')]
            structured_text = '\n'.join(lines)
            
            return structured_text.strip()
            
        except Exception as e:
            return f"Error using html2text: {str(e)}"
    
    
    @classmethod
    async def extract_links(cls, file_data: bytes, file_path: str) -> List[Dict[str, str]]:
        """
        Extract all links from HTML
        
        Args:
            file_data: HTML file data
            file_path: File path
            
        Returns:
            List of link information
        """
        if not HTML_SUPPORT:
            return []
        
        try:
            html_content = cls._decode_html_content(file_data)
            if html_content is None:
                return []
            
            # Use regex to extract links
            links = []
            link_pattern = r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>'
            matches = re.findall(link_pattern, html_content, re.IGNORECASE | re.DOTALL)
            
            for href, text in matches:
                # Clean text content
                text = re.sub(r'<[^>]+>', '', text).strip()
                
                if href and not href.startswith('#'):  # Exclude anchor links
                    links.append({
                        'text': text,
                        'href': href,
                        'title': ''
                    })
            
            return links
            
        except Exception:
            return []
    
    @classmethod
    async def parse_html_for_multimodal(cls, file_data: bytes, file_path: str, css_files: Optional[List[str]] = None, js_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse HTML file for multimodal models
        Render HTML as screenshot and return format suitable for multimodal models
        
        Args:
            file_data: HTML file data
            file_path: File path
            css_files: Optional list of CSS file paths
            js_files: Optional list of JS file paths
            
        Returns:
            Image data format suitable for multimodal models
        """
        if not RENDERER_SUPPORT:
            raise Exception("WebRenderer not installed, unable to render HTML for multimodal models")
        
        if not WebRenderer.is_supported():
            raise Exception("Playwright not installed, unable to render HTML for multimodal models")
        
        # Create temporary file for rendering
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp(prefix='llm_judge_html_')
        temp_html_path = os.path.join(temp_dir, Path(file_path).name)
        
        try:
            # Write HTML content
            with open(temp_html_path, 'wb') as f:
                f.write(file_data)
            
            # If there are CSS/JS files, copy to temporary directory
            temp_css_files = []
            temp_js_files = []
            
            if css_files:
                for css_file in css_files:
                    if os.path.exists(css_file):
                        css_name = Path(css_file).name
                        temp_css_path = os.path.join(temp_dir, css_name)
                        import shutil
                        shutil.copy(css_file, temp_css_path)
                        temp_css_files.append(temp_css_path)
            
            if js_files:
                for js_file in js_files:
                    if os.path.exists(js_file):
                        js_name = Path(js_file).name
                        temp_js_path = os.path.join(temp_dir, js_name)
                        import shutil
                        shutil.copy(js_file, temp_js_path)
                        temp_js_files.append(temp_js_path)
            
            # Render
            renderer = WebRenderer()
            screenshot_data = await renderer.render_html_file(
                temp_html_path,
                css_files=temp_css_files if temp_css_files else None,
                js_files=temp_js_files if temp_js_files else None
            )
            
            # Convert to base64
            base64_data = base64.b64encode(screenshot_data).decode('utf-8')
            
            return {
                'inline_data': {
                    'mime_type': 'image/png',
                    'data': base64_data
                }
            }
            
        finally:
            # Clean up temporary files
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    @classmethod
    async def extract_images(cls, file_data: bytes, file_path: str) -> List[Dict[str, str]]:
        """
        Extract all image information from HTML
        
        Args:
            file_data: HTML file data
            file_path: File path
            
        Returns:
            List of image information
        """
        if not HTML_SUPPORT:
            return []
        
        try:
            html_content = cls._decode_html_content(file_data)
            if html_content is None:
                return []
            
            # Use regex to extract images
            images = []
            img_pattern = r'<img[^>]*src=["\']([^"\']*)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*(?:title=["\']([^"\']*)["\'])?[^>]*>'
            matches = re.findall(img_pattern, html_content, re.IGNORECASE)
            
            for match in matches:
                src = match[0] if len(match) > 0 else ''
                alt = match[1] if len(match) > 1 else ''
                title = match[2] if len(match) > 2 else ''
                
                if src:
                    images.append({
                        'src': src,
                        'alt': alt,
                        'title': title
                    })
            
            return images
            
        except Exception:
            return []
