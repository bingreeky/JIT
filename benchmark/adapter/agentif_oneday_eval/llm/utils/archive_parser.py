"""
Archive File Parser
Supports parsing ZIP and TAR.GZ format archive files
Supports detecting and rendering HTML frontend projects within archives
"""
import os
import zipfile
import tarfile
import tempfile
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .attachment_parser import AttachmentParser

try:
    from .web_renderer import WebRenderer
    RENDERER_SUPPORT = True
except ImportError:
    RENDERER_SUPPORT = False

def _debug_log(msg, data=None):
    """No-op hook kept to avoid changing upstream control flow."""
    return None


class ArchiveParser:
    """Archive File Parser"""
    
    SUPPORTED_FORMATS = {'.zip', '.tar', '.gz', '.tgz', '.tar.gz'}
    
    # Maximum recursion depth to prevent infinite recursion
    MAX_RECURSION_DEPTH = 5
    
    # Maximum single file size (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    def __init__(self, attachment_parser: Optional['AttachmentParser'] = None):
        """
        Initialize archive file parser
        
        Args:
            attachment_parser: AttachmentParser instance for recursively parsing files within archives
        """
        self.attachment_parser = attachment_parser
        # Store screenshots from the last parse operation
        self._last_screenshots: List[Dict[str, Any]] = []
    
    def get_last_screenshots(self) -> List[Dict[str, Any]]:
        """
        Get screenshots from the last parse operation
        
        Returns:
            List of {"filename": str, "data": bytes} dictionaries
        """
        return self._last_screenshots
    
    def clear_screenshots(self):
        """Clear stored screenshots"""
        self._last_screenshots = []
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a supported archive format"""
        file_path_lower = file_path.lower()
        # Check special case for .tar.gz
        if file_path_lower.endswith('.tar.gz'):
            return True
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def _is_tar_gz_file(cls, file_path: str) -> bool:
        """Check if file is a tar.gz file"""
        return file_path.lower().endswith('.tar.gz') or file_path.lower().endswith('.tgz')
    
    async def parse_archive(
        self, 
        file_data: bytes, 
        file_path: str, 
        recursion_depth: int = 0,
        render_html: bool = True
    ) -> str:
        """
        Parse archive file
        
        Args:
            file_data: Archive file data
            file_path: File path
            recursion_depth: Current recursion depth
            render_html: Whether to render HTML files
            
        Returns:
            Parsed text content
        """
        # Clear screenshots from previous parse at top level
        if recursion_depth == 0:
            self._last_screenshots = []
        
        # Check recursion depth
        if recursion_depth >= self.MAX_RECURSION_DEPTH:
            return f"Archive nesting level exceeds limit ({self.MAX_RECURSION_DEPTH} levels)"
        
        try:
            file_name = Path(file_path).name
            
            # Determine archive file type
            if file_path.lower().endswith('.zip'):
                return await self._parse_zip(file_data, file_name, recursion_depth, render_html)
            elif self._is_tar_gz_file(file_path):
                return await self._parse_tar_gz(file_data, file_name, recursion_depth, render_html)
            elif file_path.lower().endswith('.tar'):
                return await self._parse_tar(file_data, file_name, recursion_depth, render_html)
            else:
                return f"Unsupported archive format: {Path(file_path).suffix}"
                
        except Exception as e:
            return f"Failed to parse archive file: {str(e)}"
    
    async def _parse_zip(
        self, 
        file_data: bytes, 
        file_name: str, 
        recursion_depth: int,
        render_html: bool = True
    ) -> str:
        """Parse ZIP file"""
        try:
            result_parts = []
            result_parts.append(f"Archive file: {file_name} (ZIP format)")
            result_parts.append("=" * 60)
            result_parts.append("")
            
            # Open ZIP file in memory
            with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_file:
                # Get file list
                file_list = zip_file.namelist()
                
                # Filter out directories and hidden files
                valid_files = [
                    f for f in file_list 
                    if not f.endswith('/') and not Path(f).name.startswith('.')
                ]
                
                # Detect HTML files
                html_files = self._detect_html_files(valid_files)
                
                # #region agent log
                _debug_log("ZIP parsing started", {"file_name": file_name, "valid_files_count": len(valid_files), "html_files": html_files, "RENDERER_SUPPORT": RENDERER_SUPPORT, "render_html": render_html})
                # #endregion
                
                result_parts.append(f"Total files in archive: {len(valid_files)}")
                if html_files:
                    result_parts.append(f"Detected {len(html_files)} HTML files")
                result_parts.append("")
                
                # If contains HTML and rendering is needed, try to render the whole project
                html_render_success = False
                if html_files and render_html and RENDERER_SUPPORT:
                    # #region agent log
                    web_renderer_supported = WebRenderer.is_supported() if RENDERER_SUPPORT else False
                    _debug_log("Attempting HTML rendering", {"html_files": html_files, "WebRenderer.is_supported": web_renderer_supported})
                    # #endregion
                    try:
                        rendered_result, screenshots = await self._render_zip_htmls(file_data, file_name, html_files, zip_file)
                        if rendered_result:
                            result_parts.append(rendered_result)
                            result_parts.append("")
                            html_render_success = True
                            # Store screenshots for later retrieval
                            self._last_screenshots.extend(screenshots)
                            # #region agent log
                            _debug_log("HTML rendering successful", {"rendered_result_len": len(rendered_result), "screenshots_count": len(screenshots)})
                            # #endregion
                    except Exception as e:
                        result_parts.append(f"Failed to render HTML project: {str(e)}, continuing with text parsing")
                        result_parts.append("")
                        # #region agent log
                        _debug_log("HTML rendering failed", {"error": str(e)})
                        # #endregion
                else:
                    # #region agent log
                    _debug_log("Skipping HTML rendering", {"html_files": bool(html_files), "render_html": render_html, "RENDERER_SUPPORT": RENDERER_SUPPORT})
                    # #endregion
                
                # Parse each file
                parsed_count = 0
                skipped_count = 0
                
                for file_path in valid_files:
                    try:
                        # If HTML rendering succeeded, skip text parsing for HTML/CSS/JS files (screenshot already includes styles)
                        file_ext = Path(file_path).suffix.lower()
                        if html_render_success and file_ext in {'.html', '.htm', '.xhtml', '.css', '.js'}:
                            # #region agent log
                            _debug_log("Skipping frontend file (already rendered)", {"file_path": file_path, "file_ext": file_ext})
                            # #endregion
                            skipped_count += 1
                            continue
                        
                        # Get file info
                        file_info = zip_file.getinfo(file_path)
                        
                        # Check file size
                        if file_info.file_size > self.MAX_FILE_SIZE:
                            result_parts.append(f"📁 {file_path}")
                            result_parts.append(f"   File too large ({file_info.file_size} bytes), skipping parsing")
                            result_parts.append("")
                            skipped_count += 1
                            continue
                        
                        # Read file content
                        file_content = zip_file.read(file_path)
                        
                        # Parse file content
                        parsed_content = await self._parse_file_in_archive(
                            file_content, 
                            file_path, 
                            recursion_depth
                        )
                        
                        if parsed_content:
                            result_parts.append(f"📁 {file_path}")
                            result_parts.append("-" * 60)
                            result_parts.append(parsed_content)
                            result_parts.append("")
                            parsed_count += 1
                        else:
                            skipped_count += 1
                            
                    except Exception as e:
                        result_parts.append(f"📁 {file_path}")
                        result_parts.append(f"   Parsing failed: {str(e)}")
                        result_parts.append("")
                        skipped_count += 1
                
                result_parts.append("=" * 60)
                result_parts.append(f"Parsing complete: {parsed_count} successful, {skipped_count} skipped")
            
            return "\n".join(result_parts)
            
        except zipfile.BadZipFile:
            return f"ZIP file corrupted or invalid format: {file_name}"
        except Exception as e:
            return f"Failed to parse ZIP file: {str(e)}"
    
    async def _parse_tar_gz(
        self, 
        file_data: bytes, 
        file_name: str, 
        recursion_depth: int,
        render_html: bool = True
    ) -> str:
        """Parse TAR.GZ file"""
        try:
            result_parts = []
            result_parts.append(f"Archive file: {file_name} (TAR.GZ format)")
            result_parts.append("=" * 60)
            result_parts.append("")
            
            # Open TAR.GZ file in memory
            with tarfile.open(fileobj=BytesIO(file_data), mode='r:gz') as tar_file:
                # Get file list
                members = tar_file.getmembers()
                
                # Filter out directories and hidden files
                valid_files = [
                    m for m in members 
                    if m.isfile() and not Path(m.name).name.startswith('.')
                ]
                
                # Detect HTML files
                html_files = self._detect_html_files([m.name for m in valid_files])
                
                result_parts.append(f"Total files in archive: {len(valid_files)}")
                if html_files:
                    result_parts.append(f"Detected {len(html_files)} HTML files")
                result_parts.append("")
                
                # If contains HTML and rendering is needed, try to render the whole project
                html_render_success = False
                if html_files and render_html and RENDERER_SUPPORT:
                    try:
                        rendered_result, screenshots = await self._render_tar_htmls(file_data, file_name, html_files, tar_file)
                        if rendered_result:
                            result_parts.append(rendered_result)
                            result_parts.append("")
                            html_render_success = True
                            # Store screenshots for later retrieval
                            self._last_screenshots.extend(screenshots)
                    except Exception as e:
                        result_parts.append(f"Failed to render HTML project: {str(e)}, continuing with text parsing")
                        result_parts.append("")
                
                # Parse each file
                parsed_count = 0
                skipped_count = 0
                
                for member in valid_files:
                    try:
                        # If HTML rendering succeeded, skip text parsing for HTML/CSS/JS files
                        file_ext = Path(member.name).suffix.lower()
                        if html_render_success and file_ext in {'.html', '.htm', '.xhtml', '.css', '.js'}:
                            skipped_count += 1
                            continue
                        
                        # Check file size
                        if member.size > self.MAX_FILE_SIZE:
                            result_parts.append(f"📁 {member.name}")
                            result_parts.append(f"   File too large ({member.size} bytes), skipping parsing")
                            result_parts.append("")
                            skipped_count += 1
                            continue
                        
                        # Read file content
                        file_obj = tar_file.extractfile(member)
                        if file_obj is None:
                            skipped_count += 1
                            continue
                        
                        file_content = file_obj.read()
                        file_obj.close()
                        
                        # Parse file content
                        parsed_content = await self._parse_file_in_archive(
                            file_content, 
                            member.name, 
                            recursion_depth
                        )
                        
                        if parsed_content:
                            result_parts.append(f"📁 {member.name}")
                            result_parts.append("-" * 60)
                            result_parts.append(parsed_content)
                            result_parts.append("")
                            parsed_count += 1
                        else:
                            skipped_count += 1
                            
                    except Exception as e:
                        result_parts.append(f"📁 {member.name}")
                        result_parts.append(f"   Parsing failed: {str(e)}")
                        result_parts.append("")
                        skipped_count += 1
                
                result_parts.append("=" * 60)
                result_parts.append(f"Parsing complete: {parsed_count} successful, {skipped_count} skipped")
            
            return "\n".join(result_parts)
            
        except tarfile.ReadError:
            return f"TAR.GZ file corrupted or invalid format: {file_name}"
        except Exception as e:
            return f"Failed to parse TAR.GZ file: {str(e)}"
    
    async def _parse_tar(
        self, 
        file_data: bytes, 
        file_name: str, 
        recursion_depth: int,
        render_html: bool = True
    ) -> str:
        """Parse TAR file"""
        try:
            result_parts = []
            result_parts.append(f"Archive file: {file_name} (TAR format)")
            result_parts.append("=" * 60)
            result_parts.append("")
            
            # Open TAR file in memory
            with tarfile.open(fileobj=BytesIO(file_data), mode='r:') as tar_file:
                # Get file list
                members = tar_file.getmembers()
                
                # Filter out directories and hidden files
                valid_files = [
                    m for m in members 
                    if m.isfile() and not Path(m.name).startswith('.')
                ]
                
                # Detect HTML files
                html_files = self._detect_html_files([m.name for m in valid_files])
                
                result_parts.append(f"Total files in archive: {len(valid_files)}")
                if html_files:
                    result_parts.append(f"Detected {len(html_files)} HTML files")
                result_parts.append("")
                
                # If contains HTML and rendering is needed, try to render the whole project
                html_render_success = False
                if html_files and render_html and RENDERER_SUPPORT:
                    try:
                        rendered_result, screenshots = await self._render_tar_htmls(file_data, file_name, html_files, tar_file)
                        if rendered_result:
                            result_parts.append(rendered_result)
                            result_parts.append("")
                            html_render_success = True
                            # Store screenshots for later retrieval
                            self._last_screenshots.extend(screenshots)
                    except Exception as e:
                        result_parts.append(f"Failed to render HTML project: {str(e)}, continuing with text parsing")
                        result_parts.append("")
                
                # Parse each file
                parsed_count = 0
                skipped_count = 0
                
                for member in valid_files:
                    try:
                        # If HTML rendering succeeded, skip text parsing for HTML/CSS/JS files
                        file_ext = Path(member.name).suffix.lower()
                        if html_render_success and file_ext in {'.html', '.htm', '.xhtml', '.css', '.js'}:
                            skipped_count += 1
                            continue
                        
                        # Check file size
                        if member.size > self.MAX_FILE_SIZE:
                            result_parts.append(f"📁 {member.name}")
                            result_parts.append(f"   File too large ({member.size} bytes), skipping parsing")
                            result_parts.append("")
                            skipped_count += 1
                            continue
                        
                        # Read file content
                        file_obj = tar_file.extractfile(member)
                        if file_obj is None:
                            skipped_count += 1
                            continue
                        
                        file_content = file_obj.read()
                        file_obj.close()
                        
                        # Parse file content
                        parsed_content = await self._parse_file_in_archive(
                            file_content, 
                            member.name, 
                            recursion_depth
                        )
                        
                        if parsed_content:
                            result_parts.append(f"📁 {member.name}")
                            result_parts.append("-" * 60)
                            result_parts.append(parsed_content)
                            result_parts.append("")
                            parsed_count += 1
                        else:
                            skipped_count += 1
                            
                    except Exception as e:
                        result_parts.append(f"📁 {member.name}")
                        result_parts.append(f"   Parsing failed: {str(e)}")
                        result_parts.append("")
                        skipped_count += 1
                
                result_parts.append("=" * 60)
                result_parts.append(f"Parsing complete: {parsed_count} successful, {skipped_count} skipped")
            
            return "\n".join(result_parts)
            
        except tarfile.ReadError:
            return f"TAR file corrupted or invalid format: {file_name}"
        except Exception as e:
            return f"Failed to parse TAR file: {str(e)}"
    
    async def _parse_file_in_archive(
        self, 
        file_data: bytes, 
        file_path: str, 
        recursion_depth: int
    ) -> str:
        """
        Parse single file within archive
        
        Args:
            file_data: File data
            file_path: File path
            recursion_depth: Current recursion depth
            
        Returns:
            Parsed text content
        """
        if not self.attachment_parser:
            return f"File type: {Path(file_path).suffix} (no parser)"
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Check if it's an archive file, if so recursively parse
            if self.is_supported(file_path):
                return await self.parse_archive(file_data, file_path, recursion_depth + 1)
            
            # Check if file is supported
            if not self.attachment_parser.is_supported(file_path):
                return f"Unsupported file type: {file_ext}"
            
            # #region agent log
            is_html = file_ext in {'.html', '.htm', '.xhtml'}
            _debug_log("Parsing file in archive", {"file_path": file_path, "file_ext": file_ext, "is_html": is_html, "file_data_len": len(file_data)})
            # #endregion
            
            # Use AttachmentParser to parse file content
            content = await self.attachment_parser._parse_file_content(
                file_data, 
                file_ext, 
                file_path,
                extract_mode="text"
            )
            
            # #region agent log
            if is_html:
                _debug_log("HTML file parsing result", {"file_path": file_path, "content_len": len(content) if content else 0, "content_preview": content[:200] if content else ""})
            # #endregion
            
            return content
            
        except Exception as e:
            return f"Failed to parse file: {str(e)}"
    
    @classmethod
    async def get_archive_info(cls, file_data: bytes, file_path: str) -> Dict[str, Any]:
        """
        Get archive file basic information
        
        Args:
            file_data: Archive file data
            file_path: File path
            
        Returns:
            Archive file information dictionary
        """
        try:
            file_name = Path(file_path).name
            info = {
                'file_name': file_name,
                'file_size': len(file_data)
            }
            
            if file_path.lower().endswith('.zip'):
                with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_file:
                    file_list = zip_file.namelist()
                    valid_files = [f for f in file_list if not f.endswith('/')]
                    info.update({
                        'format': 'ZIP',
                        'total_files': len(valid_files),
                        'file_list': valid_files[:20]  # Only return first 20 filenames
                    })
            
            elif cls._is_tar_gz_file(file_path):
                with tarfile.open(fileobj=BytesIO(file_data), mode='r:gz') as tar_file:
                    members = tar_file.getmembers()
                    valid_files = [m.name for m in members if m.isfile()]
                    info.update({
                        'format': 'TAR.GZ',
                        'total_files': len(valid_files),
                        'file_list': valid_files[:20]
                    })
            
            elif file_path.lower().endswith('.tar'):
                with tarfile.open(fileobj=BytesIO(file_data), mode='r:') as tar_file:
                    members = tar_file.getmembers()
                    valid_files = [m.name for m in members if m.isfile()]
                    info.update({
                        'format': 'TAR',
                        'total_files': len(valid_files),
                        'file_list': valid_files[:20]
                    })
            
            return info
            
        except Exception as e:
            return {
                'file_name': Path(file_path).name,
                'error': f"Failed to get archive file info: {str(e)}"
            }
    
    @classmethod
    def _detect_html_files(cls, file_list: List[str]) -> List[str]:
        """
        Detect HTML files in file list
        
        Args:
            file_list: List of file paths
            
        Returns:
            List of HTML file paths
        """
        html_files = []
        for file_path in file_list:
            ext = Path(file_path).suffix.lower()
            if ext in {'.html', '.htm', '.xhtml'}:
                html_files.append(file_path)
        return html_files
    
    async def _render_zip_htmls(
        self,
        file_data: bytes,
        file_name: str,
        html_files: List[str],
        zip_file: zipfile.ZipFile
    ) -> tuple:
        """
        Render HTML files in ZIP archive
        
        Args:
            file_data: Archive file data
            file_name: File name
            html_files: List of HTML file paths
            zip_file: Opened ZipFile object
            
        Returns:
            Tuple of (text_description, screenshots_list)
            screenshots_list: List of {"filename": str, "data": bytes}
        """
        if not RENDERER_SUPPORT or not WebRenderer.is_supported():
            return "", []
        
        # Create temporary directory to extract entire project
        temp_dir = tempfile.mkdtemp(prefix='llm_judge_zip_')
        screenshots = []
        
        try:
            # Extract all files to temporary directory
            zip_file.extractall(temp_dir)
            
            # Render each HTML file
            result_parts = []
            result_parts.append("🎨 HTML project rendering results:")
            result_parts.append("-" * 60)
            
            renderer = WebRenderer()
            
            for html_file in html_files:
                try:
                    html_path = os.path.join(temp_dir, html_file)
                    
                    if not os.path.exists(html_path):
                        continue
                    
                    # Render HTML
                    screenshot_data = await renderer.render_html_project(
                        temp_dir,
                        entry_html=html_file
                    )
                    
                    result_parts.append(f"✓ {html_file}: Rendered ({len(screenshot_data)} bytes)")
                    
                    # Store screenshot data
                    screenshots.append({
                        "filename": html_file,
                        "data": screenshot_data
                    })
                    
                except Exception as e:
                    result_parts.append(f"✗ {html_file}: Rendering failed - {str(e)}")
            
            return "\n".join(result_parts), screenshots
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    async def _render_tar_htmls(
        self,
        file_data: bytes,
        file_name: str,
        html_files: List[str],
        tar_file: tarfile.TarFile
    ) -> tuple:
        """
        Render HTML files in TAR archive
        
        Args:
            file_data: Archive file data
            file_name: File name
            html_files: List of HTML file paths
            tar_file: Opened TarFile object
            
        Returns:
            Tuple of (text_description, screenshots_list)
            screenshots_list: List of {"filename": str, "data": bytes}
        """
        if not RENDERER_SUPPORT or not WebRenderer.is_supported():
            return "", []
        
        # Create temporary directory to extract entire project
        temp_dir = tempfile.mkdtemp(prefix='llm_judge_tar_')
        screenshots = []
        
        try:
            # Extract all files to temporary directory
            tar_file.extractall(temp_dir)
            
            # Render each HTML file
            result_parts = []
            result_parts.append("🎨 HTML project rendering results:")
            result_parts.append("-" * 60)
            
            renderer = WebRenderer()
            
            for html_file in html_files:
                try:
                    html_path = os.path.join(temp_dir, html_file)
                    
                    if not os.path.exists(html_path):
                        continue
                    
                    # Render HTML
                    screenshot_data = await renderer.render_html_project(
                        temp_dir,
                        entry_html=html_file
                    )
                    
                    result_parts.append(f"✓ {html_file}: Rendered ({len(screenshot_data)} bytes)")
                    
                    # Store screenshot data
                    screenshots.append({
                        "filename": html_file,
                        "data": screenshot_data
                    })
                    
                except Exception as e:
                    result_parts.append(f"✗ {html_file}: Rendering failed - {str(e)}")
            
            return "\n".join(result_parts), screenshots
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
