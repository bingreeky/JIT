"""
Web Renderer
Uses Playwright to render HTML files as screenshots
Supports standalone HTML files and frontend projects within archives
"""
import os
import base64
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio

try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_SUPPORT = True
except ImportError:
    PLAYWRIGHT_SUPPORT = False


class WebRenderer:
    """Web Renderer - Uses Playwright to render HTML as screenshots"""
    
    # Default viewport size
    DEFAULT_VIEWPORT = {'width': 1920, 'height': 1080}
    
    # Page load timeout (milliseconds)
    PAGE_TIMEOUT = 30000
    
    # Screenshot format
    SCREENSHOT_FORMAT = 'png'
    
    def __init__(self):
        """Initialize Web Renderer"""
        self.temp_dirs = []  # Track created temporary directories for cleanup
    
    @classmethod
    def is_supported(cls) -> bool:
        """Check if Playwright rendering is supported"""
        return PLAYWRIGHT_SUPPORT
    
    async def render_html_file(
        self, 
        html_path: str, 
        css_files: Optional[List[str]] = None,
        js_files: Optional[List[str]] = None,
        viewport: Optional[Dict[str, int]] = None
    ) -> bytes:
        """
        Render single HTML file as screenshot
        
        Args:
            html_path: HTML file path (absolute path)
            css_files: Optional list of CSS file paths
            js_files: Optional list of JS file paths
            viewport: Optional viewport size configuration
            
        Returns:
            PNG format screenshot data (bytes)
        """
        if not PLAYWRIGHT_SUPPORT:
            raise Exception("Playwright not installed, unable to render HTML file")
        
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=True)
                
                try:
                    # Create page
                    page = await browser.new_page(
                        viewport=viewport or self.DEFAULT_VIEWPORT
                    )
                    
                    # Set timeout
                    page.set_default_timeout(self.PAGE_TIMEOUT)
                    
                    # Load HTML file
                    file_url = f"file://{html_path}"
                    await page.goto(file_url, wait_until='networkidle')
                    
                    # Inject additional CSS and JS resources
                    await self._inject_resources(page, css_files, js_files)
                    
                    # Wait briefly to ensure rendering is complete
                    await asyncio.sleep(1)
                    
                    # Take screenshot
                    screenshot_data = await page.screenshot(
                        full_page=True,
                        type=self.SCREENSHOT_FORMAT
                    )
                    
                    return screenshot_data
                    
                finally:
                    await browser.close()
                    
        except Exception as e:
            raise Exception(f"Failed to render HTML file: {str(e)}")
    
    async def render_html_content(
        self,
        html_content: str,
        css_files: Optional[List[str]] = None,
        js_files: Optional[List[str]] = None,
        viewport: Optional[Dict[str, int]] = None
    ) -> bytes:
        """
        Render HTML content (string) as screenshot
        
        Args:
            html_content: HTML content string
            css_files: Optional list of CSS file paths
            js_files: Optional list of JS file paths
            viewport: Optional viewport size configuration
            
        Returns:
            PNG format screenshot data (bytes)
        """
        if not PLAYWRIGHT_SUPPORT:
            raise Exception("Playwright not installed, unable to render HTML content")
        
        # Create temporary HTML file
        temp_dir = self._setup_temp_directory()
        temp_html_path = os.path.join(temp_dir, "index.html")
        
        try:
            # Write HTML content
            with open(temp_html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Render
            return await self.render_html_file(
                temp_html_path,
                css_files=css_files,
                js_files=js_files,
                viewport=viewport
            )
            
        finally:
            self._cleanup_temp_directory(temp_dir)
    
    async def render_html_project(
        self,
        project_dir: str,
        entry_html: str = "index.html",
        viewport: Optional[Dict[str, int]] = None
    ) -> bytes:
        """
        Render extracted frontend project from archive
        
        Args:
            project_dir: Project root directory path
            entry_html: Entry HTML filename (relative to project_dir)
            viewport: Optional viewport size configuration
            
        Returns:
            PNG format screenshot data (bytes)
        """
        if not PLAYWRIGHT_SUPPORT:
            raise Exception("Playwright not installed, unable to render HTML project")
        
        html_path = os.path.join(project_dir, entry_html)
        
        if not os.path.exists(html_path):
            raise FileNotFoundError(f"Entry HTML file does not exist: {html_path}")
        
        # Auto-discover project CSS and JS files
        css_files, js_files = self._discover_project_resources(project_dir, entry_html)
        
        # Render (no need for additional injection, relative paths in HTML will work automatically)
        return await self.render_html_file(
            html_path,
            css_files=None,  # Resources within project are referenced by HTML itself
            js_files=None,
            viewport=viewport
        )
    
    async def _inject_resources(
        self,
        page: Page,
        css_files: Optional[List[str]] = None,
        js_files: Optional[List[str]] = None
    ) -> None:
        """
        Inject additional CSS and JS resources into page
        
        Args:
            page: Playwright page object
            css_files: List of CSS file paths
            js_files: List of JS file paths
        """
        # Inject CSS files
        if css_files:
            for css_file in css_files:
                if os.path.exists(css_file):
                    try:
                        with open(css_file, 'r', encoding='utf-8') as f:
                            css_content = f.read()
                        await page.add_style_tag(content=css_content)
                    except Exception as e:
                        print(f"Failed to inject CSS file {css_file}: {str(e)}")
        
        # Inject JS files
        if js_files:
            for js_file in js_files:
                if os.path.exists(js_file):
                    try:
                        with open(js_file, 'r', encoding='utf-8') as f:
                            js_content = f.read()
                        await page.add_script_tag(content=js_content)
                    except Exception as e:
                        print(f"Failed to inject JS file {js_file}: {str(e)}")
    
    def _discover_project_resources(
        self,
        project_dir: str,
        entry_html: str
    ) -> tuple[List[str], List[str]]:
        """
        Auto-discover CSS and JS resource files in project
        
        Args:
            project_dir: Project root directory
            entry_html: Entry HTML filename
            
        Returns:
            (css_files, js_files) tuple
        """
        css_files = []
        js_files = []
        
        # Search common resource directories
        resource_dirs = [
            project_dir,
            os.path.join(project_dir, 'css'),
            os.path.join(project_dir, 'styles'),
            os.path.join(project_dir, 'js'),
            os.path.join(project_dir, 'scripts'),
            os.path.join(project_dir, 'assets'),
        ]
        
        for resource_dir in resource_dirs:
            if not os.path.exists(resource_dir):
                continue
                
            for file_name in os.listdir(resource_dir):
                file_path = os.path.join(resource_dir, file_name)
                
                if not os.path.isfile(file_path):
                    continue
                
                # Collect CSS files
                if file_name.endswith('.css'):
                    css_files.append(file_path)
                
                # Collect JS files
                elif file_name.endswith('.js'):
                    js_files.append(file_path)
        
        return css_files, js_files
    
    def _setup_temp_directory(self) -> str:
        """
        Create temporary directory
        
        Returns:
            Temporary directory path
        """
        temp_dir = tempfile.mkdtemp(prefix='llm_judge_render_')
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def _cleanup_temp_directory(self, temp_dir: str) -> None:
        """
        Clean up temporary directory
        
        Args:
            temp_dir: Temporary directory path to clean up
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
        except Exception as e:
            print(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")
    
    def cleanup_all_temp_directories(self) -> None:
        """Clean up all created temporary directories"""
        for temp_dir in self.temp_dirs[:]:  # Copy list to avoid modifying during iteration
            self._cleanup_temp_directory(temp_dir)
    
    def __del__(self):
        """Destructor to ensure cleanup of temporary files"""
        self.cleanup_all_temp_directories()
    
    @classmethod
    async def screenshot_to_base64(cls, screenshot_data: bytes) -> str:
        """
        Convert screenshot data to base64 encoding
        
        Args:
            screenshot_data: Screenshot bytes data
            
        Returns:
            Base64 encoded string
        """
        return base64.b64encode(screenshot_data).decode('utf-8')
    
    @classmethod
    async def render_and_encode(
        cls,
        html_path: str,
        css_files: Optional[List[str]] = None,
        js_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Render HTML and return format suitable for multimodal models
        
        Args:
            html_path: HTML file path
            css_files: Optional list of CSS file paths
            js_files: Optional list of JS file paths
            
        Returns:
            Image data format suitable for multimodal models
        """
        renderer = cls()
        try:
            screenshot_data = await renderer.render_html_file(
                html_path,
                css_files=css_files,
                js_files=js_files
            )
            
            base64_data = await cls.screenshot_to_base64(screenshot_data)
            
            return {
                'inline_data': {
                    'mime_type': 'image/png',
                    'data': base64_data
                }
            }
        finally:
            renderer.cleanup_all_temp_directories()
