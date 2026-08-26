"""
Jupyter Notebook File Parser
Supports parsing structured content from .ipynb files
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class NotebookParser:
    """Jupyter Notebook Parser"""
    
    SUPPORTED_FORMATS = {'.ipynb'}
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a supported Notebook format"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    async def parse_notebook(cls, file_data: bytes, file_path: str) -> str:
        """
        Parse Jupyter Notebook file
        
        Args:
            file_data: Notebook file data
            file_path: File path
            
        Returns:
            Parsed structured text content
        """
        try:
            # Parse JSON content
            notebook_content = json.loads(file_data.decode('utf-8'))
            
            # Get basic info
            file_name = Path(file_path).name
            metadata = notebook_content.get('metadata', {})
            cells = notebook_content.get('cells', [])
            
            # Build result
            result_parts = []
            
            # Add file info
            result_parts.append(f"Jupyter Notebook: {file_name}")
            result_parts.append(f"Total cells: {len(cells)}")
            
            # Get language info
            language_info = metadata.get('language_info', {})
            language = language_info.get('name', 'python')
            if language:
                result_parts.append(f"Programming language: {language}")
            
            result_parts.append("")
            result_parts.append("=" * 60)
            result_parts.append("")
            
            # Parse each cell
            for i, cell in enumerate(cells, 1):
                cell_type = cell.get('cell_type', 'unknown')
                
                if cell_type == 'markdown':
                    cell_content = cls._parse_markdown_cell(cell, i)
                elif cell_type == 'code':
                    cell_content = cls._parse_code_cell(cell, i, language)
                elif cell_type == 'raw':
                    cell_content = cls._parse_raw_cell(cell, i)
                else:
                    cell_content = f"[Cell {i}] Unknown type: {cell_type}"
                
                if cell_content:
                    result_parts.append(cell_content)
                    result_parts.append("")
            
            return "\n".join(result_parts)
            
        except json.JSONDecodeError as e:
            return f"Failed to parse Notebook file: JSON format error - {str(e)}"
        except Exception as e:
            return f"Failed to parse Notebook file: {str(e)}"
    
    @classmethod
    def _parse_markdown_cell(cls, cell: Dict[str, Any], index: int) -> str:
        """Parse Markdown cell"""
        source = cell.get('source', [])
        
        # source can be a list of strings or a single string
        if isinstance(source, list):
            content = ''.join(source)
        else:
            content = source
        
        if not content.strip():
            return ""
        
        result = [
            f"[Cell {index} - Markdown]",
            "-" * 40,
            content.strip()
        ]
        
        return "\n".join(result)
    
    @classmethod
    def _parse_code_cell(cls, cell: Dict[str, Any], index: int, language: str = 'python') -> str:
        """Parse code cell"""
        source = cell.get('source', [])
        outputs = cell.get('outputs', [])
        execution_count = cell.get('execution_count')
        
        # source can be a list of strings or a single string
        if isinstance(source, list):
            code_content = ''.join(source)
        else:
            code_content = source
        
        if not code_content.strip() and not outputs:
            return ""
        
        result = []
        
        # Add cell header
        exec_info = f" (execution count: {execution_count})" if execution_count else ""
        result.append(f"[Cell {index} - Code{exec_info}]")
        result.append("-" * 40)
        
        # Add code content
        if code_content.strip():
            result.append(f"```{language}")
            result.append(code_content.strip())
            result.append("```")
        
        # Add output content
        if outputs:
            result.append("")
            result.append("Output:")
            for output in outputs:
                output_content = cls._parse_output(output)
                if output_content:
                    result.append(output_content)
        
        return "\n".join(result)
    
    @classmethod
    def _parse_raw_cell(cls, cell: Dict[str, Any], index: int) -> str:
        """Parse raw cell"""
        source = cell.get('source', [])
        
        # source can be a list of strings or a single string
        if isinstance(source, list):
            content = ''.join(source)
        else:
            content = source
        
        if not content.strip():
            return ""
        
        result = [
            f"[Cell {index} - Raw]",
            "-" * 40,
            content.strip()
        ]
        
        return "\n".join(result)
    
    @classmethod
    def _parse_output(cls, output: Dict[str, Any]) -> str:
        """Parse cell output"""
        output_type = output.get('output_type', 'unknown')
        
        if output_type == 'stream':
            # Standard output or error output
            stream_name = output.get('name', 'stdout')
            text = output.get('text', [])
            if isinstance(text, list):
                text_content = ''.join(text)
            else:
                text_content = text
            
            if text_content.strip():
                return f"[{stream_name}]\n{text_content.strip()}"
        
        elif output_type == 'execute_result':
            # Execution result
            data = output.get('data', {})
            return cls._parse_output_data(data, 'execute_result')
        
        elif output_type == 'display_data':
            # Display data
            data = output.get('data', {})
            return cls._parse_output_data(data, 'display_data')
        
        elif output_type == 'error':
            # Error information
            ename = output.get('ename', 'Error')
            evalue = output.get('evalue', '')
            traceback = output.get('traceback', [])
            
            error_parts = [f"[Error: {ename}]"]
            if evalue:
                error_parts.append(f"Error message: {evalue}")
            if traceback:
                # Clean ANSI escape sequences
                traceback_text = '\n'.join(traceback)
                import re
                traceback_text = re.sub(r'\x1b\[[0-9;]*m', '', traceback_text)
                error_parts.append(f"Stack trace:\n{traceback_text}")
            
            return "\n".join(error_parts)
        
        return ""
    
    @classmethod
    def _parse_output_data(cls, data: Dict[str, Any], output_type: str) -> str:
        """Parse output data"""
        result_parts = []
        
        # Prioritize text output
        if 'text/plain' in data:
            text = data['text/plain']
            if isinstance(text, list):
                text_content = ''.join(text)
            else:
                text_content = text
            result_parts.append(f"[{output_type}]")
            result_parts.append(text_content.strip())
        
        # Handle HTML output
        elif 'text/html' in data:
            html = data['text/html']
            if isinstance(html, list):
                html_content = ''.join(html)
            else:
                html_content = html
            result_parts.append(f"[{output_type} - HTML]")
            # Simplify HTML output, extract text content
            import re
            # Simple HTML tag removal
            text_content = re.sub(r'<[^>]+>', '', html_content)
            text_content = text_content.strip()
            if len(text_content) > 500:
                text_content = text_content[:500] + "..."
            result_parts.append(text_content)
        
        # Handle image output
        elif 'image/png' in data or 'image/jpeg' in data or 'image/jpg' in data:
            image_type = 'image/png' if 'image/png' in data else ('image/jpeg' if 'image/jpeg' in data else 'image/jpg')
            result_parts.append(f"[{output_type} - Image output]")
            result_parts.append(f"Image type: {image_type}")
            result_parts.append("(Image data omitted)")
        
        # Handle other data types
        elif data:
            data_types = ', '.join(data.keys())
            result_parts.append(f"[{output_type} - {data_types}]")
            result_parts.append("(Complex data type, content omitted)")
        
        return "\n".join(result_parts)
    
    @classmethod
    async def get_notebook_info(cls, file_data: bytes, file_path: str) -> Dict[str, Any]:
        """
        Get Notebook basic information
        
        Args:
            file_data: Notebook file data
            file_path: File path
            
        Returns:
            Notebook information dictionary
        """
        try:
            notebook_content = json.loads(file_data.decode('utf-8'))
            
            cells = notebook_content.get('cells', [])
            metadata = notebook_content.get('metadata', {})
            language_info = metadata.get('language_info', {})
            
            # Count cell types
            cell_types = {}
            for cell in cells:
                cell_type = cell.get('cell_type', 'unknown')
                cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
            
            return {
                'file_name': Path(file_path).name,
                'file_type': 'Jupyter Notebook',
                'total_cells': len(cells),
                'cell_types': cell_types,
                'language': language_info.get('name', 'unknown'),
                'language_version': language_info.get('version', 'unknown'),
                'nbformat': notebook_content.get('nbformat'),
                'nbformat_minor': notebook_content.get('nbformat_minor')
            }
            
        except Exception as e:
            return {
                'file_name': Path(file_path).name,
                'error': f"Failed to get Notebook info: {str(e)}"
            }
