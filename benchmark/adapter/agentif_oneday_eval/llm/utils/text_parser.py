"""
Text and Code File Parser
Supports various text formats and programming language files
"""
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class TextParser:
    """Text and Code File Parser"""
    
    # Programming language file extension mapping
    PROGRAMMING_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.R': 'r',
        '.m': 'matlab',
        '.pl': 'perl',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.fish': 'fish',
        '.ps1': 'powershell',
        '.bat': 'batch',
        '.cmd': 'batch'
    }
    
    # Markup languages and config files
    MARKUP_LANGUAGES = {
        '.html': 'html',
        '.htm': 'html',
        '.xhtml': 'xhtml',
        '.xml': 'xml',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.rst': 'restructuredtext',
        '.tex': 'latex',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'config',
        '.config': 'config'
    }
    
    # Data files
    DATA_FILES = {
        '.csv': 'csv',
        '.tsv': 'tsv',
        '.sql': 'sql',
        '.log': 'log',
        '.txt': 'text'
    }
    
    # All supported formats
    SUPPORTED_FORMATS = {**PROGRAMMING_LANGUAGES, **MARKUP_LANGUAGES, **DATA_FILES}
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a supported text format"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def get_file_type(cls, file_path: str) -> str:
        """Get file type"""
        extension = Path(file_path).suffix.lower()
        return cls.SUPPORTED_FORMATS.get(extension, 'text')
    
    @classmethod
    def get_supported_formats(cls) -> Dict[str, str]:
        """Get supported file formats"""
        return cls.SUPPORTED_FORMATS.copy()
    
    @classmethod
    async def parse_text_file(cls, file_data: bytes, file_path: str, include_metadata: bool = True) -> str:
        """
        Parse text file
        
        Args:
            file_data: File data
            file_path: File path
            include_metadata: Whether to include file metadata
            
        Returns:
            Parsed text content
        """
        try:
            # Try different encodings to decode file content
            content = cls._decode_text_content(file_data)
            if content is None:
                return "File encoding not supported, unable to parse"
            
            file_type = cls.get_file_type(file_path)
            file_name = Path(file_path).name
            
            # Build result
            result_parts = []
            
            if include_metadata:
                # Add file info
                metadata = cls._get_file_metadata(content, file_path)
                result_parts.append(f"File info: {file_name}")
                result_parts.append(f"File type: {file_type}")
                result_parts.append(f"File size: {len(file_data)} bytes")
                result_parts.append(f"Lines: {metadata['line_count']}")
                
                if metadata.get('encoding'):
                    result_parts.append(f"Encoding: {metadata['encoding']}")
                
                result_parts.append("")
            
            # Special handling based on file type
            if file_type in cls.PROGRAMMING_LANGUAGES.values():
                formatted_content = cls._format_code_content(content, file_type)
                result_parts.append(f"Code content:\n```{file_type}\n{formatted_content}\n```")
            elif file_type == 'markdown':
                result_parts.append(f"Markdown content:\n{content}")
            elif file_type == 'json':
                formatted_content = cls._format_json_content(content)
                result_parts.append(f"JSON content:\n```json\n{formatted_content}\n```")
            elif file_type in ['yaml', 'yml']:
                result_parts.append(f"YAML content:\n```yaml\n{content}\n```")
            elif file_type == 'csv':
                formatted_content = cls._format_csv_content(content)
                result_parts.append(f"CSV content:\n{formatted_content}")
            elif file_type == 'sql':
                result_parts.append(f"SQL content:\n```sql\n{content}\n```")
            elif file_type == 'log':
                formatted_content = cls._format_log_content(content)
                result_parts.append(f"Log content:\n{formatted_content}")
            else:
                result_parts.append(f"Text content:\n{content}")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            return f"Failed to parse text file: {str(e)}"
    
    @classmethod
    def _decode_text_content(cls, file_data: bytes) -> Optional[str]:
        """Try to decode text content using different encodings"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                content = file_data.decode(encoding)
                # Check if contains obvious garbled characters
                if cls._is_valid_text(content):
                    return content
            except UnicodeDecodeError:
                continue
        
        return None
    
    @classmethod
    def _is_valid_text(cls, content: str) -> bool:
        """Check if text is valid (no obvious garbled characters)"""
        # Check control character ratio
        control_chars = sum(1 for c in content if ord(c) < 32 and c not in '\t\n\r')
        if len(content) > 0 and control_chars / len(content) > 0.1:
            return False
        
        # Check for too many replacement characters
        replacement_chars = content.count('\ufffd')
        if len(content) > 0 and replacement_chars / len(content) > 0.05:
            return False
        
        return True
    
    @classmethod
    def _get_file_metadata(cls, content: str, file_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        lines = content.split('\n')
        
        metadata = {
            'line_count': len(lines),
            'char_count': len(content),
            'word_count': len(content.split()),
            'file_name': Path(file_path).name,
            'file_type': cls.get_file_type(file_path)
        }
        
        # Detect possible encoding info
        first_lines = '\n'.join(lines[:5]).lower()
        if 'coding:' in first_lines or 'encoding:' in first_lines:
            encoding_match = re.search(r'coding[:=]\s*([-\w.]+)', first_lines)
            if encoding_match:
                metadata['declared_encoding'] = encoding_match.group(1)
        
        return metadata
    
    @classmethod
    def _format_code_content(cls, content: str, language: str) -> str:
        """Format code content"""
        lines = content.split('\n')
        
        # If code is too long, only show beginning and add summary
        max_lines = 200
        if len(lines) > max_lines:
            # Show first part of code
            displayed_lines = lines[:max_lines]
            
            # Add code summary
            summary = cls._analyze_code_structure(content, language)
            
            result = '\n'.join(displayed_lines)
            result += f"\n\n... ({len(lines) - max_lines} more lines not shown)\n"
            result += f"\nCode structure summary:\n{summary}"
            
            return result
        
        return content
    
    @classmethod
    def _analyze_code_structure(cls, content: str, language: str) -> str:
        """Analyze code structure"""
        lines = content.split('\n')
        summary = []
        
        if language == 'python':
            # Analyze Python code structure
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            imports = re.findall(r'^(?:from\s+\S+\s+)?import\s+(.+)', content, re.MULTILINE)
            
            if classes:
                summary.append(f"Classes: {', '.join(classes[:5])}" + ("..." if len(classes) > 5 else ""))
            if functions:
                summary.append(f"Functions: {', '.join(functions[:10])}" + ("..." if len(functions) > 10 else ""))
            if imports:
                summary.append(f"Imports: {len(imports)} modules")
        
        elif language in ['javascript', 'typescript']:
            # Analyze JavaScript/TypeScript code structure
            functions = re.findall(r'function\s+(\w+)', content)
            classes = re.findall(r'class\s+(\w+)', content)
            exports = re.findall(r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)', content)
            
            if classes:
                summary.append(f"Classes: {', '.join(classes[:5])}")
            if functions:
                summary.append(f"Functions: {', '.join(functions[:10])}")
            if exports:
                summary.append(f"Exports: {', '.join(exports[:5])}")
        
        elif language == 'java':
            # Analyze Java code structure
            classes = re.findall(r'(?:public\s+)?class\s+(\w+)', content)
            methods = re.findall(r'(?:public|private|protected)?\s*(?:static\s+)?\w+\s+(\w+)\s*\(', content)
            
            if classes:
                summary.append(f"Classes: {', '.join(classes)}")
            if methods:
                summary.append(f"Methods: {len(methods)} total")
        
        # General statistics
        summary.append(f"Total lines: {len(lines)}")
        non_empty_lines = [line for line in lines if line.strip()]
        summary.append(f"Non-empty lines: {len(non_empty_lines)}")
        
        return '\n'.join(summary) if summary else "Unable to analyze code structure"
    
    @classmethod
    def _format_json_content(cls, content: str) -> str:
        """Format JSON content"""
        try:
            import json
            # Try to parse and format JSON
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            # If parsing fails, return original content
            return content
    
    @classmethod
    def _format_csv_content(cls, content: str) -> str:
        """Format CSV content"""
        lines = content.split('\n')
        
        # Limit displayed lines
        max_lines = 50
        if len(lines) > max_lines:
            displayed_lines = lines[:max_lines]
            result = '\n'.join(displayed_lines)
            result += f"\n... ({len(lines) - max_lines} more lines not shown)"
            return result
        
        return content
    
    @classmethod
    def _format_log_content(cls, content: str) -> str:
        """Format log content"""
        lines = content.split('\n')
        
        # Limit displayed lines
        max_lines = 100
        if len(lines) > max_lines:
            # Show first 50 and last 50 lines
            displayed_lines = lines[:50] + ['... (middle section omitted) ...'] + lines[-50:]
            return '\n'.join(displayed_lines)
        
        return content
    
    @classmethod
    async def extract_code_functions(cls, file_data: bytes, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract function/method definitions from code file
        
        Args:
            file_data: File data
            file_path: File path
            
        Returns:
            List of function information
        """
        try:
            content = cls._decode_text_content(file_data)
            if content is None:
                return []
            
            file_type = cls.get_file_type(file_path)
            functions = []
            
            if file_type == 'python':
                # Extract Python functions and classes
                for match in re.finditer(r'^(class|def)\s+(\w+)\s*\((.*?)\):', content, re.MULTILINE):
                    func_type, name, params = match.groups()
                    line_num = content[:match.start()].count('\n') + 1
                    functions.append({
                        'type': func_type,
                        'name': name,
                        'parameters': params,
                        'line_number': line_num
                    })
            
            elif file_type in ['javascript', 'typescript']:
                # Extract JavaScript/TypeScript functions
                for match in re.finditer(r'function\s+(\w+)\s*\((.*?)\)', content):
                    name, params = match.groups()
                    line_num = content[:match.start()].count('\n') + 1
                    functions.append({
                        'type': 'function',
                        'name': name,
                        'parameters': params,
                        'line_number': line_num
                    })
            
            return functions
            
        except Exception:
            return []
    
    @classmethod
    async def get_text_summary(cls, file_data: bytes, file_path: str) -> str:
        """
        Get text file summary
        
        Args:
            file_data: File data
            file_path: File path
            
        Returns:
            File summary
        """
        try:
            content = cls._decode_text_content(file_data)
            if content is None:
                return "Unable to read file content"
            
            lines = content.split('\n')
            file_type = cls.get_file_type(file_path)
            
            summary_parts = []
            summary_parts.append(f"File type: {file_type}")
            summary_parts.append(f"Total lines: {len(lines)}")
            summary_parts.append(f"Characters: {len(content)}")
            
            # Non-empty line statistics
            non_empty_lines = [line for line in lines if line.strip()]
            summary_parts.append(f"Non-empty lines: {len(non_empty_lines)}")
            
            # Add specific info based on file type
            if file_type in cls.PROGRAMMING_LANGUAGES.values():
                structure_summary = cls._analyze_code_structure(content, file_type)
                summary_parts.append(f"Code structure: {structure_summary}")
            
            # Add content preview (first few lines)
            preview_lines = lines[:5]
            if preview_lines:
                summary_parts.append("Content preview:")
                for i, line in enumerate(preview_lines, 1):
                    if line.strip():
                        summary_parts.append(f"  {i}: {line[:100]}{'...' if len(line) > 100 else ''}")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            return f"Failed to generate file summary: {str(e)}"
