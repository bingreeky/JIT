"""
Subtitle File Parser
Supports parsing ASS and SRT format subtitle files
"""
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from io import BytesIO


class SubtitleParser:
    """Subtitle File Parser"""
    
    SUPPORTED_FORMATS = {'.ass', '.srt'}
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a supported subtitle format"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    async def parse_subtitle(cls, file_data: bytes, file_path: str) -> str:
        """
        Parse subtitle file
        
        Args:
            file_data: File data
            file_path: File path
            
        Returns:
            Parsed text content
        """
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name
        
        # Try to decode file content
        content = cls._decode_text_content(file_data)
        if content is None:
            return f"Failed to decode subtitle file: {file_name}"
        
        try:
            if file_ext == '.ass':
                return await cls._parse_ass(content, file_name)
            elif file_ext == '.srt':
                return await cls._parse_srt(content, file_name)
            else:
                return f"Unsupported subtitle format: {file_ext}"
        except Exception as e:
            return f"Failed to parse subtitle file: {str(e)}"
    
    @classmethod
    def _decode_text_content(cls, file_data: bytes) -> Optional[str]:
        """Try to decode text content using different encodings"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'big5', 'latin-1', 'iso-8859-1', 'cp1252']
        
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
        # Check control character ratio (excluding common ones)
        control_chars = sum(1 for c in content if ord(c) < 32 and c not in '\t\n\r')
        if len(content) > 0 and control_chars / len(content) > 0.1:
            return False
        
        # Check for too many replacement characters
        replacement_chars = content.count('\ufffd')
        if len(content) > 0 and replacement_chars / len(content) > 0.05:
            return False
        
        return True
    
    @classmethod
    async def _parse_ass(cls, content: str, file_name: str) -> str:
        """
        Parse ASS subtitle file using ass library
        
        Args:
            content: File content as string
            file_path: File path
            
        Returns:
            Parsed text content
        """
        try:
            import ass
        except ImportError:
            return f"ASS parsing requires 'ass' library. Please install it: pip install ass"
        
        result_parts = []
        result_parts.append(f"Subtitle file: {file_name} (ASS format)")
        result_parts.append("=" * 60)
        result_parts.append("")
        
        try:
            # Parse ASS file from string
            ass_file = ass.parse(BytesIO(content.encode('utf-8')))
            
            # Extract script info
            if ass_file.script_info:
                result_parts.append("Script Information:")
                for key, value in ass_file.script_info.items():
                    if value:
                        result_parts.append(f"  {key}: {value}")
                result_parts.append("")
            
            # Extract styles
            if ass_file.styles:
                result_parts.append(f"Styles ({len(ass_file.styles)}):")
                for style in ass_file.styles[:10]:  # Show first 10 styles
                    style_info = f"  - {style.name}"
                    if hasattr(style, 'fontname') and style.fontname:
                        style_info += f" (Font: {style.fontname})"
                    result_parts.append(style_info)
                if len(ass_file.styles) > 10:
                    result_parts.append(f"  ... and {len(ass_file.styles) - 10} more styles")
                result_parts.append("")
            
            # Extract dialogue/events
            if ass_file.events:
                result_parts.append(f"Subtitle Events ({len(ass_file.events)}):")
                result_parts.append("-" * 60)
                
                for event in ass_file.events:
                    if event.type == 'Dialogue':
                        # Format time
                        start_time = cls._format_ass_time(event.start)
                        end_time = cls._format_ass_time(event.end)
                        
                        # Extract text (remove ASS formatting tags)
                        text = cls._clean_ass_text(event.text)
                        
                        if text.strip():
                            result_parts.append(f"[{start_time} --> {end_time}] {text}")
                
                result_parts.append("")
            
            result_parts.append("=" * 60)
            result_parts.append(f"Total events: {len(ass_file.events) if ass_file.events else 0}")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            return f"Failed to parse ASS file: {str(e)}"
    
    @classmethod
    def _format_ass_time(cls, time_seconds: float) -> str:
        """Format ASS time (seconds) to HH:MM:SS.mmm format"""
        hours = int(time_seconds // 3600)
        minutes = int((time_seconds % 3600) // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int((time_seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    @classmethod
    def _clean_ass_text(cls, text: str) -> str:
        """Remove ASS formatting tags from text"""
        # Remove ASS override tags like {\an8}, {\b1}, etc.
        text = re.sub(r'\{[^}]*\}', '', text)
        # Remove line breaks that are part of formatting
        text = text.replace('\\N', '\n').replace('\\n', '\n')
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    @classmethod
    async def _parse_srt(cls, content: str, file_name: str) -> str:
        """
        Parse SRT subtitle file using structured parsing
        
        Args:
            content: File content as string
            file_name: File name
            
        Returns:
            Parsed text content
        """
        result_parts = []
        result_parts.append(f"Subtitle file: {file_name} (SRT format)")
        result_parts.append("=" * 60)
        result_parts.append("")
        
        try:
            # Parse SRT content
            subtitles = cls._parse_srt_content(content)
            
            if not subtitles:
                return f"No subtitle entries found in SRT file: {file_name}"
            
            result_parts.append(f"Subtitle Entries ({len(subtitles)}):")
            result_parts.append("-" * 60)
            
            for subtitle in subtitles:
                start_time = subtitle['start']
                end_time = subtitle['end']
                text = subtitle['text']
                
                result_parts.append(f"[{start_time} --> {end_time}] {text}")
            
            result_parts.append("")
            result_parts.append("=" * 60)
            result_parts.append(f"Total entries: {len(subtitles)}")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            return f"Failed to parse SRT file: {str(e)}"
    
    @classmethod
    def _parse_srt_content(cls, content: str) -> List[Dict[str, str]]:
        """
        Parse SRT content into structured format
        
        Args:
            content: SRT file content
            
        Returns:
            List of subtitle dictionaries with 'index', 'start', 'end', 'text' keys
        """
        subtitles = []
        
        # Split by double newlines (SRT entries are separated by blank lines)
        # Handle different line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        entries = re.split(r'\n\s*\n', content.strip())
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = entry.strip().split('\n')
            if len(lines) < 2:
                continue
            
            # First line is index
            index = lines[0].strip()
            
            # Second line is timestamp
            timestamp_line = lines[1].strip()
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', timestamp_line)
            
            if not time_match:
                continue
            
            start_time = time_match.group(1).replace(',', '.')
            end_time = time_match.group(2).replace(',', '.')
            
            # Rest of the lines are subtitle text
            text_lines = lines[2:]
            text = '\n'.join(text_lines).strip()
            
            if text:
                subtitles.append({
                    'index': index,
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
        
        return subtitles
