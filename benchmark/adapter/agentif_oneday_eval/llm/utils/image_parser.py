"""
Image File Parser
Supports parsing and base64 encoding of various image formats
"""
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional


class ImageParser:
    """Image File Parser"""
    
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico'
    }
    
    MIME_TYPES = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg', 
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.ico': 'image/x-icon'
    }
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a supported image format"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def get_mime_type(cls, file_path: str) -> str:
        """Get image MIME type"""
        extension = Path(file_path).suffix.lower()
        return cls.MIME_TYPES.get(extension, 'image/jpeg')
    
    @classmethod
    async def parse_image(cls, file_data: bytes, file_path: str) -> Dict[str, Any]:
        """
        Parse image file
        
        Args:
            file_data: Image file data
            file_path: File path
            
        Returns:
            Dictionary containing image info and base64 encoding
        """
        try:
            # Get image basic info
            image = Image.open(BytesIO(file_data))
            width, height = image.size
            format_name = image.format or "Unknown"
            mode = image.mode
            
            # Convert to base64 encoding
            image_base64 = base64.b64encode(file_data).decode('utf-8')
            
            # Get MIME type
            mime_type = cls.get_mime_type(file_path)
            
            return {
                'type': 'image',
                'file_name': Path(file_path).name,
                'width': width,
                'height': height,
                'format': format_name,
                'mode': mode,
                'mime_type': mime_type,
                'base64_data': image_base64,
                'size_bytes': len(file_data)
            }
            
        except Exception as e:
            raise Exception(f"Failed to parse image file: {str(e)}")
    
    @classmethod
    async def parse_image_for_text(cls, file_data: bytes, file_path: str) -> str:
        """
        Parse image file and return text description (for text-only mode)
        
        Args:
            file_data: Image file data
            file_path: File path
            
        Returns:
            Text description of the image
        """
        try:
            image_info = await cls.parse_image(file_data, file_path)
            
            return (
                f"Image file info:\n"
                f"- File name: {image_info['file_name']}\n"
                f"- Dimensions: {image_info['width']}x{image_info['height']}\n"
                f"- Format: {image_info['format']}\n"
                f"- Color mode: {image_info['mode']}\n"
                f"- File size: {image_info['size_bytes']} bytes\n"
                f"- MIME type: {image_info['mime_type']}"
            )
            
        except Exception as e:
            return f"Failed to parse image file: {str(e)}"
    
    @classmethod
    async def parse_image_for_multimodal(cls, file_data: bytes, file_path: str) -> Dict[str, Any]:
        """
        Parse image file for multimodal models
        
        Args:
            file_data: Image file data
            file_path: File path
            
        Returns:
            Image data format suitable for multimodal models
        """
        try:
            image_info = await cls.parse_image(file_data, file_path)
            
            return {
                'inline_data': {
                    'mime_type': image_info['mime_type'],
                    'data': image_info['base64_data']
                }
            }
            
        except Exception as e:
            raise Exception(f"Failed to parse image for multimodal model: {str(e)}")
    
    @classmethod
    def optimize_image_for_api(cls, file_data: bytes, max_size: int = 1024*1024) -> bytes:
        """
        Optimize image size for API limits
        
        Args:
            file_data: Original image data
            max_size: Maximum file size (bytes)
            
        Returns:
            Optimized image data
        """
        try:
            if len(file_data) <= max_size:
                return file_data
            
            # If file is too large, compress it
            image = Image.open(BytesIO(file_data))
            
            # Calculate compression ratio
            quality = 85
            while quality > 10:
                output = BytesIO()
                if image.format == 'PNG':
                    # PNG uses optimize parameter
                    image.save(output, format='PNG', optimize=True)
                else:
                    # JPEG uses quality parameter
                    image.save(output, format='JPEG', quality=quality, optimize=True)
                
                compressed_data = output.getvalue()
                if len(compressed_data) <= max_size:
                    return compressed_data
                
                quality -= 10
            
            # If still too large, try resizing
            width, height = image.size
            scale = 0.8
            
            while scale > 0.1:
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                output = BytesIO()
                if image.format == 'PNG':
                    resized_image.save(output, format='PNG', optimize=True)
                else:
                    resized_image.save(output, format='JPEG', quality=75, optimize=True)
                
                compressed_data = output.getvalue()
                if len(compressed_data) <= max_size:
                    return compressed_data
                
                scale -= 0.1
            
            # If still too large, return original data
            return file_data
            
        except Exception:
            # If optimization fails, return original data
            return file_data
