import os
import magic
import logging
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger
from pydantic import BaseModel, Field
from werkzeug.security import generate_password_hash, check_password_hash

class SecurityConfig(BaseModel):
    max_file_size: int = Field(default=2 * 1024 * 1024 * 1024)  # 2GB
    allowed_mime_types: List[str] = Field(default=[
        'video/mp4',
        'video/x-msvideo',
        'video/quicktime',
        'video/x-matroska'
    ])
    allowed_extensions: List[str] = Field(default=['.mp4', '.avi', '.mov', '.mkv'])
    temp_dir: str = Field(default='temp')
    upload_dir: str = Field(default='uploads')

class SecurityManager:
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._setup_directories()
        self.mime = magic.Magic(mime=True)
        
    def _setup_directories(self) -> None:
        """Create necessary directories with proper permissions"""
        for dir_path in [self.config.temp_dir, self.config.upload_dir]:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            # Set directory permissions to 755 (rwxr-xr-x)
            path.chmod(0o755)

    def validate_file(self, file_path: str) -> Dict[str, bool]:
        """Validate file size, type and extension"""
        try:
            path = Path(file_path)
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.config.max_file_size:
                return {
                    'valid': False,
                    'reason': f'حجم الملف يتجاوز الحد الأقصى المسموح به ({self.config.max_file_size / (1024*1024):.0f}MB)'
                }
            
            # Check file extension
            if path.suffix.lower() not in self.config.allowed_extensions:
                return {
                    'valid': False,
                    'reason': 'امتداد الملف غير مدعوم'
                }
            
            # Check MIME type
            mime_type = self.mime.from_file(str(path))
            if mime_type not in self.config.allowed_mime_types:
                return {
                    'valid': False,
                    'reason': 'نوع الملف غير مدعوم'
                }
            
            return {'valid': True, 'reason': 'الملف صالح'}
            
        except Exception as e:
            logger.error(f"Error validating file: {str(e)}")
            return {
                'valid': False,
                'reason': f'فشل في التحقق من الملف: {str(e)}'
            }

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks"""
        # Remove potentially dangerous characters
        sanitized = ''.join(c for c in filename if c.isalnum() or c in '._- ')
        return sanitized

    def get_safe_path(self, filename: str, directory: str = None) -> Path:
        """Get safe file path preventing directory traversal"""
        safe_name = self.sanitize_filename(filename)
        base_dir = Path(directory or self.config.upload_dir)
        return base_dir / safe_name

    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """Clean up old temporary files"""
        try:
            current_time = Path().stat().st_mtime
            for directory in [self.config.temp_dir, self.config.upload_dir]:
                path = Path(directory)
                if not path.exists():
                    continue
                    
                for file_path in path.glob('*'):
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_hours * 3600:
                            file_path.unlink()
                            logger.info(f"Deleted old file: {file_path}")
                            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")

class UserAuth:
    def __init__(self):
        self.users = {}
        
    def register_user(self, username: str, password: str) -> bool:
        """Register a new user"""
        if username in self.users:
            return False
        self.users[username] = generate_password_hash(password)
        return True
        
    def verify_user(self, username: str, password: str) -> bool:
        """Verify user credentials"""
        if username not in self.users:
            return False
        return check_password_hash(self.users[username], password)

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        if not self.verify_user(username, old_password):
            return False
        self.users[username] = generate_password_hash(new_password)
        return True
