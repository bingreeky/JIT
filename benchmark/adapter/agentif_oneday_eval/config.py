"""
Application Configuration Module - Simplified version, only contains settings needed for scoring script
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application Settings"""
    model_config = ConfigDict(
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
        protected_namespaces=()  # Allow fields starting with model_
    )
    
    # Basic application configuration
    app_name: str = "LLM Scoring Script"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Logging configuration
    log_level: str = "INFO"
    # Note: Log file path is fixed to logs/llm_score.log in project root in logging_config.py
    
    # LLM configuration
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    chatgpt_api_key: str = ""
    chatgpt_base_url: str = "https://api.openai.com/v1"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_default_model: str = "qwen/qwen3.5-122b-a10b"
    llm_timeout: int = 180  # Timeout in seconds
    llm_max_retries: int = 3
    llm_retry_min_wait: int = 1  # LLM retry minimum wait time in seconds
    llm_retry_max_wait: int = 30  # LLM retry maximum wait time in seconds
    llm_max_concurrent: int = 5  # Maximum concurrent requests
    llm_max_tokens: int = 600000  # Maximum tokens for LLM requests
    
    # Google Search Grounding configuration (only for Gemini models)
    enable_google_search_grounding: bool = False
    
    # Web search configuration
    enable_web_search: bool = True
    web_search_max_results: int = 3
    web_fetch_timeout: int = 30
    
    # Local attachment path configuration
    attachment_path: str = "Data/Attachment"
    
    # PPTX rendering configuration
    pptx_render_mode: str = "text"  # text | image | auto
    pptx_render_dpi: int = 150
    pptx_render_max_slides: int = 50
    
    # HTML content length limit
    max_html_content_length: int = 100000
    
    # Web Search Fallback configuration
    web_search_fallback_enabled: bool = True
    web_search_fallback_model: str = "qwen/qwen3.5-122b-a10b"
    web_search_fallback_max_length: int = 10000


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get application settings"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
