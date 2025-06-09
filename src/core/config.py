"""
AI QA Agent - Configuration Management System
Comprehensive configuration with validation and type safety.
"""

from pydantic import BaseSettings, Field, validator
from typing import Optional, List
from enum import Enum
import os


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class AIProvider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class Settings(BaseSettings):
    """
    Application configuration with validation and type safety.
    """
    
    # Environment Configuration
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # API Configuration
    host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API port number"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development"
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of worker processes"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./app.db",
        description="Database connection URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )
    database_pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Database connection pool overflow"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    redis_max_connections: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Redis connection pool size"
    )
    
    # AI Provider Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    default_ai_provider: AIProvider = Field(
        default=AIProvider.OPENAI,
        description="Default AI provider to use"
    )
    
    # AI Generation Configuration
    max_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens for AI generation"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="AI generation temperature"
    )
    ai_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="AI request timeout in seconds"
    )
    
    # File Processing Configuration
    max_upload_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        ge=1024 * 1024,
        le=500 * 1024 * 1024,
        description="Maximum file upload size in bytes"
    )
    supported_languages: List[str] = Field(
        default=["python", "javascript", "typescript", "java", "go", "rust"],
        description="Supported programming languages"
    )
    max_files_per_repo: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum files to analyze per repository"
    )
    analysis_timeout: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Analysis timeout in seconds"
    )
    
    # Security Configuration
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        min_length=32,
        description="Secret key for cryptographic operations"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Access token expiration time"
    )
    allowed_hosts: List[str] = Field(
        default=["*"],
        description="Allowed hosts for CORS"
    )
    
    # Background Task Configuration
    task_timeout: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Background task timeout"
    )
    max_concurrent_tasks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent background tasks"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("environment")
    def set_environment_defaults(cls, v, values):
        """Set environment-specific defaults."""
        if v == Environment.DEVELOPMENT:
            values.update({
                "debug": True,
                "log_level": "DEBUG",
                "reload": True,
                "database_echo": True
            })
        elif v == Environment.TESTING:
            values.update({
                "debug": True,
                "log_level": "DEBUG",
                "database_echo": False
            })
        elif v == Environment.PRODUCTION:
            values.update({
                "debug": False,
                "log_level": "INFO",
                "reload": False,
                "database_echo": False
            })
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v, values):
        """Validate secret key in production."""
        if (values.get("environment") == Environment.PRODUCTION and 
            v == "dev-secret-key-change-in-production"):
            raise ValueError("Secret key must be changed in production")
        return v
    
    def get_database_url(self) -> str:
        """Get database URL with environment-specific handling."""
        if self.environment == Environment.TESTING:
            return "sqlite:///./test.db"
        return self.database_url
    
    def validate_ai_config(self) -> bool:
        """Validate AI provider configuration."""
        if self.default_ai_provider == AIProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OpenAI API key required when using OpenAI provider")
        
        if self.default_ai_provider == AIProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("Anthropic API key required when using Anthropic provider")
        
        if not any([self.openai_api_key, self.anthropic_api_key]):
            raise ValueError("At least one AI provider must be configured")
        
        return True
    
    def get_ai_config(self) -> dict:
        """Get AI provider configuration."""
        return {
            "default_provider": self.default_ai_provider,
            "openai": {
                "api_key": self.openai_api_key,
                "available": bool(self.openai_api_key)
            },
            "anthropic": {
                "api_key": self.anthropic_api_key,
                "available": bool(self.anthropic_api_key)
            },
            "generation_config": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timeout": self.ai_timeout
            }
        }
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration."""
        if self.environment == Environment.PRODUCTION:
            return {
                "allow_origins": self.allowed_hosts,
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["*"],
            }
        else:
            return {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            }
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection function for FastAPI."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
