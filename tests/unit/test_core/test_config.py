"""Tests for configuration management system."""

import pytest
import os
from unittest.mock import patch
from src.core.config import Settings, Environment, AIProvider


class TestSettings:
    """Test Settings class configuration and validation."""
    
    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()
        
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.log_level == "INFO"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.database_url == "sqlite:///./app.db"
        assert settings.default_ai_provider == AIProvider.OPENAI
    
    def test_environment_development_defaults(self):
        """Test development environment sets correct defaults."""
        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.debug == True
        assert settings.reload == True
        assert settings.database_echo == True
    
    def test_get_database_url(self):
        """Test database URL generation."""
        settings = Settings(environment=Environment.TESTING)
        assert settings.get_database_url() == "sqlite:///./test.db"
    
    def test_ai_config_validation(self):
        """Test AI configuration validation."""
        settings = Settings()
        with pytest.raises(ValueError, match="At least one AI provider must be configured"):
            settings.validate_ai_config()
        
        settings = Settings(openai_api_key="test-key")
        assert settings.validate_ai_config() == True
