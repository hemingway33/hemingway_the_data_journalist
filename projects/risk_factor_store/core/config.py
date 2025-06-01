"""
Configuration settings for the Credit Risk Factor Store.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # Application
    app_name: str = "Credit Risk Factor Store"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/risk_factors",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # API Keys
    fred_api_key: Optional[str] = Field(default=None, env="FRED_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    
    # Data Sources
    update_frequency_minutes: int = Field(default=15, env="UPDATE_FREQUENCY_MINUTES")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Risk Models
    default_pd_threshold: float = Field(default=0.05, env="DEFAULT_PD_THRESHOLD")
    portfolio_var_confidence: float = Field(default=0.95, env="PORTFOLIO_VAR_CONFIDENCE")
    stress_test_scenarios: int = Field(default=1000, env="STRESS_TEST_SCENARIOS")
    
    # Monitoring
    enable_prometheus: bool = Field(default=True, env="ENABLE_PROMETHEUS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Feature Engineering
    lookback_days: int = Field(default=252, env="LOOKBACK_DAYS")  # 1 year
    volatility_window: int = Field(default=30, env="VOLATILITY_WINDOW")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings 