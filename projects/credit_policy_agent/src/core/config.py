"""Configuration management for the Credit Policy Agent."""

import os
from functools import lru_cache
from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///./credit_policy.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")


class RedisConfig(BaseSettings):
    """Redis configuration."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")


class APIConfig(BaseSettings):
    """API configuration."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    secret_key: str = Field(default="your-secret-key-here", env="API_SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""
    
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")


class ModelConfig(BaseSettings):
    """Model configuration."""
    
    model_registry_path: str = Field(default="./models", env="MODEL_REGISTRY_PATH")
    model_validation_threshold: float = Field(default=0.8, env="MODEL_VALIDATION_THRESHOLD")
    model_drift_threshold: float = Field(default=0.1, env="MODEL_DRIFT_THRESHOLD")
    auto_retrain_enabled: bool = Field(default=True, env="AUTO_RETRAIN_ENABLED")


class PolicyConfig(BaseSettings):
    """Policy configuration."""
    
    policy_version_retention: int = Field(default=10, env="POLICY_VERSION_RETENTION")
    policy_update_interval_hours: int = Field(default=24, env="POLICY_UPDATE_INTERVAL_HOURS")
    ab_test_enabled: bool = Field(default=True, env="AB_TEST_ENABLED")
    ab_test_traffic_split: float = Field(default=0.1, env="AB_TEST_TRAFFIC_SPLIT")
    
    @field_validator("ab_test_traffic_split")
    @classmethod
    def validate_traffic_split(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("AB test traffic split must be between 0 and 1")
        return v


class OptimizationConfig(BaseSettings):
    """Optimization configuration."""
    
    optimization_interval_hours: int = Field(default=6, env="OPTIMIZATION_INTERVAL_HOURS")
    max_optimization_iterations: int = Field(default=100, env="MAX_OPTIMIZATION_ITERATIONS")
    optimization_timeout_minutes: int = Field(default=30, env="OPTIMIZATION_TIMEOUT_MINUTES")
    
    # Optimization objectives weights
    compliance_weight: float = Field(default=0.4, env="COMPLIANCE_WEIGHT")
    risk_weight: float = Field(default=0.4, env="RISK_WEIGHT")
    profit_weight: float = Field(default=0.2, env="PROFIT_WEIGHT")
    
    @field_validator("compliance_weight", "risk_weight", "profit_weight")
    @classmethod
    def validate_weights(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Weight must be between 0 and 1")
        return v


class DataSourceConfig(BaseSettings):
    """Data source configuration."""
    
    data_quality_threshold: float = Field(default=0.95, env="DATA_QUALITY_THRESHOLD")
    data_freshness_hours: int = Field(default=24, env="DATA_FRESHNESS_HOURS")
    alternative_data_sources: List[str] = Field(default=[], env="ALTERNATIVE_DATA_SOURCES")


class Config(BaseSettings):
    """Main configuration class."""
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    app_name: str = Field(default="Credit Policy Agent", env="APP_NAME")
    version: str = Field(default="0.1.0", env="VERSION")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    api: APIConfig = APIConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    model: ModelConfig = ModelConfig()
    policy: PolicyConfig = PolicyConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_config() -> Config:
    """Get cached configuration instance."""
    return Config() 