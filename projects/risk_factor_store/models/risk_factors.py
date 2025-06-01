"""
SQLAlchemy models for risk factors and related entities.
"""

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Boolean, 
    JSON, Index, ForeignKey, Enum, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime
from typing import Dict, Any, Optional

from ..core.database import Base


class RiskFactorCategory(enum.Enum):
    """Risk factor categories."""
    ECONOMIC = "economic"
    MARKET = "market"
    CREDIT = "credit"
    INDUSTRY = "industry"
    MACROECONOMIC = "macroeconomic"
    ALTERNATIVE = "alternative"
    REGULATORY = "regulatory"


class DataSource(enum.Enum):
    """Data source types."""
    FRED = "fred"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    NEWS_API = "news_api"
    WEB_SCRAPING = "web_scraping"
    CREDIT_BUREAU = "credit_bureau"
    INTERNAL = "internal"


class RiskFactor(Base):
    """Risk factor master table."""
    
    __tablename__ = "risk_factors"
    
    id = Column(Integer, primary_key=True, index=True)
    factor_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(Enum(RiskFactorCategory), nullable=False, index=True)
    data_source = Column(Enum(DataSource), nullable=False)
    source_identifier = Column(String(200))  # External system identifier
    unit = Column(String(50))
    frequency = Column(String(20))  # daily, weekly, monthly, quarterly
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    factor_values = relationship("RiskFactorValue", back_populates="risk_factor")
    
    # Indexes
    __table_args__ = (
        Index('idx_risk_factor_category_active', 'category', 'is_active'),
        Index('idx_risk_factor_source', 'data_source', 'source_identifier'),
    )


class RiskFactorValue(Base):
    """Time series values for risk factors."""
    
    __tablename__ = "risk_factor_values"
    
    id = Column(Integer, primary_key=True, index=True)
    risk_factor_id = Column(Integer, ForeignKey("risk_factors.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    raw_value = Column(Float)  # Original value before processing
    quality_score = Column(Float, default=1.0)  # Data quality score
    is_interpolated = Column(Boolean, default=False)
    value_metadata = Column(JSON)  # Additional metadata
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    risk_factor = relationship("RiskFactor", back_populates="factor_values")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('risk_factor_id', 'date', name='uq_factor_date'),
        Index('idx_factor_value_date', 'risk_factor_id', 'date'),
    )


class CreditEntity(Base):
    """Credit entities (borrowers, companies, etc.)."""
    
    __tablename__ = "credit_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String(100), unique=True, nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)  # individual, corporate, sovereign
    name = Column(String(200))
    industry_code = Column(String(20), index=True)
    country = Column(String(3), index=True)  # ISO country code
    credit_rating = Column(String(10), index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    risk_scores = relationship("CreditRiskScore", back_populates="entity")
    portfolio_positions = relationship("PortfolioPosition", back_populates="entity")


class CreditRiskScore(Base):
    """Credit risk scores and assessments."""
    
    __tablename__ = "credit_risk_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("credit_entities.id"), nullable=False)
    score_date = Column(DateTime, nullable=False, index=True)
    
    # Probability scores
    probability_of_default = Column(Float, nullable=False)
    loss_given_default = Column(Float)
    exposure_at_default = Column(Float)
    expected_loss = Column(Float)
    
    # Model information
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50))  # logistic, xgboost, neural_network
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    
    # Risk factors used
    risk_factors_snapshot = Column(JSON)  # Risk factor values used in calculation
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    entity = relationship("CreditEntity", back_populates="risk_scores")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('entity_id', 'score_date', 'model_version', 
                        name='uq_entity_score_date_model'),
        Index('idx_risk_score_date_pd', 'score_date', 'probability_of_default'),
    )


class Portfolio(Base):
    """Investment portfolios."""
    
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    portfolio_type = Column(String(50))  # loan, bond, mixed
    currency = Column(String(3), default="USD")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    positions = relationship("PortfolioPosition", back_populates="portfolio")
    risk_metrics = relationship("PortfolioRiskMetric", back_populates="portfolio")


class PortfolioPosition(Base):
    """Portfolio positions."""
    
    __tablename__ = "portfolio_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    entity_id = Column(Integer, ForeignKey("credit_entities.id"), nullable=False)
    position_date = Column(DateTime, nullable=False)
    
    # Position details
    notional_amount = Column(Float, nullable=False)
    market_value = Column(Float)
    accrued_interest = Column(Float, default=0.0)
    weight = Column(Float)  # Portfolio weight
    
    # Risk metrics
    duration = Column(Float)
    modified_duration = Column(Float)
    convexity = Column(Float)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    entity = relationship("CreditEntity", back_populates="portfolio_positions")
    
    # Constraints
    __table_args__ = (
        Index('idx_portfolio_position_date', 'portfolio_id', 'position_date'),
    )


class PortfolioRiskMetric(Base):
    """Portfolio-level risk metrics."""
    
    __tablename__ = "portfolio_risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    metric_date = Column(DateTime, nullable=False)
    
    # VaR metrics
    var_1d = Column(Float)  # 1-day VaR
    var_10d = Column(Float)  # 10-day VaR
    expected_shortfall = Column(Float)
    
    # Portfolio statistics
    total_exposure = Column(Float)
    average_pd = Column(Float)
    average_lgd = Column(Float)
    expected_loss = Column(Float)
    
    # Concentration metrics
    concentration_ratio = Column(Float)  # Top 10 positions concentration
    herfindahl_index = Column(Float)
    
    # Correlation metrics
    average_correlation = Column(Float)
    eigen_risk = Column(Float)  # Principal component risk
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="risk_metrics")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'metric_date', name='uq_portfolio_metric_date'),
        Index('idx_portfolio_risk_date', 'portfolio_id', 'metric_date'),
    )


class DataQualityCheck(Base):
    """Data quality monitoring."""
    
    __tablename__ = "data_quality_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    check_date = Column(DateTime, nullable=False, index=True)
    risk_factor_id = Column(Integer, ForeignKey("risk_factors.id"), nullable=False)
    
    # Quality metrics
    completeness_score = Column(Float)  # % of non-null values
    validity_score = Column(Float)  # % of values passing validation
    consistency_score = Column(Float)  # Consistency with historical patterns
    timeliness_score = Column(Float)  # Data freshness score
    overall_score = Column(Float)  # Weighted overall quality score
    
    # Check details
    records_checked = Column(Integer)
    missing_values = Column(Integer)
    outliers_detected = Column(Integer)
    validation_errors = Column(JSON)  # Detailed error information
    
    created_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        Index('idx_quality_check_date_factor', 'check_date', 'risk_factor_id'),
    ) 