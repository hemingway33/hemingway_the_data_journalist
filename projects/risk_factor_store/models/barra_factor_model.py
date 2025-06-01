"""
Barra-style factor model database models.
"""

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Boolean, 
    JSON, Index, ForeignKey, Enum, UniqueConstraint, Table
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime
from typing import Dict, Any, Optional

from ..core.database import Base


class FactorType(enum.Enum):
    """Types of risk factors in the Barra model."""
    COUNTRY = "country"
    INDUSTRY = "industry"
    STYLE = "style"
    MACROECONOMIC = "macroeconomic"
    CURRENCY = "currency"
    STATISTICAL = "statistical"


class ExposureType(enum.Enum):
    """Types of factor exposures."""
    FUNDAMENTAL = "fundamental"
    STATISTICAL = "statistical"
    HYBRID = "hybrid"


class VolatilityRegime(enum.Enum):
    """Volatility regime classifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRISIS = "crisis"


# Association table for factor hierarchies
factor_hierarchy = Table(
    'factor_hierarchy',
    Base.metadata,
    Column('parent_factor_id', Integer, ForeignKey('barra_risk_factors.id'), primary_key=True),
    Column('child_factor_id', Integer, ForeignKey('barra_risk_factors.id'), primary_key=True),
    Column('hierarchy_weight', Float, default=1.0)
)


class RiskFactor(Base):
    """Systematic risk factors in the Barra model."""
    
    __tablename__ = "barra_risk_factors"
    
    id = Column(Integer, primary_key=True, index=True)
    factor_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    factor_type = Column(Enum(FactorType), nullable=False, index=True)
    
    # Factor hierarchy information
    level = Column(Integer, default=0)  # 0=Global, 1=Regional, 2=Country, etc.
    region = Column(String(50), index=True)
    country = Column(String(3), index=True)  # ISO country code
    
    # Factor characteristics
    is_systematic = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True, index=True)
    estimation_universe = Column(String(100))  # Which universe this factor applies to
    
    # Model parameters
    volatility_target = Column(Float)  # Target volatility for normalization
    decay_factor = Column(Float, default=0.95)  # Exponential decay for estimation
    
    # Metadata
    data_source = Column(String(100))
    frequency = Column(String(20))  # daily, weekly, monthly
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    factor_returns = relationship("FactorReturn", back_populates="factor")
    exposures = relationship("FactorExposure", back_populates="factor")
    
    # Factor hierarchy relationships
    children = relationship(
        "RiskFactor",
        secondary=factor_hierarchy,
        primaryjoin=id == factor_hierarchy.c.parent_factor_id,
        secondaryjoin=id == factor_hierarchy.c.child_factor_id,
        backref="parents"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_factor_type_level', 'factor_type', 'level'),
        Index('idx_factor_country_region', 'country', 'region'),
    )


class FactorReturn(Base):
    """Time series of factor returns."""
    
    __tablename__ = "barra_factor_returns"
    
    id = Column(Integer, primary_key=True, index=True)
    factor_id = Column(Integer, ForeignKey("barra_risk_factors.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Returns and volatility
    factor_return = Column(Float, nullable=False)
    cumulative_return = Column(Float)
    volatility = Column(Float)  # Annualized volatility
    volatility_regime = Column(Enum(VolatilityRegime))
    
    # Statistical measures
    skewness = Column(Float)
    kurtosis = Column(Float)
    var_99 = Column(Float)  # 99% VaR
    
    # Model diagnostics
    t_statistic = Column(Float)
    r_squared = Column(Float)
    information_ratio = Column(Float)
    
    # Estimation parameters
    estimation_window = Column(Integer)  # Number of observations used
    half_life = Column(Float)  # Half-life for exponential weighting
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    factor = relationship("RiskFactor", back_populates="factor_returns")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('factor_id', 'date', name='uq_factor_return_date'),
        Index('idx_factor_return_date_vol', 'date', 'volatility'),
    )


class Entity(Base):
    """Entities in the risk model universe."""
    
    __tablename__ = "barra_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200))
    
    # Entity classification
    entity_type = Column(String(50), nullable=False)  # equity, bond, loan, etc.
    sector = Column(String(100), index=True)
    industry = Column(String(100), index=True)
    sub_industry = Column(String(100), index=True)
    
    # Geographic information
    country = Column(String(3), index=True)  # ISO country code
    region = Column(String(50), index=True)
    
    # Entity characteristics
    market_cap = Column(Float)
    currency = Column(String(3))
    listing_exchange = Column(String(50))
    
    # Model inclusion
    is_in_universe = Column(Boolean, default=True)
    universe_start_date = Column(DateTime)
    universe_end_date = Column(DateTime)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    exposures = relationship("FactorExposure", back_populates="entity")
    returns = relationship("EntityReturn", back_populates="entity")
    risk_metrics = relationship("EntityRiskMetric", back_populates="entity")
    
    # Indexes
    __table_args__ = (
        Index('idx_entity_sector_country', 'sector', 'country'),
        Index('idx_entity_universe', 'is_in_universe', 'universe_start_date'),
    )


class FactorExposure(Base):
    """Factor exposures for entities."""
    
    __tablename__ = "barra_factor_exposures"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("barra_entities.id"), nullable=False)
    factor_id = Column(Integer, ForeignKey("barra_risk_factors.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Exposure values
    exposure = Column(Float, nullable=False)
    standardized_exposure = Column(Float)  # Cross-sectionally standardized
    raw_exposure = Column(Float)  # Before any adjustments
    
    # Exposure characteristics
    exposure_type = Column(Enum(ExposureType), nullable=False)
    confidence = Column(Float, default=1.0)  # Confidence in exposure estimate
    
    # Statistical measures
    t_statistic = Column(Float)
    r_squared = Column(Float)
    estimation_error = Column(Float)
    
    # Estimation details
    estimation_method = Column(String(50))
    estimation_window = Column(Integer)
    data_points_used = Column(Integer)
    
    # Quality indicators
    is_interpolated = Column(Boolean, default=False)
    quality_score = Column(Float, default=1.0)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    entity = relationship("Entity", back_populates="exposures")
    factor = relationship("RiskFactor", back_populates="exposures")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('entity_id', 'factor_id', 'date', name='uq_entity_factor_exposure'),
        Index('idx_exposure_date_factor', 'date', 'factor_id'),
        Index('idx_exposure_standardized', 'date', 'standardized_exposure'),
    )


class EntityReturn(Base):
    """Entity returns for risk model estimation."""
    
    __tablename__ = "barra_entity_returns"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("barra_entities.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Returns
    total_return = Column(Float, nullable=False)
    excess_return = Column(Float)  # Over risk-free rate
    specific_return = Column(Float)  # Idiosyncratic return
    
    # Systematic vs idiosyncratic decomposition
    systematic_return = Column(Float)
    factor_explained_return = Column(Float)
    
    # Statistical measures
    volatility = Column(Float)
    beta = Column(Float)  # Market beta
    
    # Quality indicators
    return_type = Column(String(20))  # price, total, adjusted
    currency = Column(String(3))
    is_adjusted = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    entity = relationship("Entity", back_populates="returns")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('entity_id', 'date', name='uq_entity_return_date'),
        Index('idx_entity_return_date', 'date', 'total_return'),
    )


class EntityRiskMetric(Base):
    """Risk metrics for individual entities."""
    
    __tablename__ = "barra_entity_risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("barra_entities.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Total risk decomposition
    total_risk = Column(Float, nullable=False)  # Annualized volatility
    systematic_risk = Column(Float)
    idiosyncratic_risk = Column(Float)
    
    # Factor risk contributions
    country_risk = Column(Float)
    industry_risk = Column(Float)
    style_risk = Column(Float)
    macro_risk = Column(Float)
    
    # Risk ratios
    systematic_risk_ratio = Column(Float)  # Systematic / Total
    diversification_ratio = Column(Float)  # 1 - (Idiosyncratic / Total)
    
    # Statistical measures
    beta = Column(Float)
    tracking_error = Column(Float)
    information_ratio = Column(Float)
    
    # Model diagnostics
    r_squared = Column(Float)  # Factor model R²
    bias_statistic = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    entity = relationship("Entity", back_populates="risk_metrics")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('entity_id', 'date', name='uq_entity_risk_date'),
        Index('idx_entity_risk_total', 'date', 'total_risk'),
    )


class FactorCovariance(Base):
    """Factor covariance matrix estimates."""
    
    __tablename__ = "barra_factor_covariances"
    
    id = Column(Integer, primary_key=True, index=True)
    factor1_id = Column(Integer, ForeignKey("barra_risk_factors.id"), nullable=False)
    factor2_id = Column(Integer, ForeignKey("barra_risk_factors.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Covariance estimates
    covariance = Column(Float, nullable=False)
    correlation = Column(Float)
    
    # Estimation method
    estimation_method = Column(String(50))  # sample, shrinkage, factor_model
    shrinkage_intensity = Column(Float)  # For shrinkage estimators
    
    # Statistical measures
    standard_error = Column(Float)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    
    # Estimation parameters
    estimation_window = Column(Integer)
    effective_observations = Column(Float)  # For exponentially weighted
    half_life = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('factor1_id', 'factor2_id', 'date', name='uq_factor_covariance'),
        Index('idx_covariance_date', 'date', 'covariance'),
    )


class PortfolioRiskAttribution(Base):
    """Portfolio risk attribution to factors."""
    
    __tablename__ = "barra_portfolio_risk_attributions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(100), nullable=False, index=True)
    factor_id = Column(Integer, ForeignKey("barra_risk_factors.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Risk attribution
    factor_exposure = Column(Float)  # Portfolio's exposure to this factor
    factor_risk_contribution = Column(Float)  # Risk contributed by this factor
    marginal_risk_contribution = Column(Float)  # Marginal contribution
    
    # Attribution components
    variance_contribution = Column(Float)
    covariance_contribution = Column(Float)
    interaction_contribution = Column(Float)
    
    # Percentage attributions
    percent_of_total_risk = Column(Float)
    percent_of_systematic_risk = Column(Float)
    
    # Active risk attribution (vs benchmark)
    active_exposure = Column(Float)
    active_risk_contribution = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    factor = relationship("RiskFactor")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'factor_id', 'date', name='uq_portfolio_factor_attribution'),
        Index('idx_portfolio_attribution_date', 'portfolio_id', 'date'),
    )


class ModelValidation(Base):
    """Model validation and backtesting results."""
    
    __tablename__ = "barra_model_validations"
    
    id = Column(Integer, primary_key=True, index=True)
    validation_date = Column(DateTime, nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    
    # Validation period
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    
    # Model performance metrics
    factor_r_squared = Column(Float)  # Average factor model R²
    bias_statistic = Column(Float)
    volatility_bias = Column(Float)
    
    # Forecast accuracy
    realized_volatility_rmse = Column(Float)
    forecast_correlation = Column(Float)
    
    # Coverage tests
    var_coverage_99 = Column(Float)  # % of observations within 99% VaR
    var_coverage_95 = Column(Float)  # % of observations within 95% VaR
    
    # Factor stability
    factor_turnover = Column(Float)  # % of factors changed
    exposure_stability = Column(Float)
    
    # Universe statistics
    entities_in_universe = Column(Integer)
    countries_covered = Column(Integer)
    industries_covered = Column(Integer)
    
    # Validation metadata
    validation_type = Column(String(50))  # out_of_sample, cross_validation
    notes = Column(Text)
    
    created_at = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_validation_date_version', 'validation_date', 'model_version'),
    ) 