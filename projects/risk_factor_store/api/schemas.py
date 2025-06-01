"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


# Enums
class RiskFactorCategoryEnum(str, Enum):
    ECONOMIC = "economic"
    MARKET = "market"
    CREDIT = "credit"
    INDUSTRY = "industry"
    MACROECONOMIC = "macroeconomic"
    ALTERNATIVE = "alternative"
    REGULATORY = "regulatory"


class DataSourceEnum(str, Enum):
    FRED = "fred"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    NEWS_API = "news_api"
    WEB_SCRAPING = "web_scraping"
    CREDIT_BUREAU = "credit_bureau"
    INTERNAL = "internal"


class EntityTypeEnum(str, Enum):
    INDIVIDUAL = "individual"
    CORPORATE = "corporate"
    SOVEREIGN = "sovereign"


# Risk Factor Schemas
class RiskFactorBase(BaseModel):
    factor_id: str = Field(..., description="Unique risk factor identifier")
    name: str = Field(..., description="Risk factor name")
    description: Optional[str] = Field(None, description="Risk factor description")
    category: RiskFactorCategoryEnum
    data_source: DataSourceEnum
    source_identifier: Optional[str] = Field(None, description="External system identifier")
    unit: Optional[str] = Field(None, description="Measurement unit")
    frequency: Optional[str] = Field(None, description="Data frequency")


class RiskFactorCreate(RiskFactorBase):
    pass


class RiskFactorUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    frequency: Optional[str] = None
    is_active: Optional[bool] = None


class RiskFactorValue(BaseModel):
    date: datetime
    value: float
    raw_value: Optional[float] = None
    quality_score: Optional[float] = Field(default=1.0, ge=0, le=1)
    is_interpolated: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None


class RiskFactor(RiskFactorBase):
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RiskFactorWithValues(RiskFactor):
    latest_value: Optional[RiskFactorValue] = None
    recent_values: List[RiskFactorValue] = []


# Credit Entity Schemas
class CreditEntityBase(BaseModel):
    entity_id: str = Field(..., description="Unique entity identifier")
    entity_type: EntityTypeEnum
    name: Optional[str] = None
    industry_code: Optional[str] = None
    country: Optional[str] = Field(None, description="ISO country code")
    credit_rating: Optional[str] = None


class CreditEntityCreate(CreditEntityBase):
    pass


class CreditEntityUpdate(BaseModel):
    name: Optional[str] = None
    industry_code: Optional[str] = None
    country: Optional[str] = None
    credit_rating: Optional[str] = None
    is_active: Optional[bool] = None


class CreditEntity(CreditEntityBase):
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Credit Scoring Schemas
class EntityFeatures(BaseModel):
    # Entity demographics
    age_years: Optional[float] = Field(default=0, description="Entity age in years")
    total_assets: Optional[float] = Field(default=1, description="Total assets")
    debt_to_equity: Optional[float] = Field(default=0, description="Debt-to-equity ratio")
    current_ratio: Optional[float] = Field(default=1, description="Current ratio")
    roe: Optional[float] = Field(default=0, description="Return on equity")
    asset_turnover: Optional[float] = Field(default=0, description="Asset turnover")
    
    # Industry and geography
    industry_risk_score: Optional[float] = Field(default=0.5, ge=0, le=1)
    country_risk_score: Optional[float] = Field(default=0.5, ge=0, le=1)
    
    # Credit history
    payment_history_score: Optional[float] = Field(default=0.5, ge=0, le=1)
    credit_utilization: Optional[float] = Field(default=0, ge=0, le=1)
    number_of_accounts: Optional[int] = Field(default=0, ge=0)
    recent_inquiries: Optional[int] = Field(default=0, ge=0)
    
    # Exposure details
    current_exposure: Optional[float] = Field(default=0, description="Current exposure amount")
    credit_limit: Optional[float] = Field(default=0, description="Credit limit")
    exposure_amount: Optional[float] = Field(default=1000000, description="Total exposure")
    facility_type: Optional[str] = Field(default="term", description="Facility type")
    seniority: Optional[str] = Field(default="senior_unsecured", description="Debt seniority")
    industry: Optional[str] = Field(default="", description="Industry sector")
    entity_type: Optional[str] = Field(default="corporate", description="Entity type")


class CreditScoringRequest(BaseModel):
    entity_id: str
    entity_features: EntityFeatures
    as_of_date: Optional[datetime] = None
    model_type: Optional[str] = Field(default="xgboost", description="Model type to use")


class CreditScore(BaseModel):
    entity_id: str
    score_date: datetime
    probability_of_default: float = Field(..., ge=0, le=1)
    confidence_interval_lower: Optional[float] = Field(None, ge=0, le=1)
    confidence_interval_upper: Optional[float] = Field(None, ge=0, le=1)
    loss_given_default: float = Field(..., ge=0, le=1)
    exposure_at_default: float = Field(..., ge=0)
    expected_loss: float = Field(..., ge=0)
    risk_rating: str
    model_version: str
    model_type: str
    risk_factors_snapshot: Dict[str, float]


# Portfolio Schemas
class PortfolioBase(BaseModel):
    portfolio_id: str = Field(..., description="Unique portfolio identifier")
    name: str = Field(..., description="Portfolio name")
    description: Optional[str] = None
    portfolio_type: Optional[str] = Field(default="mixed", description="Portfolio type")
    currency: Optional[str] = Field(default="USD", description="Portfolio currency")


class PortfolioCreate(PortfolioBase):
    pass


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    portfolio_type: Optional[str] = None
    currency: Optional[str] = None
    is_active: Optional[bool] = None


class Portfolio(PortfolioBase):
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PortfolioPosition(BaseModel):
    entity_id: str
    notional_amount: float = Field(..., gt=0)
    market_value: Optional[float] = None
    weight: Optional[float] = Field(None, ge=0, le=1)
    duration: Optional[float] = None
    modified_duration: Optional[float] = None
    convexity: Optional[float] = None


class PortfolioPositionCreate(PortfolioPosition):
    portfolio_id: str
    position_date: datetime


# Portfolio Analytics Schemas
class PortfolioStatistics(BaseModel):
    total_notional: float
    total_market_value: float
    number_of_positions: int
    portfolio_duration: float
    portfolio_modified_duration: float
    portfolio_convexity: float
    top_10_concentration: float
    herfindahl_index: float
    industry_concentration: float
    country_concentration: float


class PortfolioCreditMetrics(BaseModel):
    total_exposure: float
    average_pd: float
    average_lgd: float
    expected_loss: float
    expected_loss_rate: float


class PortfolioVaRMetrics(BaseModel):
    var: float
    expected_shortfall: float
    portfolio_volatility: float
    confidence_level: float
    time_horizon: int


class StressTestScenario(BaseModel):
    name: str
    factor_shocks: Dict[str, float] = Field(..., description="Factor shock values")


class StressTestResult(BaseModel):
    base_value: float
    stressed_value: float
    portfolio_pnl: float
    portfolio_return: float


class RiskContribution(BaseModel):
    entity_id: str
    entity_name: str
    market_value: float
    size_contribution: float
    risk_contribution: float


class PositionSummary(BaseModel):
    total_positions: int
    industries: int
    countries: int
    rating_distribution: Dict[str, int]


class PortfolioReport(BaseModel):
    portfolio_id: str
    report_date: datetime
    basic_statistics: PortfolioStatistics
    credit_metrics: PortfolioCreditMetrics
    var_metrics: PortfolioVaRMetrics
    stress_test_results: Dict[str, StressTestResult]
    top_risk_contributors: List[RiskContribution]
    positions_summary: PositionSummary


# Data Quality Schemas
class DataQualityMetrics(BaseModel):
    completeness_score: float = Field(..., ge=0, le=1)
    validity_score: float = Field(..., ge=0, le=1)
    consistency_score: float = Field(..., ge=0, le=1)
    timeliness_score: float = Field(..., ge=0, le=1)
    overall_score: float = Field(..., ge=0, le=1)
    records_checked: int = Field(..., ge=0)
    missing_values: int = Field(..., ge=0)
    outliers_detected: int = Field(..., ge=0)
    validation_errors: List[str] = []


# Response Schemas
class MessageResponse(BaseModel):
    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int 