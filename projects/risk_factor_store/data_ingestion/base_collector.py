"""
Base data collector class for risk factor ingestion.
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import structlog
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db, get_redis
from ..models.risk_factors import RiskFactor, RiskFactorValue, DataQualityCheck

logger = structlog.get_logger()


class BaseDataCollector(ABC):
    """Base class for data collectors."""
    
    def __init__(self, session: Session, redis_client=None):
        """
        Initialize the data collector.
        
        Args:
            session: Database session
            redis_client: Redis client for caching
        """
        self.session = session
        self.redis_client = redis_client or get_redis()
        self.max_retries = settings.max_retries
        self.request_timeout = settings.request_timeout
        
    @abstractmethod
    async def collect_data(self, risk_factor: RiskFactor) -> Optional[pd.DataFrame]:
        """
        Collect data for a specific risk factor.
        
        Args:
            risk_factor: Risk factor to collect data for
            
        Returns:
            DataFrame with collected data or None if failed
        """
        pass
    
    @abstractmethod
    def get_supported_factors(self) -> List[str]:
        """
        Get list of risk factor IDs supported by this collector.
        
        Returns:
            List of supported risk factor IDs
        """
        pass
    
    async def fetch_with_retry(self, url: str, params: Dict = None, 
                              headers: Dict = None) -> Optional[Dict]:
        """
        Fetch data from URL with retry logic.
        
        Args:
            url: URL to fetch from
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, params=params, 
                                         headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.warning(
                                "HTTP error", 
                                url=url, 
                                status=response.status,
                                attempt=attempt + 1
                            )
            except Exception as e:
                logger.error(
                    "Request failed",
                    url=url,
                    error=str(e),
                    attempt=attempt + 1
                )
                
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        return None
    
    def validate_data(self, data: pd.DataFrame, risk_factor: RiskFactor) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean collected data.
        
        Args:
            data: Raw data DataFrame
            risk_factor: Risk factor metadata
            
        Returns:
            Tuple of (cleaned_data, quality_metrics)
        """
        quality_metrics = {
            "records_checked": len(data),
            "missing_values": 0,
            "outliers_detected": 0,
            "validation_errors": [],
            "completeness_score": 1.0,
            "validity_score": 1.0,
            "consistency_score": 1.0,
            "timeliness_score": 1.0
        }
        
        if data.empty:
            quality_metrics["completeness_score"] = 0.0
            quality_metrics["validation_errors"].append("No data collected")
            return data, quality_metrics
        
        # Check for missing values
        missing_count = data['value'].isna().sum()
        quality_metrics["missing_values"] = missing_count
        quality_metrics["completeness_score"] = 1.0 - (missing_count / len(data))
        
        # Remove rows with missing values
        data_clean = data.dropna(subset=['value'])
        
        # Outlier detection using IQR method
        if len(data_clean) > 0:
            Q1 = data_clean['value'].quantile(0.25)
            Q3 = data_clean['value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data_clean['value'] < lower_bound) | 
                       (data_clean['value'] > upper_bound))
            outlier_count = outliers.sum()
            quality_metrics["outliers_detected"] = outlier_count
            
            if outlier_count > 0:
                quality_metrics["validity_score"] = 1.0 - (outlier_count / len(data_clean))
                logger.warning(
                    "Outliers detected",
                    risk_factor_id=risk_factor.factor_id,
                    outlier_count=outlier_count
                )
        
        # Check data freshness
        if not data_clean.empty:
            latest_date = pd.to_datetime(data_clean['date']).max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 7:  # Data older than 7 days
                quality_metrics["timeliness_score"] = max(0.0, 1.0 - (days_old / 30))
        
        # Calculate overall quality score
        weights = {
            "completeness_score": 0.3,
            "validity_score": 0.3,
            "consistency_score": 0.2,
            "timeliness_score": 0.2
        }
        
        quality_metrics["overall_score"] = sum(
            quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return data_clean, quality_metrics
    
    def store_data(self, data: pd.DataFrame, risk_factor: RiskFactor,
                   quality_metrics: Dict) -> bool:
        """
        Store validated data in the database.
        
        Args:
            data: Cleaned data DataFrame
            risk_factor: Risk factor metadata
            quality_metrics: Data quality metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store risk factor values
            for _, row in data.iterrows():
                factor_value = RiskFactorValue(
                    risk_factor_id=risk_factor.id,
                    date=pd.to_datetime(row['date']),
                    value=float(row['value']),
                    raw_value=float(row.get('raw_value', row['value'])),
                    quality_score=quality_metrics["overall_score"],
                    is_interpolated=row.get('is_interpolated', False),
                    value_metadata=row.get('metadata', {})
                )
                
                # Use merge/upsert pattern
                existing = self.session.query(RiskFactorValue).filter(
                    RiskFactorValue.risk_factor_id == risk_factor.id,
                    RiskFactorValue.date == factor_value.date
                ).first()
                
                if existing:
                    existing.value = factor_value.value
                    existing.raw_value = factor_value.raw_value
                    existing.quality_score = factor_value.quality_score
                    existing.is_interpolated = factor_value.is_interpolated
                    existing.value_metadata = factor_value.value_metadata
                else:
                    self.session.add(factor_value)
            
            # Store quality check results
            quality_check = DataQualityCheck(
                check_date=datetime.now(),
                risk_factor_id=risk_factor.id,
                completeness_score=quality_metrics["completeness_score"],
                validity_score=quality_metrics["validity_score"],
                consistency_score=quality_metrics["consistency_score"],
                timeliness_score=quality_metrics["timeliness_score"],
                overall_score=quality_metrics["overall_score"],
                records_checked=quality_metrics["records_checked"],
                missing_values=quality_metrics["missing_values"],
                outliers_detected=quality_metrics["outliers_detected"],
                validation_errors=quality_metrics["validation_errors"]
            )
            self.session.add(quality_check)
            
            self.session.commit()
            
            # Cache latest value in Redis
            if not data.empty:
                latest_value = data.iloc[-1]
                cache_key = f"risk_factor:{risk_factor.factor_id}:latest"
                cache_data = {
                    "value": float(latest_value['value']),
                    "date": latest_value['date'].isoformat(),
                    "quality_score": quality_metrics["overall_score"]
                }
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour expiry
                    str(cache_data)
                )
            
            logger.info(
                "Data stored successfully",
                risk_factor_id=risk_factor.factor_id,
                records_stored=len(data),
                quality_score=quality_metrics["overall_score"]
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to store data",
                risk_factor_id=risk_factor.factor_id,
                error=str(e)
            )
            self.session.rollback()
            return False
    
    async def process_risk_factor(self, risk_factor: RiskFactor) -> bool:
        """
        Complete processing pipeline for a risk factor.
        
        Args:
            risk_factor: Risk factor to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(
                "Processing risk factor",
                risk_factor_id=risk_factor.factor_id,
                category=risk_factor.category.value
            )
            
            # Collect data
            raw_data = await self.collect_data(risk_factor)
            if raw_data is None or raw_data.empty:
                logger.warning(
                    "No data collected",
                    risk_factor_id=risk_factor.factor_id
                )
                return False
            
            # Validate and clean data
            clean_data, quality_metrics = self.validate_data(raw_data, risk_factor)
            
            # Store data
            success = self.store_data(clean_data, risk_factor, quality_metrics)
            
            return success
            
        except Exception as e:
            logger.error(
                "Failed to process risk factor",
                risk_factor_id=risk_factor.factor_id,
                error=str(e)
            )
            return False 