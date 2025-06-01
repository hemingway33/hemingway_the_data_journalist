"""
FRED (Federal Reserve Economic Data) collector for economic indicators.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import structlog
from fredapi import Fred

from .base_collector import BaseDataCollector
from ..core.config import settings
from ..models.risk_factors import RiskFactor

logger = structlog.get_logger()


class FREDCollector(BaseDataCollector):
    """Collector for FRED economic data."""
    
    def __init__(self, session, redis_client=None):
        super().__init__(session, redis_client)
        self.fred = None
        if settings.fred_api_key:
            try:
                self.fred = Fred(api_key=settings.fred_api_key)
            except Exception as e:
                logger.error("Failed to initialize FRED API", error=str(e))
    
    def get_supported_factors(self) -> List[str]:
        """Get list of supported FRED risk factors."""
        return [
            "FRED_GDP_GROWTH",
            "FRED_UNEMPLOYMENT_RATE", 
            "FRED_INFLATION_RATE",
            "FRED_FEDERAL_FUNDS_RATE",
            "FRED_10Y_TREASURY_YIELD",
            "FRED_CREDIT_SPREAD",
            "FRED_VIX",
            "FRED_CONSUMER_CONFIDENCE",
            "FRED_INDUSTRIAL_PRODUCTION",
            "FRED_REAL_ESTATE_PRICE_INDEX",
            "FRED_CURRENCY_INDEX",
            "FRED_COMMODITY_PRICE_INDEX"
        ]
    
    def get_fred_series_mapping(self) -> dict:
        """Map risk factor IDs to FRED series IDs."""
        return {
            "FRED_GDP_GROWTH": "GDP",
            "FRED_UNEMPLOYMENT_RATE": "UNRATE",
            "FRED_INFLATION_RATE": "CPIAUCNS",
            "FRED_FEDERAL_FUNDS_RATE": "FEDFUNDS",
            "FRED_10Y_TREASURY_YIELD": "GS10",
            "FRED_CREDIT_SPREAD": "BAA10Y",  # BAA Corporate Bond Yield - 10Y Treasury
            "FRED_VIX": "VIXCLS",
            "FRED_CONSUMER_CONFIDENCE": "UMCSENT",
            "FRED_INDUSTRIAL_PRODUCTION": "INDPRO",
            "FRED_REAL_ESTATE_PRICE_INDEX": "CSUSHPISA",
            "FRED_CURRENCY_INDEX": "DTWEXBGS",
            "FRED_COMMODITY_PRICE_INDEX": "PPIACO"
        }
    
    async def collect_data(self, risk_factor: RiskFactor) -> Optional[pd.DataFrame]:
        """
        Collect data from FRED API.
        
        Args:
            risk_factor: Risk factor to collect data for
            
        Returns:
            DataFrame with collected data or None if failed
        """
        if not self.fred:
            logger.error("FRED API not initialized", 
                        risk_factor_id=risk_factor.factor_id)
            return None
        
        series_mapping = self.get_fred_series_mapping()
        fred_series_id = series_mapping.get(risk_factor.factor_id)
        
        if not fred_series_id:
            logger.error("Unknown FRED series", 
                        risk_factor_id=risk_factor.factor_id)
            return None
        
        try:
            # Get data for the last year by default
            start_date = datetime.now() - timedelta(days=settings.lookback_days)
            
            # Fetch data from FRED
            data = self.fred.get_series(
                fred_series_id,
                start=start_date.strftime('%Y-%m-%d')
            )
            
            if data.empty:
                logger.warning("No data returned from FRED",
                             series_id=fred_series_id,
                             risk_factor_id=risk_factor.factor_id)
                return None
            
            # Convert to DataFrame format expected by base class
            df = pd.DataFrame({
                'date': data.index,
                'value': data.values,
                'raw_value': data.values
            })
            
            # Apply transformations based on risk factor type
            df = self._apply_transformations(df, risk_factor.factor_id)
            
            logger.info("Successfully collected FRED data",
                       series_id=fred_series_id,
                       risk_factor_id=risk_factor.factor_id,
                       records=len(df))
            
            return df
            
        except Exception as e:
            logger.error("Failed to collect FRED data",
                        series_id=fred_series_id,
                        risk_factor_id=risk_factor.factor_id,
                        error=str(e))
            return None
    
    def _apply_transformations(self, df: pd.DataFrame, factor_id: str) -> pd.DataFrame:
        """
        Apply factor-specific transformations.
        
        Args:
            df: Raw data DataFrame
            factor_id: Risk factor ID
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        if factor_id == "FRED_GDP_GROWTH":
            # Convert to year-over-year growth rate
            df['value'] = df['value'].pct_change(periods=4) * 100
            
        elif factor_id == "FRED_INFLATION_RATE":
            # Convert CPI to year-over-year inflation rate
            df['value'] = df['value'].pct_change(periods=12) * 100
            
        elif factor_id == "FRED_CREDIT_SPREAD":
            # If we have BAA10Y spread directly, use it
            # Otherwise calculate as BAA - 10Y Treasury
            pass
            
        elif factor_id in ["FRED_INDUSTRIAL_PRODUCTION", "FRED_REAL_ESTATE_PRICE_INDEX"]:
            # Convert to year-over-year growth rate
            df['value'] = df['value'].pct_change(periods=12) * 100
            
        elif factor_id == "FRED_COMMODITY_PRICE_INDEX":
            # Convert to year-over-year change
            df['value'] = df['value'].pct_change(periods=12) * 100
        
        # Remove rows with NaN values created by transformations
        df = df.dropna()
        
        return df
    
    async def get_series_info(self, series_id: str) -> Optional[dict]:
        """
        Get metadata about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Series metadata or None if failed
        """
        if not self.fred:
            return None
            
        try:
            info = self.fred.get_series_info(series_id)
            return {
                'title': info.get('title'),
                'units': info.get('units'),
                'frequency': info.get('frequency'),
                'last_updated': info.get('last_updated'),
                'notes': info.get('notes')
            }
        except Exception as e:
            logger.error("Failed to get FRED series info",
                        series_id=series_id,
                        error=str(e))
            return None 