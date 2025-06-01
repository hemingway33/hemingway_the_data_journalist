"""
Market data collector for stock indices, bonds, and market indicators.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import yfinance as yf
import structlog

from .base_collector import BaseDataCollector
from ..core.config import settings
from ..models.risk_factors import RiskFactor

logger = structlog.get_logger()


class MarketDataCollector(BaseDataCollector):
    """Collector for market data from Yahoo Finance and other sources."""
    
    def get_supported_factors(self) -> List[str]:
        """Get list of supported market risk factors."""
        return [
            "MARKET_SP500_INDEX",
            "MARKET_NASDAQ_INDEX", 
            "MARKET_DOW_JONES_INDEX",
            "MARKET_RUSSELL_2000_INDEX",
            "MARKET_10Y_TREASURY_PRICE",
            "MARKET_CORPORATE_BOND_INDEX",
            "MARKET_HIGH_YIELD_INDEX",
            "MARKET_CREDIT_DEFAULT_SWAP",
            "MARKET_VOLATILITY_INDEX",
            "MARKET_DOLLAR_INDEX",
            "MARKET_GOLD_PRICE",
            "MARKET_OIL_PRICE",
            "MARKET_EMERGING_MARKETS_INDEX"
        ]
    
    def get_ticker_mapping(self) -> dict:
        """Map risk factor IDs to ticker symbols."""
        return {
            "MARKET_SP500_INDEX": "^GSPC",
            "MARKET_NASDAQ_INDEX": "^IXIC",
            "MARKET_DOW_JONES_INDEX": "^DJI",
            "MARKET_RUSSELL_2000_INDEX": "^RUT",
            "MARKET_10Y_TREASURY_PRICE": "^TNX",
            "MARKET_CORPORATE_BOND_INDEX": "LQD",  # iShares iBoxx Investment Grade Corporate Bond ETF
            "MARKET_HIGH_YIELD_INDEX": "HYG",  # iShares iBoxx High Yield Corporate Bond ETF
            "MARKET_CREDIT_DEFAULT_SWAP": "CDX",  # Approximation
            "MARKET_VOLATILITY_INDEX": "^VIX",
            "MARKET_DOLLAR_INDEX": "DX-Y.NYB",
            "MARKET_GOLD_PRICE": "GC=F",
            "MARKET_OIL_PRICE": "CL=F",
            "MARKET_EMERGING_MARKETS_INDEX": "EEM"  # iShares MSCI Emerging Markets ETF
        }
    
    async def collect_data(self, risk_factor: RiskFactor) -> Optional[pd.DataFrame]:
        """
        Collect market data from Yahoo Finance.
        
        Args:
            risk_factor: Risk factor to collect data for
            
        Returns:
            DataFrame with collected data or None if failed
        """
        ticker_mapping = self.get_ticker_mapping()
        ticker = ticker_mapping.get(risk_factor.factor_id)
        
        if not ticker:
            logger.error("Unknown market ticker", 
                        risk_factor_id=risk_factor.factor_id)
            return None
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=settings.lookback_days)
            
            # Fetch data using yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if data.empty:
                logger.warning("No market data returned",
                             ticker=ticker,
                             risk_factor_id=risk_factor.factor_id)
                return None
            
            # Use adjusted close price as the primary value
            df = pd.DataFrame({
                'date': data.index,
                'value': data['Adj Close'].values,
                'raw_value': data['Close'].values,
                'volume': data['Volume'].values,
                'high': data['High'].values,
                'low': data['Low'].values,
                'open': data['Open'].values
            })
            
            # Apply factor-specific transformations
            df = self._apply_transformations(df, risk_factor.factor_id)
            
            # Calculate additional risk metrics
            df = self._calculate_risk_metrics(df)
            
            logger.info("Successfully collected market data",
                       ticker=ticker,
                       risk_factor_id=risk_factor.factor_id,
                       records=len(df))
            
            return df
            
        except Exception as e:
            logger.error("Failed to collect market data",
                        ticker=ticker,
                        risk_factor_id=risk_factor.factor_id,
                        error=str(e))
            return None
    
    def _apply_transformations(self, df: pd.DataFrame, factor_id: str) -> pd.DataFrame:
        """
        Apply factor-specific transformations.
        
        Args:
            df: Raw market data DataFrame
            factor_id: Risk factor ID
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        if factor_id in ["MARKET_SP500_INDEX", "MARKET_NASDAQ_INDEX", 
                        "MARKET_DOW_JONES_INDEX", "MARKET_RUSSELL_2000_INDEX"]:
            # For equity indices, we might want returns or levels
            # Keep both price level and calculate returns
            df['returns'] = df['value'].pct_change()
            
        elif factor_id == "MARKET_10Y_TREASURY_PRICE":
            # Treasury yield - convert to yield if price is given
            # Assuming ^TNX gives yield directly
            pass
            
        elif factor_id in ["MARKET_CORPORATE_BOND_INDEX", "MARKET_HIGH_YIELD_INDEX"]:
            # Bond indices - calculate returns
            df['returns'] = df['value'].pct_change()
            
        elif factor_id == "MARKET_VOLATILITY_INDEX":
            # VIX is already in the right format (volatility percentage)
            pass
            
        elif factor_id in ["MARKET_GOLD_PRICE", "MARKET_OIL_PRICE"]:
            # Commodity prices - calculate returns
            df['returns'] = df['value'].pct_change()
            
        return df
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional risk metrics.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            DataFrame with additional risk metrics
        """
        df = df.copy()
        
        # Calculate rolling volatility (annualized)
        window = min(settings.volatility_window, len(df))
        if 'returns' in df.columns and window > 1:
            df['volatility'] = (df['returns'].rolling(window=window).std() * 
                              (252 ** 0.5))  # Annualized volatility
        
        # Calculate rolling correlation with market (if not market itself)
        # This would require additional market data - placeholder for now
        
        # Calculate momentum indicators
        if len(df) >= 20:
            df['sma_20'] = df['value'].rolling(window=20).mean()
            df['momentum_20'] = (df['value'] / df['sma_20'] - 1) * 100
        
        if len(df) >= 50:
            df['sma_50'] = df['value'].rolling(window=50).mean()
            df['momentum_50'] = (df['value'] / df['sma_50'] - 1) * 100
        
        # Calculate relative strength index (RSI)
        df['rsi'] = self._calculate_rsi(df['value'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            window: RSI calculation window
            
        Returns:
            RSI series
        """
        if len(prices) <= window:
            return pd.Series([None] * len(prices), index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def get_market_info(self, ticker: str) -> Optional[dict]:
        """
        Get information about a market ticker.
        
        Args:
            ticker: Market ticker symbol
            
        Returns:
            Ticker information or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('shortName', info.get('longName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'currency': info.get('currency'),
                'exchange': info.get('exchange'),
                'description': info.get('longBusinessSummary', '')[:500]  # Truncate
            }
            
        except Exception as e:
            logger.error("Failed to get market info",
                        ticker=ticker,
                        error=str(e))
            return None 