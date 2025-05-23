"""
Market Factors Risk Model for SME Lending

This module provides a comprehensive framework for capturing systematic risks
affecting SME loan portfolios through macroeconomic, industry, and market factors.
The risk factor container allows for periodic updates and integration with 
survival models to adjust default probability predictions.

Key Features:
- Macroeconomic indicators (GDP, unemployment, inflation, etc.)
- Industry-specific factors (growth rates, default rates, regulatory changes)
- Market factors (credit spreads, volatility, lending standards)
- Regional/geographic risk factors
- Automated data retrieval from public APIs
- Manual data input capabilities
- Historical data storage and analysis
- Risk score calculation and portfolio adjustments
"""

import pandas as pd
import numpy as np
import requests
import json
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import yfinance as yf
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskFactor:
    """Individual risk factor definition with metadata"""
    name: str
    category: str  # 'macro', 'industry', 'market', 'regional'
    description: str
    data_source: str
    update_frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    api_endpoint: Optional[str] = None
    api_key_required: bool = False
    transformation: Optional[str] = None  # 'yoy_change', 'log', 'diff', etc.
    weight: float = 1.0
    risk_direction: str = 'positive'  # 'positive' = higher value = higher risk
    last_updated: Optional[datetime] = None
    current_value: Optional[float] = None
    historical_values: Dict = field(default_factory=dict)

class MarketFactorsContainer:
    """
    Comprehensive container for market risk factors affecting SME lending.
    
    Manages collection, storage, and analysis of systematic risk factors
    that influence loan portfolio performance beyond individual borrower characteristics.
    """
    
    def __init__(self, db_path: str = "market_factors.db"):
        """
        Initialize the market factors container.
        
        Parameters:
        -----------
        db_path : str
            Path to SQLite database for storing historical factor data
        """
        self.db_path = db_path
        self.factors = {}
        self.industry_mappings = {}
        self.regional_mappings = {}
        self._setup_database()
        self._initialize_factors()
        
    def _setup_database(self):
        """Setup SQLite database for historical factor storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create factors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_history (
                factor_name TEXT,
                date TEXT,
                value REAL,
                source TEXT,
                PRIMARY KEY (factor_name, date)
            )
        ''')
        
        # Create portfolio adjustments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_adjustments (
                date TEXT,
                adjustment_type TEXT,
                factor_name TEXT,
                adjustment_value REAL,
                portfolio_segment TEXT,
                PRIMARY KEY (date, adjustment_type, factor_name, portfolio_segment)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _initialize_factors(self):
        """Initialize comprehensive set of SME lending risk factors."""
        
        # MACROECONOMIC FACTORS
        macro_factors = [
            RiskFactor(
                name="gdp_growth_rate",
                category="macro",
                description="US GDP Growth Rate (Year-over-Year %)",
                data_source="FRED",
                update_frequency="quarterly",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={api_key}&file_type=json",
                api_key_required=True,
                transformation="yoy_change",
                weight=0.25,
                risk_direction="negative"  # Lower GDP growth = higher risk
            ),
            RiskFactor(
                name="unemployment_rate",
                category="macro", 
                description="US Unemployment Rate (%)",
                data_source="FRED",
                update_frequency="monthly",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={api_key}&file_type=json",
                api_key_required=True,
                weight=0.20,
                risk_direction="positive"  # Higher unemployment = higher risk
            ),
            RiskFactor(
                name="inflation_rate",
                category="macro",
                description="US CPI Inflation Rate (Year-over-Year %)",
                data_source="FRED",
                update_frequency="monthly", 
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={api_key}&file_type=json",
                api_key_required=True,
                transformation="yoy_change",
                weight=0.15,
                risk_direction="positive"  # Higher inflation = higher risk (for SMEs)
            ),
            RiskFactor(
                name="fed_funds_rate",
                category="macro",
                description="Federal Funds Rate (%)",
                data_source="FRED",
                update_frequency="daily",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={api_key}&file_type=json",
                api_key_required=True,
                weight=0.18,
                risk_direction="positive"  # Higher rates = higher risk for SMEs
            ),
            RiskFactor(
                name="consumer_confidence",
                category="macro",
                description="Consumer Confidence Index",
                data_source="FRED",
                update_frequency="monthly",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=CSCICP03USM665S&api_key={api_key}&file_type=json",
                api_key_required=True,
                weight=0.12,
                risk_direction="negative"  # Lower confidence = higher risk
            ),
            RiskFactor(
                name="business_confidence",
                category="macro",
                description="Small Business Optimism Index",
                data_source="NFIB",
                update_frequency="monthly",
                weight=0.22,
                risk_direction="negative"  # Lower optimism = higher risk
            ),
            RiskFactor(
                name="manufacturing_pmi",
                category="macro",
                description="ISM Manufacturing PMI",
                data_source="ISM",
                update_frequency="monthly",
                weight=0.15,
                risk_direction="negative"  # Below 50 = higher risk
            ),
            RiskFactor(
                name="services_pmi", 
                category="macro",
                description="ISM Services PMI",
                data_source="ISM",
                update_frequency="monthly",
                weight=0.18,
                risk_direction="negative"  # Below 50 = higher risk
            )
        ]
        
        # MARKET FACTORS
        market_factors = [
            RiskFactor(
                name="vix_volatility",
                category="market",
                description="VIX Volatility Index",
                data_source="Yahoo Finance",
                update_frequency="daily",
                weight=0.15,
                risk_direction="positive"  # Higher volatility = higher risk
            ),
            RiskFactor(
                name="credit_spreads",
                category="market", 
                description="High Yield Credit Spreads (bps)",
                data_source="FRED",
                update_frequency="daily",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=BAMLH0A0HYM2&api_key={api_key}&file_type=json",
                api_key_required=True,
                weight=0.25,
                risk_direction="positive"  # Higher spreads = higher risk
            ),
            RiskFactor(
                name="bank_lending_standards",
                category="market",
                description="Net % of Banks Tightening Standards for Commercial Loans",
                data_source="FRED",
                update_frequency="quarterly",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=DRTSCLCC&api_key={api_key}&file_type=json",
                api_key_required=True,
                weight=0.30,
                risk_direction="positive"  # Tighter standards = higher risk
            ),
            RiskFactor(
                name="corporate_bond_yield",
                category="market",
                description="Corporate Bond Yield (BBB)",
                data_source="FRED",
                update_frequency="daily",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=BAMLC0A4CBBB&api_key={api_key}&file_type=json",
                api_key_required=True,
                weight=0.20,
                risk_direction="positive"  # Higher yields = higher risk
            ),
            RiskFactor(
                name="commercial_real_estate",
                category="market",
                description="Commercial Real Estate Price Index",
                data_source="FRED",
                update_frequency="monthly",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=CPILFESL&api_key={api_key}&file_type=json",
                api_key_required=True,
                transformation="yoy_change",
                weight=0.15,
                risk_direction="negative"  # Lower prices = higher risk
            ),
            RiskFactor(
                name="small_business_lending_volume",
                category="market",
                description="Small Business Lending Volume ($ Billions)",
                data_source="SBA",
                update_frequency="monthly",
                weight=0.20,
                risk_direction="negative"  # Lower volume = higher risk
            )
        ]
        
        # INDUSTRY-SPECIFIC FACTORS
        industry_factors = [
            RiskFactor(
                name="retail_sales_growth",
                category="industry",
                description="Retail Sales Growth Rate (%)",
                data_source="FRED",
                update_frequency="monthly",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=RSAFS&api_key={api_key}&file_type=json",
                api_key_required=True,
                transformation="yoy_change",
                weight=0.20,
                risk_direction="negative"
            ),
            RiskFactor(
                name="construction_spending",
                category="industry",
                description="Construction Spending Growth (%)", 
                data_source="FRED",
                update_frequency="monthly",
                api_endpoint="https://api.stlouisfed.org/fred/series/observations?series_id=TTLCONS&api_key={api_key}&file_type=json",
                api_key_required=True,
                transformation="yoy_change",
                weight=0.25,
                risk_direction="negative"
            ),
            RiskFactor(
                name="technology_spending",
                category="industry",
                description="Business Technology Spending Index",
                data_source="BEA",
                update_frequency="quarterly",
                weight=0.15,
                risk_direction="negative"
            ),
            RiskFactor(
                name="energy_prices",
                category="industry",
                description="WTI Crude Oil Price ($/barrel)",
                data_source="EIA",
                update_frequency="daily",
                weight=0.18,
                risk_direction="positive"  # Higher energy costs = higher risk for most SMEs
            ),
            RiskFactor(
                name="supply_chain_stress",
                category="industry",
                description="Global Supply Chain Pressure Index",
                data_source="NY Fed",
                update_frequency="monthly",
                weight=0.22,
                risk_direction="positive"  # Higher stress = higher risk
            )
        ]
        
        # REGIONAL FACTORS
        regional_factors = [
            RiskFactor(
                name="regional_unemployment",
                category="regional",
                description="Regional Unemployment Rates by State",
                data_source="BLS",
                update_frequency="monthly",
                weight=0.30,
                risk_direction="positive"
            ),
            RiskFactor(
                name="regional_gdp_growth",
                category="regional",
                description="Regional GDP Growth by State",
                data_source="BEA",
                update_frequency="quarterly",
                weight=0.25,
                risk_direction="negative"
            ),
            RiskFactor(
                name="regional_real_estate",
                category="regional",
                description="Regional Real Estate Price Indices",
                data_source="FHFA",
                update_frequency="monthly",
                weight=0.20,
                risk_direction="negative"
            ),
            RiskFactor(
                name="business_formation_rate",
                category="regional",
                description="New Business Formation Rate by Region",
                data_source="Census",
                update_frequency="monthly",
                weight=0.15,
                risk_direction="negative"
            ),
            RiskFactor(
                name="regional_bank_health",
                category="regional",
                description="Regional Banking Sector Health Index",
                data_source="FDIC",
                update_frequency="quarterly",
                weight=0.25,
                risk_direction="negative"
            )
        ]
        
        # Combine all factors
        all_factors = macro_factors + market_factors + industry_factors + regional_factors
        
        # Store in factors dictionary
        for factor in all_factors:
            self.factors[factor.name] = factor
            
        # Initialize industry mappings
        self._setup_industry_mappings()
        
    def _setup_industry_mappings(self):
        """Setup industry-specific factor mappings."""
        self.industry_mappings = {
            'retail': ['retail_sales_growth', 'consumer_confidence', 'unemployment_rate'],
            'construction': ['construction_spending', 'regional_real_estate', 'fed_funds_rate'],
            'technology': ['technology_spending', 'business_confidence', 'vix_volatility'],
            'manufacturing': ['manufacturing_pmi', 'supply_chain_stress', 'energy_prices'],
            'services': ['services_pmi', 'consumer_confidence', 'regional_unemployment'],
            'energy': ['energy_prices', 'corporate_bond_yield', 'credit_spreads'],
            'healthcare': ['services_pmi', 'regional_gdp_growth', 'business_confidence'],
            'hospitality': ['consumer_confidence', 'unemployment_rate', 'regional_real_estate'],
            'transportation': ['energy_prices', 'supply_chain_stress', 'gdp_growth_rate'],
            'agriculture': ['commodity_prices', 'regional_real_estate', 'credit_spreads']
        }
        
    def get_factor_data_yahoo(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Retrieve factor data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_factor_data_fred(self, series_id: str, api_key: str, 
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve factor data from FRED API."""
        try:
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': api_key,
                'file_type': 'json'
            }
            
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
                
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            observations = data['observations']
            
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df = df.set_index('date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving FRED data for {series_id}: {e}")
            return pd.DataFrame()
    
    def update_factor_manual(self, factor_name: str, value: float, 
                           date: str = None, source: str = "manual"):
        """Manually update a factor value."""
        if factor_name not in self.factors:
            logger.warning(f"Factor {factor_name} not found")
            return
            
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        # Update factor object
        factor = self.factors[factor_name]
        factor.current_value = value
        factor.last_updated = datetime.now()
        factor.historical_values[date] = value
        
        # Store in database
        self._store_factor_value(factor_name, date, value, source)
        
        logger.info(f"Updated {factor_name} to {value} for {date}")
        
    def update_factor_automatic(self, factor_name: str, api_key: str = None):
        """Automatically update a factor from its data source."""
        if factor_name not in self.factors:
            logger.warning(f"Factor {factor_name} not found")
            return
            
        factor = self.factors[factor_name]
        
        try:
            if factor.data_source == "Yahoo Finance":
                # Handle Yahoo Finance factors
                if factor_name == "vix_volatility":
                    data = self.get_factor_data_yahoo("^VIX", period="5d")
                    if not data.empty:
                        latest_value = data['Close'].iloc[-1]
                        latest_date = data.index[-1].strftime('%Y-%m-%d')
                        self.update_factor_manual(factor_name, latest_value, latest_date, "Yahoo Finance")
                        
            elif factor.data_source == "FRED" and api_key:
                # Extract series ID from API endpoint or use predefined mapping
                series_mapping = {
                    'gdp_growth_rate': 'GDP',
                    'unemployment_rate': 'UNRATE',
                    'inflation_rate': 'CPIAUCSL',
                    'fed_funds_rate': 'FEDFUNDS',
                    'consumer_confidence': 'CSCICP03USM665S',
                    'credit_spreads': 'BAMLH0A0HYM2',
                    'bank_lending_standards': 'DRTSCLCC',
                    'corporate_bond_yield': 'BAMLC0A4CBBB',
                    'retail_sales_growth': 'RSAFS',
                    'construction_spending': 'TTLCONS'
                }
                
                if factor_name in series_mapping:
                    series_id = series_mapping[factor_name]
                    data = self.get_factor_data_fred(series_id, api_key)
                    
                    if not data.empty:
                        latest_value = data['value'].iloc[-1]
                        latest_date = data.index[-1].strftime('%Y-%m-%d')
                        
                        # Apply transformation if specified
                        if factor.transformation == "yoy_change":
                            if len(data) >= 12:  # Need at least 12 months for YoY
                                year_ago_value = data['value'].iloc[-13]  # 12 months ago
                                latest_value = ((latest_value - year_ago_value) / year_ago_value) * 100
                        
                        self.update_factor_manual(factor_name, latest_value, latest_date, "FRED")
            
        except Exception as e:
            logger.error(f"Error updating {factor_name}: {e}")
            
    def update_all_factors(self, api_key: str = None):
        """Update all factors that have automatic data sources."""
        logger.info("Starting automatic factor updates...")
        
        updated_count = 0
        for factor_name, factor in self.factors.items():
            if factor.data_source in ["Yahoo Finance", "FRED"]:
                try:
                    self.update_factor_automatic(factor_name, api_key)
                    updated_count += 1
                except Exception as e:
                    logger.error(f"Failed to update {factor_name}: {e}")
                    
        logger.info(f"Updated {updated_count} factors automatically")
        
    def _store_factor_value(self, factor_name: str, date: str, value: float, source: str):
        """Store factor value in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO factor_history 
            (factor_name, date, value, source)
            VALUES (?, ?, ?, ?)
        ''', (factor_name, date, value, source))
        
        conn.commit()
        conn.close()
        
    def get_factor_history(self, factor_name: str, 
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve historical factor data."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT date, value FROM factor_history WHERE factor_name = ?"
        params = [factor_name]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
        return df
        
    def calculate_risk_score(self, industry: str = None, region: str = None, 
                           date: str = None) -> Dict:
        """
        Calculate comprehensive risk score based on current factor values.
        
        Parameters:
        -----------
        industry : str
            Industry classification for industry-specific weighting
        region : str  
            Region for regional factor weighting
        date : str
            Date for historical risk score calculation
            
        Returns:
        --------
        Dict : Risk score breakdown and total score
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        risk_components = {
            'macro_risk': 0.0,
            'market_risk': 0.0,
            'industry_risk': 0.0,
            'regional_risk': 0.0,
            'total_risk': 0.0
        }
        
        factor_contributions = {}
        
        # Calculate risk score for each factor
        for factor_name, factor in self.factors.items():
            try:
                # Get factor value (current or historical)
                if factor.current_value is not None and date == datetime.now().strftime('%Y-%m-%d'):
                    value = factor.current_value
                else:
                    history = self.get_factor_history(factor_name, start_date=date, end_date=date)
                    if history.empty:
                        continue
                    value = history['value'].iloc[0]
                
                # Normalize factor value to risk score (0-1 scale)
                # This is a simplified approach - in practice, you'd want historical percentiles
                normalized_score = self._normalize_factor_value(factor_name, value)
                
                # Apply risk direction
                if factor.risk_direction == "negative":
                    normalized_score = 1 - normalized_score
                    
                # Weight the score
                weighted_score = normalized_score * factor.weight
                
                # Add to appropriate risk component
                risk_components[f"{factor.category}_risk"] += weighted_score
                factor_contributions[factor_name] = {
                    'value': value,
                    'normalized_score': normalized_score,
                    'weighted_score': weighted_score,
                    'category': factor.category
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate risk for {factor_name}: {e}")
                
        # Apply industry-specific adjustments
        if industry and industry in self.industry_mappings:
            industry_factors = self.industry_mappings[industry]
            industry_multiplier = 1.0
            
            for factor_name in industry_factors:
                if factor_name in factor_contributions:
                    # Increase weight for industry-relevant factors
                    factor_contributions[factor_name]['weighted_score'] *= 1.2
                    industry_multiplier += 0.1
                    
            risk_components['industry_risk'] *= industry_multiplier
            
        # Calculate total risk score (weighted average)
        category_weights = {
            'macro_risk': 0.35,
            'market_risk': 0.30, 
            'industry_risk': 0.25,
            'regional_risk': 0.10
        }
        
        total_risk = sum(risk_components[cat] * weight 
                        for cat, weight in category_weights.items())
        risk_components['total_risk'] = total_risk
        
        return {
            'risk_score': total_risk,
            'risk_components': risk_components,
            'factor_contributions': factor_contributions,
            'calculation_date': date,
            'industry': industry,
            'region': region
        }
        
    def _normalize_factor_value(self, factor_name: str, value: float) -> float:
        """
        Normalize factor value to 0-1 scale based on historical distribution.
        This is a simplified version - in practice, use historical percentiles.
        """
        # Get historical data for normalization
        history = self.get_factor_history(factor_name)
        
        if history.empty or len(history) < 10:
            # If insufficient history, use simple bounds
            factor_bounds = {
                'unemployment_rate': (3.0, 15.0),
                'inflation_rate': (-2.0, 8.0),
                'fed_funds_rate': (0.0, 8.0),
                'vix_volatility': (10.0, 80.0),
                'credit_spreads': (200, 2000),
                'gdp_growth_rate': (-5.0, 8.0)
            }
            
            if factor_name in factor_bounds:
                min_val, max_val = factor_bounds[factor_name]
                return max(0, min(1, (value - min_val) / (max_val - min_val)))
            else:
                return 0.5  # Neutral if no bounds defined
        else:
            # Use historical percentile
            percentile = (history['value'] <= value).mean()
            return percentile
            
    def get_portfolio_risk_adjustment(self, industry: str = None, 
                                    region: str = None) -> float:
        """
        Calculate portfolio-level risk adjustment multiplier.
        
        Returns:
        --------
        float : Risk adjustment multiplier for default probabilities
        """
        risk_analysis = self.calculate_risk_score(industry, region)
        total_risk = risk_analysis['risk_score']
        
        # Convert risk score to adjustment multiplier
        # Risk score of 0.5 = no adjustment (multiplier = 1.0)
        # Risk score of 1.0 = 50% increase in default probability (multiplier = 1.5)
        # Risk score of 0.0 = 25% decrease in default probability (multiplier = 0.75)
        
        if total_risk >= 0.5:
            multiplier = 1.0 + (total_risk - 0.5)  # 1.0 to 1.5
        else:
            multiplier = 0.75 + (total_risk * 0.5)  # 0.75 to 1.0
            
        return multiplier
        
    def generate_risk_report(self, industry: str = None, region: str = None) -> str:
        """Generate comprehensive risk factor report."""
        risk_analysis = self.calculate_risk_score(industry, region)
        
        report = []
        report.append("="*80)
        report.append("MARKET FACTORS RISK ASSESSMENT REPORT")
        report.append("="*80)
        
        report.append(f"\nAnalysis Date: {risk_analysis['calculation_date']}")
        if industry:
            report.append(f"Industry Focus: {industry.title()}")
        if region:
            report.append(f"Regional Focus: {region.title()}")
            
        report.append(f"\nOVERALL RISK SCORE: {risk_analysis['risk_score']:.3f}")
        
        # Risk level interpretation
        risk_score = risk_analysis['risk_score']
        if risk_score < 0.3:
            risk_level = "LOW RISK"
        elif risk_score < 0.7:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "HIGH RISK"
            
        report.append(f"Risk Level: {risk_level}")
        
        # Risk components breakdown
        report.append(f"\nRISK COMPONENTS:")
        components = risk_analysis['risk_components']
        for component, score in components.items():
            if component != 'total_risk':
                report.append(f"  {component.replace('_', ' ').title()}: {score:.3f}")
                
        # Top risk factors
        report.append(f"\nTOP RISK FACTORS:")
        factor_contribs = risk_analysis['factor_contributions']
        sorted_factors = sorted(factor_contribs.items(), 
                              key=lambda x: x[1]['weighted_score'], reverse=True)
        
        for i, (factor_name, contrib) in enumerate(sorted_factors[:10], 1):
            factor = self.factors[factor_name]
            report.append(f"  {i}. {factor.description}")
            report.append(f"     Current Value: {contrib['value']:.2f}")
            report.append(f"     Risk Contribution: {contrib['weighted_score']:.3f}")
            
        # Portfolio adjustment recommendation
        adjustment = self.get_portfolio_risk_adjustment(industry, region)
        report.append(f"\nPORTFOLIO ADJUSTMENT RECOMMENDATION:")
        report.append(f"  Default Probability Multiplier: {adjustment:.3f}")
        
        if adjustment > 1.1:
            report.append(f"  Recommendation: TIGHTEN underwriting criteria")
        elif adjustment < 0.9:
            report.append(f"  Recommendation: Consider RELAXING criteria")
        else:
            report.append(f"  Recommendation: MAINTAIN current criteria")
            
        report.append("\n" + "="*80)
        
        return "\n".join(report)
        
    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary of all factors and their current status."""
        summary_data = []
        
        for factor_name, factor in self.factors.items():
            summary_data.append({
                'Factor': factor_name,
                'Category': factor.category,
                'Description': factor.description,
                'Current Value': factor.current_value,
                'Last Updated': factor.last_updated,
                'Update Frequency': factor.update_frequency,
                'Data Source': factor.data_source,
                'Weight': factor.weight,
                'Risk Direction': factor.risk_direction
            })
            
        return pd.DataFrame(summary_data)


def main():
    """Demonstrate the market factors risk model."""
    print("="*80)
    print("MARKET FACTORS RISK MODEL FOR SME LENDING")
    print("="*80)
    
    # Initialize the container
    print("\n1. Initializing Market Factors Container...")
    market_factors = MarketFactorsContainer()
    
    print(f"Loaded {len(market_factors.factors)} risk factors")
    
    # Show factor summary
    print("\n2. FACTOR SUMMARY")
    print("-" * 60)
    summary = market_factors.get_factor_summary()
    
    # Group by category
    for category in ['macro', 'market', 'industry', 'regional']:
        cat_factors = summary[summary['Category'] == category]
        print(f"\n{category.upper()} FACTORS ({len(cat_factors)}):")
        for _, factor in cat_factors.iterrows():
            print(f"  • {factor['Description']}")
            print(f"    Source: {factor['Data Source']}, Update: {factor['Update Frequency']}")
    
    # Manual factor updates (simulating current market conditions)
    print("\n3. UPDATING SAMPLE FACTORS (Manual Input)")
    print("-" * 60)
    
    sample_updates = {
        'unemployment_rate': 3.7,  # Current US unemployment
        'fed_funds_rate': 5.25,    # Current Fed funds rate
        'inflation_rate': 3.2,     # Current inflation
        'vix_volatility': 18.5,    # Market volatility
        'gdp_growth_rate': 2.1,    # GDP growth
        'consumer_confidence': 102.0,  # Consumer confidence
        'business_confidence': 98.5,   # Business optimism
        'credit_spreads': 450,     # Credit spreads in bps
        'bank_lending_standards': 15.0  # % tightening standards
    }
    
    for factor_name, value in sample_updates.items():
        market_factors.update_factor_manual(factor_name, value)
        print(f"Updated {factor_name}: {value}")
    
    # Calculate risk scores for different scenarios
    print("\n4. RISK SCORE ANALYSIS")
    print("-" * 60)
    
    # Overall risk
    overall_risk = market_factors.calculate_risk_score()
    print(f"Overall Portfolio Risk Score: {overall_risk['risk_score']:.3f}")
    
    # Industry-specific risks
    industries = ['retail', 'construction', 'technology', 'manufacturing']
    print(f"\nIndustry-Specific Risk Scores:")
    for industry in industries:
        industry_risk = market_factors.calculate_risk_score(industry=industry)
        print(f"  {industry.title()}: {industry_risk['risk_score']:.3f}")
    
    # Portfolio adjustments
    print(f"\n5. PORTFOLIO ADJUSTMENT RECOMMENDATIONS")
    print("-" * 60)
    
    for industry in industries:
        adjustment = market_factors.get_portfolio_risk_adjustment(industry=industry)
        print(f"{industry.title()} Portfolio:")
        print(f"  Default Probability Multiplier: {adjustment:.3f}")
        
        if adjustment > 1.1:
            recommendation = "TIGHTEN underwriting"
        elif adjustment < 0.9:
            recommendation = "RELAX underwriting"
        else:
            recommendation = "MAINTAIN current standards"
        print(f"  Recommendation: {recommendation}")
        print()
    
    # Generate detailed risk report
    print("\n6. Generating detailed risk report for Technology sector...")
    tech_report = market_factors.generate_risk_report(industry='technology')
    print(tech_report)
    
    # Show automatic update capabilities
    print("\n7. AUTOMATIC UPDATE CAPABILITIES")
    print("-" * 60)
    print("The system supports automatic updates from:")
    
    auto_sources = {}
    for factor_name, factor in market_factors.factors.items():
        if factor.api_endpoint or factor.data_source == "Yahoo Finance":
            if factor.data_source not in auto_sources:
                auto_sources[factor.data_source] = []
            auto_sources[factor.data_source].append(factor_name)
    
    for source, factors in auto_sources.items():
        print(f"\n{source}:")
        for factor in factors[:5]:  # Show first 5
            print(f"  • {factor}")
        if len(factors) > 5:
            print(f"  ... and {len(factors)-5} more factors")
    
    print(f"\nTo enable automatic updates, provide API keys:")
    print(f"  • FRED API: market_factors.update_all_factors(api_key='your_fred_key')")
    print(f"  • Yahoo Finance: Automatically available (no key required)")
    
    print("\n" + "="*80)
    print("MARKET FACTORS RISK MODEL DEMONSTRATION COMPLETE")
    print("="*80)
    
    return market_factors


if __name__ == "__main__":
    market_factors = main()
