"""
Setup script for the Credit Risk Factor Store.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from projects.risk_factor_store.core.database import create_tables, drop_tables
from projects.risk_factor_store.core.config import settings
from projects.risk_factor_store.models.risk_factors import (
    RiskFactor, RiskFactorCategory, DataSource
)
from projects.risk_factor_store.core.database import SessionLocal
import structlog

logger = structlog.get_logger()


def create_sample_risk_factors():
    """Create sample risk factors for testing."""
    db = SessionLocal()
    
    try:
        # Economic risk factors
        economic_factors = [
            {
                "factor_id": "FRED_GDP_GROWTH",
                "name": "GDP Growth Rate",
                "description": "Year-over-year GDP growth rate",
                "category": RiskFactorCategory.ECONOMIC,
                "data_source": DataSource.FRED,
                "source_identifier": "GDP",
                "unit": "Percentage",
                "frequency": "quarterly"
            },
            {
                "factor_id": "FRED_UNEMPLOYMENT_RATE",
                "name": "Unemployment Rate",
                "description": "National unemployment rate",
                "category": RiskFactorCategory.ECONOMIC,
                "data_source": DataSource.FRED,
                "source_identifier": "UNRATE",
                "unit": "Percentage",
                "frequency": "monthly"
            },
            {
                "factor_id": "FRED_INFLATION_RATE",
                "name": "Inflation Rate",
                "description": "Consumer Price Index inflation rate",
                "category": RiskFactorCategory.ECONOMIC,
                "data_source": DataSource.FRED,
                "source_identifier": "CPIAUCNS",
                "unit": "Percentage",
                "frequency": "monthly"
            }
        ]
        
        # Market risk factors
        market_factors = [
            {
                "factor_id": "MARKET_SP500_INDEX",
                "name": "S&P 500 Index",
                "description": "S&P 500 stock market index",
                "category": RiskFactorCategory.MARKET,
                "data_source": DataSource.YAHOO_FINANCE,
                "source_identifier": "^GSPC",
                "unit": "Index Points",
                "frequency": "daily"
            },
            {
                "factor_id": "MARKET_VOLATILITY_INDEX",
                "name": "VIX Volatility Index",
                "description": "CBOE Volatility Index",
                "category": RiskFactorCategory.MARKET,
                "data_source": DataSource.YAHOO_FINANCE,
                "source_identifier": "^VIX",
                "unit": "Percentage",
                "frequency": "daily"
            }
        ]
        
        all_factors = economic_factors + market_factors
        
        for factor_data in all_factors:
            # Check if factor already exists
            existing = db.query(RiskFactor).filter(
                RiskFactor.factor_id == factor_data["factor_id"]
            ).first()
            
            if not existing:
                risk_factor = RiskFactor(**factor_data)
                db.add(risk_factor)
                logger.info("Created risk factor", factor_id=factor_data["factor_id"])
        
        db.commit()
        logger.info("Sample risk factors created successfully")
        
    except Exception as e:
        logger.error("Failed to create sample risk factors", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()


def setup_database():
    """Set up the database with tables and sample data."""
    logger.info("Setting up database...")
    
    try:
        # Create tables
        create_tables()
        logger.info("Database tables created")
        
        # Create sample risk factors
        create_sample_risk_factors()
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error("Database setup failed", error=str(e))
        raise


def reset_database():
    """Reset the database by dropping and recreating all tables."""
    logger.warning("Resetting database - all data will be lost!")
    
    try:
        # Drop all tables
        drop_tables()
        logger.info("Database tables dropped")
        
        # Recreate tables and sample data
        setup_database()
        
        logger.info("Database reset completed successfully")
        
    except Exception as e:
        logger.error("Database reset failed", error=str(e))
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Credit Risk Factor Store Setup")
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset database (WARNING: destroys all data)"
    )
    
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    else:
        setup_database() 