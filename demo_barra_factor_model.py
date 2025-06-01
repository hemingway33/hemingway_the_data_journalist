#!/usr/bin/env python3
"""
Demo script for the Barra-style Factor Model System
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from projects.risk_factor_store.core.database import SessionLocal
from projects.risk_factor_store.risk_engine.barra_factor_engine import BarraFactorEngine
from projects.risk_factor_store.models.barra_factor_model import (
    RiskFactor, FactorType, Entity, FactorExposure, ExposureType
)

def create_sample_barra_factors():
    """Create sample Barra-style factors."""
    db = SessionLocal()
    
    try:
        # Style factors
        style_factors = [
            {
                'factor_id': 'STYLE_VALUE',
                'name': 'Value Factor',
                'description': 'Book-to-market and earnings yield composite',
                'factor_type': FactorType.STYLE,
                'level': 0,
                'is_systematic': True,
                'volatility_target': 0.15
            },
            {
                'factor_id': 'STYLE_QUALITY',
                'name': 'Quality Factor', 
                'description': 'Profitability and financial strength composite',
                'factor_type': FactorType.STYLE,
                'level': 0,
                'is_systematic': True,
                'volatility_target': 0.12
            },
            {
                'factor_id': 'STYLE_GROWTH',
                'name': 'Growth Factor',
                'description': 'Earnings and sales growth composite',
                'factor_type': FactorType.STYLE,
                'level': 0,
                'is_systematic': True,
                'volatility_target': 0.18
            },
            {
                'factor_id': 'STYLE_SIZE',
                'name': 'Size Factor',
                'description': 'Market capitalization factor',
                'factor_type': FactorType.STYLE,
                'level': 0,
                'is_systematic': True,
                'volatility_target': 0.10
            }
        ]
        
        # Industry factors
        industry_factors = [
            {
                'factor_id': 'IND_TECHNOLOGY',
                'name': 'Technology Industry',
                'description': 'Technology sector factor',
                'factor_type': FactorType.INDUSTRY,
                'level': 1,
                'is_systematic': True,
                'volatility_target': 0.25
            },
            {
                'factor_id': 'IND_FINANCIALS',
                'name': 'Financials Industry',
                'description': 'Financial services sector factor',
                'factor_type': FactorType.INDUSTRY,
                'level': 1,
                'is_systematic': True,
                'volatility_target': 0.22
            },
            {
                'factor_id': 'IND_HEALTHCARE',
                'name': 'Healthcare Industry',
                'description': 'Healthcare sector factor',
                'factor_type': FactorType.INDUSTRY,
                'level': 1,
                'is_systematic': True,
                'volatility_target': 0.18
            }
        ]
        
        # Country factors
        country_factors = [
            {
                'factor_id': 'COUNTRY_US',
                'name': 'United States',
                'description': 'US country factor',
                'factor_type': FactorType.COUNTRY,
                'level': 2,
                'country': 'US',
                'is_systematic': True,
                'volatility_target': 0.16
            },
            {
                'factor_id': 'COUNTRY_GB',
                'name': 'United Kingdom',
                'description': 'UK country factor',
                'factor_type': FactorType.COUNTRY,
                'level': 2,
                'country': 'GB',
                'is_systematic': True,
                'volatility_target': 0.18
            }
        ]
        
        # Macroeconomic factors
        macro_factors = [
            {
                'factor_id': 'MACRO_INTEREST_RATE',
                'name': 'Interest Rate Factor',
                'description': 'Interest rate sensitivity factor',
                'factor_type': FactorType.MACROECONOMIC,
                'level': 0,
                'is_systematic': True,
                'volatility_target': 0.20
            },
            {
                'factor_id': 'MACRO_INFLATION',
                'name': 'Inflation Factor',
                'description': 'Inflation sensitivity factor',
                'factor_type': FactorType.MACROECONOMIC,
                'level': 0,
                'is_systematic': True,
                'volatility_target': 0.14
            }
        ]
        
        all_factors = style_factors + industry_factors + country_factors + macro_factors
        
        for factor_data in all_factors:
            existing = db.query(RiskFactor).filter(
                RiskFactor.factor_id == factor_data['factor_id']
            ).first()
            
            if not existing:
                factor = RiskFactor(**factor_data)
                db.add(factor)
                print(f"✓ Created Barra factor: {factor_data['factor_id']}")
        
        db.commit()
        print(f"\n✓ Created {len(all_factors)} Barra-style factors")
        
    except Exception as e:
        print(f"✗ Error creating factors: {e}")
        db.rollback()
    finally:
        db.close()

def create_sample_entities():
    """Create sample entities for the universe."""
    db = SessionLocal()
    
    try:
        entities = [
            {
                'entity_id': 'AAPL',
                'name': 'Apple Inc.',
                'entity_type': 'equity',
                'sector': 'Technology',
                'industry': 'Technology Hardware',
                'country': 'US',
                'market_cap': 3000000000000,  # $3T
                'currency': 'USD',
                'is_in_universe': True
            },
            {
                'entity_id': 'MSFT',
                'name': 'Microsoft Corporation',
                'entity_type': 'equity',
                'sector': 'Technology',
                'industry': 'Software',
                'country': 'US',
                'market_cap': 2800000000000,  # $2.8T
                'currency': 'USD',
                'is_in_universe': True
            },
            {
                'entity_id': 'JPM',
                'name': 'JPMorgan Chase & Co.',
                'entity_type': 'equity',
                'sector': 'Financials',
                'industry': 'Banks',
                'country': 'US',
                'market_cap': 500000000000,  # $500B
                'currency': 'USD',
                'is_in_universe': True
            },
            {
                'entity_id': 'JNJ',
                'name': 'Johnson & Johnson',
                'entity_type': 'equity',
                'sector': 'Healthcare',
                'industry': 'Pharmaceuticals',
                'country': 'US',
                'market_cap': 450000000000,  # $450B
                'currency': 'USD',
                'is_in_universe': True
            }
        ]
        
        for entity_data in entities:
            existing = db.query(Entity).filter(
                Entity.entity_id == entity_data['entity_id']
            ).first()
            
            if not existing:
                entity = Entity(**entity_data)
                db.add(entity)
                print(f"✓ Created entity: {entity_data['entity_id']} - {entity_data['name']}")
        
        db.commit()
        print(f"\n✓ Created {len(entities)} sample entities")
        
    except Exception as e:
        print(f"✗ Error creating entities: {e}")
        db.rollback()
    finally:
        db.close()

def demonstrate_factor_exposures():
    """Demonstrate factor exposure calculation."""
    print("\n" + "="*60)
    print("BARRA FACTOR EXPOSURE CALCULATION")
    print("="*60)
    
    db = SessionLocal()
    engine = BarraFactorEngine(db)
    
    try:
        # Sample fundamental data for Apple
        apple_fundamentals = {
            'market_cap': 3000000000000,
            'book_value': 65000000000,
            'roe': 0.26,
            'debt_to_equity': 1.73,
            'current_ratio': 1.07,
            'sales_growth': 0.08,
            'industry': 'TECHNOLOGY',
            'country': 'US'
        }
        
        print(f"\nCalculating factor exposures for Apple Inc.")
        print(f"Market Cap: ${apple_fundamentals['market_cap']:,.0f}")
        print(f"ROE: {apple_fundamentals['roe']:.1%}")
        print(f"Debt/Equity: {apple_fundamentals['debt_to_equity']:.2f}")
        
        # Calculate exposures
        exposures = engine.calculate_fundamental_exposures(
            'AAPL', apple_fundamentals
        )
        
        print(f"\n📊 FACTOR EXPOSURES:")
        print(f"{'Factor':<20} {'Exposure':<10} {'Description'}")
        print("-" * 50)
        
        for factor, exposure in exposures.items():
            if factor.startswith('COUNTRY_') and exposure == 0:
                continue
            if factor.startswith('IND_') and exposure == 0:
                continue
                
            descriptions = {
                'VALUE': 'Book-to-market ratio',
                'QUALITY': 'Profitability & leverage',
                'PROFITABILITY': 'Return on equity',
                'LEVERAGE': 'Financial leverage',
                'GROWTH': 'Sales growth rate',
                'LIQUIDITY': 'Current ratio',
                'SIZE': 'Log market cap',
                'COUNTRY_US': 'US domicile',
                'IND_TECHNOLOGY': 'Technology sector'
            }
            
            desc = descriptions.get(factor, 'Factor exposure')
            print(f"{factor:<20} {exposure:>8.3f}  {desc}")
        
        return exposures
        
    except Exception as e:
        print(f"✗ Error calculating exposures: {e}")
        return {}
    finally:
        db.close()

def demonstrate_risk_decomposition():
    """Demonstrate risk decomposition."""
    print("\n" + "="*60)
    print("BARRA RISK DECOMPOSITION")
    print("="*60)
    
    db = SessionLocal()
    engine = BarraFactorEngine(db)
    
    try:
        # Sample exposures for Apple
        exposures = {
            'STYLE_VALUE': -0.5,      # Growth stock (negative value exposure)
            'STYLE_QUALITY': 1.2,    # High quality
            'STYLE_GROWTH': 0.8,     # Growth characteristics
            'STYLE_SIZE': 2.1,       # Large cap
            'IND_TECHNOLOGY': 1.0,   # Technology sector
            'COUNTRY_US': 1.0,       # US domicile
            'MACRO_INTEREST_RATE': -0.3,  # Interest rate sensitive
            'MACRO_INFLATION': 0.1   # Inflation exposure
        }
        
        # Sample factor covariance matrix (simplified)
        factor_names = list(exposures.keys())
        n_factors = len(factor_names)
        
        # Create a realistic factor covariance matrix
        np.random.seed(42)
        factor_vol = np.array([0.15, 0.12, 0.18, 0.10, 0.25, 0.16, 0.20, 0.14])  # Factor volatilities
        correlation_matrix = np.eye(n_factors)
        
        # Add some correlations
        correlation_matrix[0, 2] = -0.3  # Value vs Growth
        correlation_matrix[2, 0] = -0.3
        correlation_matrix[1, 3] = 0.2   # Quality vs Size
        correlation_matrix[3, 1] = 0.2
        correlation_matrix[6, 7] = 0.4   # Interest rate vs Inflation
        correlation_matrix[7, 6] = 0.4
        
        # Convert to covariance matrix
        factor_cov = np.outer(factor_vol, factor_vol) * correlation_matrix
        
        # Idiosyncratic variance
        idiosyncratic_var = 0.04  # 20% idiosyncratic volatility
        
        print(f"\nRisk decomposition for Apple Inc.")
        print(f"Factor exposures: {len(exposures)} systematic factors")
        print(f"Idiosyncratic variance: {np.sqrt(idiosyncratic_var)*np.sqrt(252):.1%} (annualized)")
        
        # Calculate risk decomposition
        risk_decomp = engine.calculate_entity_risk_decomposition(
            'AAPL', exposures, factor_cov, factor_names, idiosyncratic_var
        )
        
        print(f"\n📈 RISK DECOMPOSITION:")
        print(f"{'Component':<25} {'Risk (Ann.)':<12} {'Contribution'}")
        print("-" * 50)
        
        total_risk = risk_decomp['total_risk']
        systematic_risk = risk_decomp['systematic_risk']
        idiosyncratic_risk = risk_decomp['idiosyncratic_risk']
        
        print(f"{'Total Risk':<25} {total_risk:>10.1%}")
        print(f"{'Systematic Risk':<25} {systematic_risk:>10.1%}   {systematic_risk/total_risk:>8.1%}")
        print(f"{'Idiosyncratic Risk':<25} {idiosyncratic_risk:>10.1%}   {idiosyncratic_risk/total_risk:>8.1%}")
        
        print(f"\n📊 FACTOR GROUP CONTRIBUTIONS:")
        print(f"{'Factor Group':<25} {'Risk (Ann.)':<12} {'% of Total'}")
        print("-" * 50)
        
        for key, value in risk_decomp.items():
            if key.endswith('_risk') and key not in ['total_risk', 'systematic_risk', 'idiosyncratic_risk']:
                group_name = key.replace('_risk', '').title()
                contribution_pct = (value / total_risk) * 100 if total_risk > 0 else 0
                print(f"{group_name:<25} {value:>10.1%}   {contribution_pct:>8.1f}%")
        
        print(f"\n📋 RISK RATIOS:")
        print(f"Systematic Risk Ratio: {risk_decomp['systematic_risk_ratio']:.1%}")
        print(f"Diversification Ratio: {risk_decomp['diversification_ratio']:.1%}")
        
        return risk_decomp
        
    except Exception as e:
        print(f"✗ Error in risk decomposition: {e}")
        return {}
    finally:
        db.close()

def demonstrate_portfolio_analytics():
    """Demonstrate portfolio risk attribution."""
    print("\n" + "="*60)
    print("BARRA PORTFOLIO RISK ATTRIBUTION")
    print("="*60)
    
    db = SessionLocal()
    engine = BarraFactorEngine(db)
    
    try:
        # Sample portfolio positions
        positions = pd.DataFrame({
            'entity_id': ['AAPL', 'MSFT', 'JPM', 'JNJ'],
            'weight': [0.4, 0.3, 0.2, 0.1],
            'market_value': [400000, 300000, 200000, 100000]
        })
        positions.set_index('entity_id', inplace=True)
        
        # Sample factor exposures for each entity
        exposures_data = {
            'AAPL': {'STYLE_VALUE': -0.5, 'STYLE_QUALITY': 1.2, 'STYLE_GROWTH': 0.8, 'STYLE_SIZE': 2.1, 'IND_TECHNOLOGY': 1.0, 'COUNTRY_US': 1.0},
            'MSFT': {'STYLE_VALUE': -0.3, 'STYLE_QUALITY': 1.0, 'STYLE_GROWTH': 0.9, 'STYLE_SIZE': 2.0, 'IND_TECHNOLOGY': 1.0, 'COUNTRY_US': 1.0},
            'JPM': {'STYLE_VALUE': 0.8, 'STYLE_QUALITY': 0.5, 'STYLE_GROWTH': -0.2, 'STYLE_SIZE': 1.5, 'IND_FINANCIALS': 1.0, 'COUNTRY_US': 1.0},
            'JNJ': {'STYLE_VALUE': 0.2, 'STYLE_QUALITY': 0.9, 'STYLE_GROWTH': 0.1, 'STYLE_SIZE': 1.8, 'IND_HEALTHCARE': 1.0, 'COUNTRY_US': 1.0}
        }
        
        # Convert to DataFrame
        exposures_df = pd.DataFrame(exposures_data).T.fillna(0)
        
        # Factor covariance matrix
        factor_names = exposures_df.columns.tolist()
        n_factors = len(factor_names)
        np.random.seed(42)
        factor_vol = np.random.uniform(0.10, 0.25, n_factors)
        correlation_matrix = np.eye(n_factors) + np.random.uniform(-0.3, 0.3, (n_factors, n_factors)) * 0.1
        np.fill_diagonal(correlation_matrix, 1.0)
        factor_cov = np.outer(factor_vol, factor_vol) * correlation_matrix
        
        print(f"\nPortfolio composition:")
        print(f"{'Entity':<8} {'Weight':<8} {'Market Value'}")
        print("-" * 30)
        for entity, row in positions.iterrows():
            print(f"{entity:<8} {row['weight']:>6.1%} ${row['market_value']:>10,.0f}")
        
        total_value = positions['market_value'].sum()
        print(f"{'Total':<8} {'100.0%':<8} ${total_value:>10,.0f}")
        
        # Calculate portfolio risk attribution
        attribution = engine.calculate_portfolio_risk_attribution(
            'DEMO_PORTFOLIO', positions, exposures_df, factor_cov, factor_names
        )
        
        print(f"\n📊 PORTFOLIO FACTOR EXPOSURES:")
        print(f"{'Factor':<20} {'Exposure':<10} {'Risk Contrib.'}")
        print("-" * 40)
        
        portfolio_exposures = attribution['portfolio_exposures']
        factor_contributions = attribution['factor_contributions']
        
        for factor in factor_names:
            exposure = portfolio_exposures.get(factor, 0)
            risk_contrib = factor_contributions.get(factor, {}).get('risk_contribution', 0)
            if abs(exposure) > 0.01:  # Only show significant exposures
                print(f"{factor:<20} {exposure:>8.3f}   {risk_contrib:>8.1%}")
        
        print(f"\n📈 PORTFOLIO RISK METRICS:")
        systematic_risk = attribution['portfolio_systematic_risk']
        print(f"Portfolio Systematic Risk: {systematic_risk:.1%} (annualized)")
        
        attribution_summary = attribution['attribution_summary']
        print(f"Total Systematic Variance: {attribution_summary['total_systematic_var']:.6f}")
        print(f"Explained Variance: {attribution_summary['explained_variance']:.6f}")
        print(f"Interaction Effects: {attribution_summary['interaction_effects']:.6f}")
        
        return attribution
        
    except Exception as e:
        print(f"✗ Error in portfolio analytics: {e}")
        return {}
    finally:
        db.close()

def main():
    """Main demo function."""
    print("🎯 BARRA-STYLE FACTOR MODEL DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases a sophisticated multi-factor risk model")
    print("inspired by Barra's methodology for risk decomposition and")
    print("portfolio analytics.\n")
    
    # Setup
    print("📋 SETUP PHASE")
    print("-" * 30)
    create_sample_barra_factors()
    create_sample_entities()
    
    # Factor exposures
    exposures = demonstrate_factor_exposures()
    
    # Risk decomposition
    risk_decomp = demonstrate_risk_decomposition()
    
    # Portfolio analytics
    portfolio_attribution = demonstrate_portfolio_analytics()
    
    # Summary
    print("\n" + "="*60)
    print("BARRA MODEL SUMMARY")
    print("="*60)
    print("✓ Multi-factor risk model with hierarchical factor structure")
    print("✓ Fundamental and statistical factor exposure calculation")
    print("✓ Systematic vs idiosyncratic risk decomposition")
    print("✓ Factor covariance matrix estimation with shrinkage")
    print("✓ Portfolio risk attribution and marginal contributions")
    print("✓ Cross-sectional standardization and outlier capping")
    print("✓ Model validation and backtesting framework")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"The system demonstrates Barra's sophisticated approach to:")
    print(f"• Factor-based risk modeling")
    print(f"• Cross-sectional regression for factor returns")
    print(f"• Structured covariance estimation")
    print(f"• Portfolio risk attribution")
    
    return True

if __name__ == "__main__":
    main() 