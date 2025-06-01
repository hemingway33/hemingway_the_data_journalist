"""
Portfolio analytics engine for portfolio-level risk assessment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.optimize import minimize
import structlog
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.risk_factors import (
    Portfolio, PortfolioPosition, PortfolioRiskMetric, 
    CreditEntity, CreditRiskScore, RiskFactorValue
)
from .credit_scoring import CreditScoringEngine

logger = structlog.get_logger()


class PortfolioAnalyticsEngine:
    """Portfolio analytics engine for risk assessment."""
    
    def __init__(self, session: Session):
        """
        Initialize portfolio analytics engine.
        
        Args:
            session: Database session
        """
        self.session = session
        self.credit_engine = CreditScoringEngine(session)
        
    def get_portfolio_positions(self, portfolio_id: str, 
                              as_of_date: datetime = None) -> pd.DataFrame:
        """
        Get portfolio positions as of a specific date.
        
        Args:
            portfolio_id: Portfolio identifier
            as_of_date: Position date
            
        Returns:
            DataFrame with portfolio positions
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Get portfolio
        portfolio = self.session.query(Portfolio).filter(
            Portfolio.portfolio_id == portfolio_id,
            Portfolio.is_active == True
        ).first()
        
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        # Get positions
        positions = self.session.query(PortfolioPosition).join(CreditEntity).filter(
            PortfolioPosition.portfolio_id == portfolio.id,
            PortfolioPosition.position_date <= as_of_date,
            PortfolioPosition.is_active == True
        ).all()
        
        if not positions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        position_data = []
        for pos in positions:
            position_data.append({
                'position_id': pos.id,
                'entity_id': pos.entity.entity_id,
                'entity_name': pos.entity.name,
                'entity_type': pos.entity.entity_type,
                'industry_code': pos.entity.industry_code,
                'country': pos.entity.country,
                'credit_rating': pos.entity.credit_rating,
                'notional_amount': pos.notional_amount,
                'market_value': pos.market_value or pos.notional_amount,
                'weight': pos.weight,
                'duration': pos.duration,
                'modified_duration': pos.modified_duration,
                'convexity': pos.convexity
            })
        
        return pd.DataFrame(position_data)
    
    def calculate_portfolio_statistics(self, positions: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic portfolio statistics.
        
        Args:
            positions: Portfolio positions DataFrame
            
        Returns:
            Dictionary of portfolio statistics
        """
        if positions.empty:
            return {}
        
        total_notional = positions['notional_amount'].sum()
        total_market_value = positions['market_value'].sum()
        
        # Calculate weights if not provided
        if 'weight' not in positions.columns or positions['weight'].isna().any():
            positions['weight'] = positions['market_value'] / total_market_value
        
        # Portfolio duration (weighted average)
        portfolio_duration = np.average(
            positions['duration'].fillna(0), 
            weights=positions['weight']
        )
        
        # Portfolio modified duration
        portfolio_mod_duration = np.average(
            positions['modified_duration'].fillna(0), 
            weights=positions['weight']
        )
        
        # Portfolio convexity
        portfolio_convexity = np.average(
            positions['convexity'].fillna(0), 
            weights=positions['weight']
        )
        
        # Concentration metrics
        sorted_weights = positions['weight'].sort_values(ascending=False)
        top_10_concentration = sorted_weights.head(10).sum()
        herfindahl_index = (positions['weight'] ** 2).sum()
        
        # Industry concentration
        industry_weights = positions.groupby('industry_code')['weight'].sum()
        industry_concentration = industry_weights.max() if not industry_weights.empty else 0
        
        # Country concentration
        country_weights = positions.groupby('country')['weight'].sum()
        country_concentration = country_weights.max() if not country_weights.empty else 0
        
        return {
            'total_notional': total_notional,
            'total_market_value': total_market_value,
            'number_of_positions': len(positions),
            'portfolio_duration': portfolio_duration,
            'portfolio_modified_duration': portfolio_mod_duration,
            'portfolio_convexity': portfolio_convexity,
            'top_10_concentration': top_10_concentration,
            'herfindahl_index': herfindahl_index,
            'industry_concentration': industry_concentration,
            'country_concentration': country_concentration
        }
    
    def calculate_portfolio_credit_metrics(self, positions: pd.DataFrame, 
                                         as_of_date: datetime = None) -> Dict[str, float]:
        """
        Calculate portfolio credit risk metrics.
        
        Args:
            positions: Portfolio positions DataFrame
            as_of_date: Calculation date
            
        Returns:
            Dictionary of credit risk metrics
        """
        if positions.empty:
            return {}
        
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Get credit scores for all entities
        entity_scores = {}
        for _, pos in positions.iterrows():
            # Get latest credit score
            latest_score = self.session.query(CreditRiskScore).filter(
                CreditRiskScore.entity_id == pos['entity_id'],
                CreditRiskScore.score_date <= as_of_date
            ).order_by(CreditRiskScore.score_date.desc()).first()
            
            if latest_score:
                entity_scores[pos['entity_id']] = {
                    'pd': latest_score.probability_of_default,
                    'lgd': latest_score.loss_given_default,
                    'ead': latest_score.exposure_at_default,
                    'expected_loss': latest_score.expected_loss
                }
            else:
                # Use default values if no score available
                entity_scores[pos['entity_id']] = {
                    'pd': 0.05,  # 5% default
                    'lgd': 0.45,  # 45% loss rate
                    'ead': pos['notional_amount'],
                    'expected_loss': 0.05 * 0.45 * pos['notional_amount']
                }
        
        # Calculate portfolio metrics
        total_exposure = positions['notional_amount'].sum()
        
        # Weighted average PD
        weighted_pd = sum(
            entity_scores[eid]['pd'] * (pos['notional_amount'] / total_exposure)
            for eid, pos in zip(positions['entity_id'], positions.itertuples())
        )
        
        # Weighted average LGD
        weighted_lgd = sum(
            entity_scores[eid]['lgd'] * (pos['notional_amount'] / total_exposure)
            for eid, pos in zip(positions['entity_id'], positions.itertuples())
        )
        
        # Portfolio expected loss
        portfolio_expected_loss = sum(
            entity_scores[eid]['expected_loss']
            for eid in positions['entity_id']
        )
        
        return {
            'total_exposure': total_exposure,
            'average_pd': weighted_pd,
            'average_lgd': weighted_lgd,
            'expected_loss': portfolio_expected_loss,
            'expected_loss_rate': portfolio_expected_loss / total_exposure
        }
    
    def calculate_portfolio_var(self, positions: pd.DataFrame, 
                              confidence_level: float = 0.95,
                              time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate portfolio Value at Risk (VaR).
        
        Args:
            positions: Portfolio positions DataFrame
            confidence_level: VaR confidence level
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR metrics
        """
        if positions.empty:
            return {}
        
        # Simplified VaR calculation using parametric approach
        # In practice, would use Monte Carlo simulation with factor models
        
        # Get individual position volatilities (simplified)
        position_volatilities = []
        for _, pos in positions.iterrows():
            # Use credit rating to estimate volatility
            rating_vol_map = {
                'AAA': 0.02, 'AA': 0.03, 'A': 0.05, 'BBB': 0.08,
                'BB': 0.12, 'B': 0.18, 'CCC': 0.30, 'D': 0.50
            }
            
            rating = pos.get('credit_rating', 'BBB')
            vol = rating_vol_map.get(rating, 0.08)
            position_volatilities.append(vol * pos['market_value'])
        
        # Portfolio volatility (assuming low correlation for simplicity)
        # In practice, would use correlation matrix
        portfolio_variance = sum(vol ** 2 for vol in position_volatilities)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Adjust for time horizon
        portfolio_volatility *= np.sqrt(time_horizon)
        
        # Calculate VaR
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha)
        var = z_score * portfolio_volatility
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = (stats.norm.pdf(z_score) / alpha) * portfolio_volatility
        
        return {
            'var': var,
            'expected_shortfall': expected_shortfall,
            'portfolio_volatility': portfolio_volatility,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon
        }
    
    def run_stress_test(self, positions: pd.DataFrame, 
                       stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """
        Run stress tests on portfolio.
        
        Args:
            positions: Portfolio positions DataFrame
            stress_scenarios: Dictionary of stress scenarios
            
        Returns:
            Dictionary of stress test results
        """
        results = {}
        
        for scenario_name, factor_shocks in stress_scenarios.items():
            # Apply factor shocks to calculate portfolio impact
            scenario_results = self._apply_stress_scenario(positions, factor_shocks)
            results[scenario_name] = scenario_results
        
        return results
    
    def _apply_stress_scenario(self, positions: pd.DataFrame, 
                             factor_shocks: Dict[str, float]) -> Dict[str, float]:
        """
        Apply stress scenario to portfolio.
        
        Args:
            positions: Portfolio positions DataFrame
            factor_shocks: Factor shock values
            
        Returns:
            Scenario impact results
        """
        # Simplified stress testing
        # In practice, would use factor sensitivities
        
        base_value = positions['market_value'].sum()
        
        # Apply shocks based on position characteristics
        stressed_values = []
        for _, pos in positions.iterrows():
            position_shock = 0
            
            # Interest rate shock
            if 'interest_rate_shock' in factor_shocks:
                rate_shock = factor_shocks['interest_rate_shock']
                duration = pos.get('duration', 0)
                # Duration impact: -Duration * Rate_Change
                position_shock += -duration * rate_shock / 100
            
            # Credit spread shock
            if 'credit_spread_shock' in factor_shocks:
                spread_shock = factor_shocks['credit_spread_shock']
                # Higher rated bonds less sensitive to credit spread shock
                rating_sensitivity = {
                    'AAA': 0.1, 'AA': 0.2, 'A': 0.4, 'BBB': 0.6,
                    'BB': 0.8, 'B': 1.0, 'CCC': 1.2, 'D': 1.5
                }
                rating = pos.get('credit_rating', 'BBB')
                sensitivity = rating_sensitivity.get(rating, 0.6)
                position_shock += -sensitivity * spread_shock / 100
            
            # Equity market shock
            if 'equity_shock' in factor_shocks and pos.get('entity_type') == 'corporate':
                equity_shock = factor_shocks['equity_shock']
                # Corporate bonds correlated with equity
                position_shock += 0.3 * equity_shock / 100
            
            stressed_value = pos['market_value'] * (1 + position_shock)
            stressed_values.append(stressed_value)
        
        stressed_portfolio_value = sum(stressed_values)
        portfolio_pnl = stressed_portfolio_value - base_value
        portfolio_return = portfolio_pnl / base_value
        
        return {
            'base_value': base_value,
            'stressed_value': stressed_portfolio_value,
            'portfolio_pnl': portfolio_pnl,
            'portfolio_return': portfolio_return * 100  # Percentage
        }
    
    def calculate_risk_contribution(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk contribution of each position.
        
        Args:
            positions: Portfolio positions DataFrame
            
        Returns:
            DataFrame with risk contributions
        """
        if positions.empty:
            return pd.DataFrame()
        
        # Simplified risk contribution calculation
        # In practice, would use marginal VaR
        
        total_exposure = positions['market_value'].sum()
        
        risk_contrib = positions.copy()
        
        # Size contribution
        risk_contrib['size_contribution'] = positions['market_value'] / total_exposure
        
        # Volatility-adjusted contribution
        rating_vol_map = {
            'AAA': 0.02, 'AA': 0.03, 'A': 0.05, 'BBB': 0.08,
            'BB': 0.12, 'B': 0.18, 'CCC': 0.30, 'D': 0.50
        }
        
        risk_contrib['volatility'] = risk_contrib['credit_rating'].map(
            lambda x: rating_vol_map.get(x, 0.08)
        )
        
        risk_contrib['risk_weighted_exposure'] = (
            risk_contrib['market_value'] * risk_contrib['volatility']
        )
        
        total_risk_weighted = risk_contrib['risk_weighted_exposure'].sum()
        risk_contrib['risk_contribution'] = (
            risk_contrib['risk_weighted_exposure'] / total_risk_weighted
        )
        
        return risk_contrib[['entity_id', 'entity_name', 'market_value', 
                           'size_contribution', 'risk_contribution']]
    
    def generate_portfolio_report(self, portfolio_id: str, 
                                as_of_date: datetime = None) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio risk report.
        
        Args:
            portfolio_id: Portfolio identifier
            as_of_date: Report date
            
        Returns:
            Complete portfolio risk report
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Get portfolio positions
        positions = self.get_portfolio_positions(portfolio_id, as_of_date)
        
        if positions.empty:
            return {'error': 'No positions found for portfolio'}
        
        # Calculate all metrics
        basic_stats = self.calculate_portfolio_statistics(positions)
        credit_metrics = self.calculate_portfolio_credit_metrics(positions, as_of_date)
        var_metrics = self.calculate_portfolio_var(positions)
        
        # Define stress scenarios
        stress_scenarios = {
            'Interest Rate Shock (+200bp)': {'interest_rate_shock': 200},
            'Credit Spread Widening (+100bp)': {'credit_spread_shock': 100},
            'Market Stress': {
                'interest_rate_shock': 100,
                'credit_spread_shock': 150,
                'equity_shock': -20
            },
            '2008 Financial Crisis': {
                'interest_rate_shock': -300,
                'credit_spread_shock': 400,
                'equity_shock': -40
            }
        }
        
        stress_results = self.run_stress_test(positions, stress_scenarios)
        risk_contributions = self.calculate_risk_contribution(positions)
        
        report = {
            'portfolio_id': portfolio_id,
            'report_date': as_of_date,
            'basic_statistics': basic_stats,
            'credit_metrics': credit_metrics,
            'var_metrics': var_metrics,
            'stress_test_results': stress_results,
            'top_risk_contributors': risk_contributions.head(10).to_dict('records'),
            'positions_summary': {
                'total_positions': len(positions),
                'industries': positions['industry_code'].nunique(),
                'countries': positions['country'].nunique(),
                'rating_distribution': positions['credit_rating'].value_counts().to_dict()
            }
        }
        
        logger.info("Portfolio report generated",
                   portfolio_id=portfolio_id,
                   positions_count=len(positions),
                   total_exposure=basic_stats.get('total_exposure', 0))
        
        return report 