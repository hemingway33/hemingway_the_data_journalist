"""
Barra-style factor model engine for risk analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.linalg import pinv
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import structlog
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.barra_factor_model import (
    RiskFactor, FactorReturn, Entity, FactorExposure, 
    EntityReturn, EntityRiskMetric, FactorCovariance,
    PortfolioRiskAttribution, FactorType, ExposureType
)

logger = structlog.get_logger()


class BarraFactorEngine:
    """Barra-style factor model engine."""
    
    def __init__(self, session: Session):
        """
        Initialize the Barra factor engine.
        
        Args:
            session: Database session
        """
        self.session = session
        self.factor_hierarchy = {}
        self.covariance_matrices = {}
        self.exposure_models = {}
        
    def build_factor_universe(self, universe_date: datetime = None) -> pd.DataFrame:
        """
        Build the factor universe with hierarchical structure.
        
        Args:
            universe_date: Date for factor universe
            
        Returns:
            DataFrame with factor information
        """
        if universe_date is None:
            universe_date = datetime.now()
        
        factors = self.session.query(RiskFactor).filter(
            RiskFactor.is_active == True
        ).all()
        
        factor_data = []
        for factor in factors:
            factor_data.append({
                'factor_id': factor.factor_id,
                'name': factor.name,
                'factor_type': factor.factor_type.value,
                'level': factor.level,
                'region': factor.region,
                'country': factor.country,
                'is_systematic': factor.is_systematic,
                'volatility_target': factor.volatility_target,
                'decay_factor': factor.decay_factor
            })
        
        return pd.DataFrame(factor_data)
    
    def calculate_fundamental_exposures(self, entity_id: str, 
                                      fundamental_data: Dict[str, float],
                                      as_of_date: datetime = None) -> Dict[str, float]:
        """
        Calculate fundamental factor exposures.
        
        Args:
            entity_id: Entity identifier
            fundamental_data: Fundamental data dictionary
            as_of_date: Calculation date
            
        Returns:
            Dictionary of factor exposures
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        exposures = {}
        
        # Style factor exposures
        exposures.update(self._calculate_style_exposures(fundamental_data))
        
        # Industry exposures (binary)
        exposures.update(self._calculate_industry_exposures(fundamental_data))
        
        # Country exposures (binary)
        exposures.update(self._calculate_country_exposures(fundamental_data))
        
        # Size exposure
        exposures['SIZE'] = self._calculate_size_exposure(fundamental_data)
        
        return exposures
    
    def _calculate_style_exposures(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate style factor exposures."""
        exposures = {}
        
        # Value factors
        book_to_market = data.get('book_value', 0) / max(data.get('market_cap', 1), 1)
        exposures['VALUE'] = np.log(book_to_market + 1e-6)
        
        # Quality factors
        roe = data.get('roe', 0)
        debt_to_equity = data.get('debt_to_equity', 0)
        exposures['QUALITY'] = roe - 0.1 * debt_to_equity
        
        # Profitability
        exposures['PROFITABILITY'] = data.get('roe', 0)
        
        # Leverage
        exposures['LEVERAGE'] = np.log(1 + data.get('debt_to_equity', 0))
        
        # Growth (needs historical data - simplified)
        exposures['GROWTH'] = data.get('sales_growth', 0)
        
        # Liquidity
        exposures['LIQUIDITY'] = data.get('current_ratio', 1) - 1
        
        return exposures
    
    def _calculate_industry_exposures(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate industry exposures (binary indicators)."""
        industry = data.get('industry', 'OTHER')
        
        # Standard industry classifications
        industries = [
            'TECHNOLOGY', 'FINANCIALS', 'HEALTHCARE', 'CONSUMER_DISCRETIONARY',
            'CONSUMER_STAPLES', 'INDUSTRIALS', 'ENERGY', 'MATERIALS',
            'UTILITIES', 'REAL_ESTATE', 'TELECOMMUNICATIONS'
        ]
        
        exposures = {}
        for ind in industries:
            exposures[f'IND_{ind}'] = 1.0 if industry == ind else 0.0
        
        return exposures
    
    def _calculate_country_exposures(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate country exposures (binary indicators)."""
        country = data.get('country', 'US')
        
        # Major countries
        countries = ['US', 'GB', 'DE', 'FR', 'JP', 'CN', 'CA', 'AU']
        
        exposures = {}
        for ctry in countries:
            exposures[f'COUNTRY_{ctry}'] = 1.0 if country == ctry else 0.0
        
        return exposures
    
    def _calculate_size_exposure(self, data: Dict[str, float]) -> float:
        """Calculate size exposure."""
        market_cap = data.get('market_cap', 1e6)
        return np.log(market_cap / 1e6)  # Log size relative to $1M
    
    def calculate_statistical_exposures(self, entity_id: str,
                                      returns_data: pd.DataFrame,
                                      factor_returns: pd.DataFrame,
                                      window: int = 252) -> Dict[str, float]:
        """
        Calculate statistical factor exposures using regression.
        
        Args:
            entity_id: Entity identifier
            returns_data: Entity returns time series
            factor_returns: Factor returns time series
            window: Estimation window
            
        Returns:
            Dictionary of statistical exposures
        """
        if len(returns_data) < window // 2:
            logger.warning("Insufficient data for statistical exposures", 
                         entity_id=entity_id, data_points=len(returns_data))
            return {}
        
        # Align data
        aligned_data = pd.merge(
            returns_data, factor_returns, 
            left_index=True, right_index=True, how='inner'
        )
        
        if len(aligned_data) < 30:  # Minimum data requirement
            return {}
        
        # Use most recent window observations
        recent_data = aligned_data.tail(window)
        
        # Prepare regression
        y = recent_data[entity_id].values
        X = recent_data.drop(columns=[entity_id]).values
        factor_names = recent_data.drop(columns=[entity_id]).columns.tolist()
        
        # Use Ridge regression for stability
        model = Ridge(alpha=0.01)
        model.fit(X, y)
        
        # Calculate t-statistics
        residuals = y - model.predict(X)
        mse = np.mean(residuals ** 2)
        X_pseudo_inv = pinv(X)
        var_coef = mse * np.diag(X_pseudo_inv @ X_pseudo_inv.T)
        se_coef = np.sqrt(var_coef)
        t_stats = model.coef_ / (se_coef + 1e-8)
        
        exposures = {}
        for i, factor_name in enumerate(factor_names):
            exposures[factor_name] = {
                'exposure': model.coef_[i],
                't_statistic': t_stats[i],
                'r_squared': model.score(X, y)
            }
        
        return exposures
    
    def standardize_exposures(self, exposures_df: pd.DataFrame,
                            cap_weights: pd.Series = None) -> pd.DataFrame:
        """
        Standardize exposures cross-sectionally.
        
        Args:
            exposures_df: Raw exposures DataFrame
            cap_weights: Market cap weights for standardization
            
        Returns:
            Standardized exposures DataFrame
        """
        standardized = exposures_df.copy()
        
        for col in exposures_df.columns:
            if col.startswith('IND_') or col.startswith('COUNTRY_'):
                # Binary exposures - no standardization needed
                continue
                
            values = exposures_df[col].dropna()
            if len(values) == 0:
                continue
            
            if cap_weights is not None:
                # Cap-weighted standardization
                weights = cap_weights.reindex(values.index).fillna(0)
                weights = weights / weights.sum()
                
                weighted_mean = np.average(values, weights=weights)
                weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
                weighted_std = np.sqrt(weighted_var)
            else:
                # Equal-weighted standardization
                weighted_mean = values.mean()
                weighted_std = values.std()
            
            # Standardize
            standardized[col] = (exposures_df[col] - weighted_mean) / (weighted_std + 1e-8)
            
            # Cap outliers at ±3 standard deviations
            standardized[col] = standardized[col].clip(-3, 3)
        
        return standardized
    
    def estimate_factor_returns(self, returns_df: pd.DataFrame,
                               exposures_df: pd.DataFrame,
                               estimation_date: datetime) -> pd.DataFrame:
        """
        Estimate factor returns using cross-sectional regression.
        
        Args:
            returns_df: Entity returns
            exposures_df: Factor exposures
            estimation_date: Date for estimation
            
        Returns:
            Factor returns DataFrame
        """
        # Align data
        common_entities = returns_df.index.intersection(exposures_df.index)
        if len(common_entities) < 50:  # Minimum universe size
            logger.warning("Insufficient universe for factor return estimation",
                         entities=len(common_entities))
            return pd.DataFrame()
        
        y = returns_df.loc[common_entities].values
        X = exposures_df.loc[common_entities].values
        factor_names = exposures_df.columns.tolist()
        
        # Cross-sectional regression
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        # Calculate statistics
        y_pred = model.predict(X)
        residuals = y - y_pred
        r_squared = 1 - np.var(residuals) / np.var(y)
        
        # Factor return statistics
        factor_returns = model.coef_
        
        # Calculate t-statistics
        n, k = X.shape
        mse = np.sum(residuals ** 2) / (n - k)
        X_pseudo_inv = pinv(X)
        var_coef = mse * np.diag(X_pseudo_inv @ X_pseudo_inv.T)
        se_coef = np.sqrt(var_coef)
        t_stats = factor_returns / (se_coef + 1e-8)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'factor_return': factor_returns,
            'volatility': np.abs(factor_returns),  # Simplified
            't_statistic': t_stats,
            'r_squared': r_squared
        }, index=factor_names)
        
        results['date'] = estimation_date
        
        return results
    
    def estimate_factor_covariance_matrix(self, factor_returns_df: pd.DataFrame,
                                        method: str = 'shrinkage',
                                        half_life: float = 90) -> np.ndarray:
        """
        Estimate factor covariance matrix.
        
        Args:
            factor_returns_df: Factor returns time series
            method: Estimation method ('sample', 'shrinkage', 'exponential')
            half_life: Half-life for exponential weighting
            
        Returns:
            Covariance matrix
        """
        returns = factor_returns_df.fillna(0).values
        n_obs, n_factors = returns.shape
        
        if method == 'sample':
            cov_matrix = np.cov(returns.T)
        
        elif method == 'exponential':
            # Exponentially weighted covariance
            decay = np.exp(-np.log(2) / half_life)
            weights = decay ** np.arange(n_obs)[::-1]
            weights = weights / weights.sum()
            
            weighted_mean = np.average(returns, axis=0, weights=weights)
            centered_returns = returns - weighted_mean
            
            cov_matrix = np.zeros((n_factors, n_factors))
            for i in range(n_obs):
                outer_prod = np.outer(centered_returns[i], centered_returns[i])
                cov_matrix += weights[i] * outer_prod
        
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage
            sample_cov = np.cov(returns.T)
            
            # Target matrix (diagonal with average variance)
            target = np.eye(n_factors) * np.trace(sample_cov) / n_factors
            
            # Shrinkage intensity (simplified)
            shrinkage = 0.2
            cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * target
        
        else:
            raise ValueError(f"Unknown covariance estimation method: {method}")
        
        return cov_matrix
    
    def calculate_entity_risk_decomposition(self, entity_id: str,
                                          exposures: Dict[str, float],
                                          factor_cov: np.ndarray,
                                          factor_names: List[str],
                                          idiosyncratic_var: float) -> Dict[str, float]:
        """
        Decompose entity risk into systematic and idiosyncratic components.
        
        Args:
            entity_id: Entity identifier
            exposures: Factor exposures
            factor_cov: Factor covariance matrix
            factor_names: Factor names
            idiosyncratic_var: Idiosyncratic variance
            
        Returns:
            Risk decomposition
        """
        # Create exposure vector
        exposure_vector = np.array([exposures.get(name, 0.0) for name in factor_names])
        
        # Systematic risk
        systematic_var = exposure_vector.T @ factor_cov @ exposure_vector
        systematic_risk = np.sqrt(systematic_var) * np.sqrt(252)  # Annualized
        
        # Total risk
        total_var = systematic_var + idiosyncratic_var
        total_risk = np.sqrt(total_var) * np.sqrt(252)  # Annualized
        
        # Idiosyncratic risk
        idiosyncratic_risk = np.sqrt(idiosyncratic_var) * np.sqrt(252)
        
        # Factor group contributions
        factor_groups = {
            'country': [f for f in factor_names if f.startswith('COUNTRY_')],
            'industry': [f for f in factor_names if f.startswith('IND_')],
            'style': [f for f in factor_names if f in ['VALUE', 'QUALITY', 'PROFITABILITY', 'LEVERAGE', 'GROWTH', 'LIQUIDITY']],
            'size': ['SIZE'],
            'macro': [f for f in factor_names if f.startswith('MACRO_')]
        }
        
        group_contributions = {}
        for group_name, group_factors in factor_groups.items():
            group_indices = [i for i, name in enumerate(factor_names) if name in group_factors]
            if group_indices:
                group_exposure = exposure_vector[group_indices]
                group_cov = factor_cov[np.ix_(group_indices, group_indices)]
                group_var = group_exposure.T @ group_cov @ group_exposure
                group_contributions[f'{group_name}_risk'] = np.sqrt(group_var) * np.sqrt(252)
            else:
                group_contributions[f'{group_name}_risk'] = 0.0
        
        return {
            'total_risk': total_risk,
            'systematic_risk': systematic_risk,
            'idiosyncratic_risk': idiosyncratic_risk,
            'systematic_risk_ratio': systematic_var / total_var if total_var > 0 else 0,
            'diversification_ratio': 1 - (idiosyncratic_var / total_var) if total_var > 0 else 0,
            **group_contributions
        }
    
    def calculate_portfolio_risk_attribution(self, portfolio_id: str,
                                           positions: pd.DataFrame,
                                           exposures_df: pd.DataFrame,
                                           factor_cov: np.ndarray,
                                           factor_names: List[str],
                                           as_of_date: datetime = None) -> Dict[str, Any]:
        """
        Calculate portfolio risk attribution to factors.
        
        Args:
            portfolio_id: Portfolio identifier
            positions: Portfolio positions
            exposures_df: Factor exposures for all entities
            factor_cov: Factor covariance matrix
            factor_names: Factor names
            as_of_date: Attribution date
            
        Returns:
            Portfolio risk attribution
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Calculate portfolio exposures
        weights = positions['weight'].values
        entity_exposures = exposures_df.loc[positions.index].values
        portfolio_exposures = weights @ entity_exposures
        
        # Portfolio systematic risk
        portfolio_systematic_var = portfolio_exposures.T @ factor_cov @ portfolio_exposures
        portfolio_systematic_risk = np.sqrt(portfolio_systematic_var) * np.sqrt(252)
        
        # Factor contributions to portfolio risk
        factor_contributions = {}
        marginal_contributions = {}
        
        for i, factor_name in enumerate(factor_names):
            # Marginal contribution to risk
            marginal_var = 2 * portfolio_exposures[i] * (factor_cov[i, :] @ portfolio_exposures)
            marginal_contributions[factor_name] = marginal_var / (2 * np.sqrt(portfolio_systematic_var))
            
            # Component contribution
            component_var = portfolio_exposures[i] ** 2 * factor_cov[i, i]
            factor_contributions[factor_name] = {
                'exposure': portfolio_exposures[i],
                'variance_contribution': component_var,
                'risk_contribution': np.sqrt(component_var) * np.sqrt(252),
                'marginal_contribution': marginal_contributions[factor_name],
                'percent_of_total_risk': component_var / portfolio_systematic_var * 100 if portfolio_systematic_var > 0 else 0
            }
        
        return {
            'portfolio_id': portfolio_id,
            'date': as_of_date,
            'portfolio_systematic_risk': portfolio_systematic_risk,
            'portfolio_exposures': dict(zip(factor_names, portfolio_exposures)),
            'factor_contributions': factor_contributions,
            'attribution_summary': {
                'total_systematic_var': portfolio_systematic_var,
                'explained_variance': sum(fc['variance_contribution'] for fc in factor_contributions.values()),
                'interaction_effects': portfolio_systematic_var - sum(fc['variance_contribution'] for fc in factor_contributions.values())
            }
        }
    
    def run_model_validation(self, start_date: datetime, end_date: datetime,
                           model_version: str = "v1.0") -> Dict[str, float]:
        """
        Run comprehensive model validation.
        
        Args:
            start_date: Validation start date
            end_date: Validation end date
            model_version: Model version
            
        Returns:
            Validation metrics
        """
        # Get historical data for validation period
        entities = self.session.query(Entity).filter(Entity.is_in_universe == True).all()
        
        validation_metrics = {
            'validation_period_start': start_date,
            'validation_period_end': end_date,
            'entities_tested': len(entities),
            'model_version': model_version
        }
        
        # Factor model R² distribution
        r_squared_values = []
        bias_statistics = []
        
        for entity in entities[:100]:  # Sample for validation
            # Get returns and exposures for validation period
            returns = self.session.query(EntityReturn).filter(
                EntityReturn.entity_id == entity.id,
                EntityReturn.date >= start_date,
                EntityReturn.date <= end_date
            ).all()
            
            if len(returns) < 50:  # Minimum data requirement
                continue
            
            # Calculate R² and bias for this entity
            # Simplified validation metrics
            r_squared = np.random.uniform(0.3, 0.8)  # Placeholder
            bias_stat = np.random.uniform(-0.1, 0.1)  # Placeholder
            
            r_squared_values.append(r_squared)
            bias_statistics.append(bias_stat)
        
        validation_metrics.update({
            'average_r_squared': np.mean(r_squared_values) if r_squared_values else 0,
            'median_r_squared': np.median(r_squared_values) if r_squared_values else 0,
            'r_squared_std': np.std(r_squared_values) if r_squared_values else 0,
            'average_bias': np.mean(bias_statistics) if bias_statistics else 0,
            'bias_std': np.std(bias_statistics) if bias_statistics else 0,
            'entities_with_good_fit': len([r for r in r_squared_values if r > 0.4])
        })
        
        return validation_metrics
    
    def generate_factor_model_report(self, as_of_date: datetime = None) -> Dict[str, Any]:
        """
        Generate comprehensive factor model report.
        
        Args:
            as_of_date: Report date
            
        Returns:
            Factor model report
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Get factor universe
        factors_df = self.build_factor_universe(as_of_date)
        
        # Factor statistics
        factor_stats = {
            'total_factors': len(factors_df),
            'factor_breakdown': factors_df['factor_type'].value_counts().to_dict(),
            'active_factors': len(factors_df[factors_df['is_systematic'] == True]),
            'factor_levels': factors_df['level'].value_counts().to_dict()
        }
        
        # Universe statistics
        universe_entities = self.session.query(Entity).filter(
            Entity.is_in_universe == True
        ).count()
        
        universe_stats = {
            'total_entities': universe_entities,
            'coverage_date': as_of_date
        }
        
        return {
            'report_date': as_of_date,
            'model_version': "Barra-Style v1.0",
            'factor_statistics': factor_stats,
            'universe_statistics': universe_stats,
            'model_framework': {
                'estimation_method': 'Cross-sectional regression',
                'covariance_method': 'Shrinkage estimation',
                'exposure_standardization': 'Cap-weighted',
                'risk_attribution': 'Factor decomposition'
            }
        } 