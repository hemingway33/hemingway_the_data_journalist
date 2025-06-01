"""
Credit scoring engine for individual credit risk assessment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import structlog

# Optional XGBoost import
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available - using alternative models")
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.risk_factors import (
    RiskFactor, RiskFactorValue, CreditEntity, CreditRiskScore
)

logger = structlog.get_logger()


class CreditScoringEngine:
    """Credit scoring engine for individual risk assessment."""
    
    def __init__(self, session: Session):
        """
        Initialize credit scoring engine.
        
        Args:
            session: Database session
        """
        self.session = session
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def get_risk_factors_snapshot(self, as_of_date: datetime = None) -> Dict[str, float]:
        """
        Get current snapshot of all risk factors.
        
        Args:
            as_of_date: Date for risk factor values (default: latest)
            
        Returns:
            Dictionary of risk factor values
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Get all active risk factors
        risk_factors = self.session.query(RiskFactor).filter(
            RiskFactor.is_active == True
        ).all()
        
        snapshot = {}
        
        for factor in risk_factors:
            # Get latest value before or on as_of_date
            latest_value = self.session.query(RiskFactorValue).filter(
                RiskFactorValue.risk_factor_id == factor.id,
                RiskFactorValue.date <= as_of_date
            ).order_by(RiskFactorValue.date.desc()).first()
            
            if latest_value:
                snapshot[factor.factor_id] = latest_value.value
            else:
                # Use mean value if no data available
                avg_value = self.session.query(RiskFactorValue).filter(
                    RiskFactorValue.risk_factor_id == factor.id
                ).with_entities(
                    RiskFactorValue.value
                ).subquery()
                
                mean_val = self.session.query(
                    avg_value.c.value
                ).scalar()
                
                if mean_val:
                    snapshot[factor.factor_id] = mean_val
                else:
                    snapshot[factor.factor_id] = 0.0
        
        return snapshot
    
    def prepare_features(self, entity_features: Dict[str, Any], 
                        risk_factors: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare feature matrix for scoring.
        
        Args:
            entity_features: Entity-specific features
            risk_factors: Risk factor snapshot
            
        Returns:
            Feature DataFrame
        """
        # Base features from entity
        features = {
            # Entity demographics
            'entity_age_years': entity_features.get('age_years', 0),
            'entity_size_log': np.log1p(entity_features.get('total_assets', 1)),
            'entity_leverage': entity_features.get('debt_to_equity', 0),
            'entity_liquidity': entity_features.get('current_ratio', 1),
            'entity_profitability': entity_features.get('roe', 0),
            'entity_efficiency': entity_features.get('asset_turnover', 0),
            
            # Industry and geography
            'industry_risk_score': entity_features.get('industry_risk_score', 0.5),
            'country_risk_score': entity_features.get('country_risk_score', 0.5),
            
            # Credit history
            'payment_history_score': entity_features.get('payment_history_score', 0.5),
            'credit_utilization': entity_features.get('credit_utilization', 0),
            'number_of_accounts': entity_features.get('number_of_accounts', 0),
            'recent_inquiries': entity_features.get('recent_inquiries', 0),
        }
        
        # Add risk factors
        features.update(risk_factors)
        
        # Create interaction features
        economic_conditions = (
            risk_factors.get('FRED_GDP_GROWTH', 0) * 0.3 +
            risk_factors.get('FRED_UNEMPLOYMENT_RATE', 5) * -0.2 +
            risk_factors.get('FRED_INFLATION_RATE', 2) * -0.1
        )
        features['economic_conditions_composite'] = economic_conditions
        
        market_stress = (
            risk_factors.get('MARKET_VOLATILITY_INDEX', 20) * 0.4 +
            risk_factors.get('FRED_CREDIT_SPREAD', 1) * 0.6
        )
        features['market_stress_composite'] = market_stress
        
        # Entity-market interaction
        features['entity_market_interaction'] = (
            features['entity_leverage'] * market_stress
        )
        
        # Convert to DataFrame
        return pd.DataFrame([features])
    
    def calculate_probability_of_default(self, features: pd.DataFrame, 
                                       model_type: str = 'xgboost') -> Dict[str, float]:
        """
        Calculate probability of default using specified model.
        
        Args:
            features: Feature matrix
            model_type: Model type ('logistic', 'xgboost', 'random_forest')
            
        Returns:
            Dictionary with PD and confidence intervals
        """
        if model_type not in self.models:
            # Train model if not available (would typically be pre-trained)
            self._train_default_model(model_type)
        
        model = self.models.get(model_type)
        scaler = self.scalers.get(model_type)
        
        if not model or not scaler:
            logger.error("Model not available", model_type=model_type)
            return {'probability_of_default': 0.05}  # Default to 5%
        
        try:
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                pd_prob = model.predict_proba(features_scaled)[0][1]  # Class 1 probability
            else:
                pd_prob = model.predict(features_scaled)[0]
            
            # Ensure probability is in valid range
            pd_prob = np.clip(pd_prob, 0.0001, 0.9999)
            
            # Calculate confidence intervals (simplified)
            confidence_width = 0.05  # ±5% confidence interval
            
            result = {
                'probability_of_default': float(pd_prob),
                'confidence_interval_lower': max(0.0001, pd_prob - confidence_width),
                'confidence_interval_upper': min(0.9999, pd_prob + confidence_width),
                'model_type': model_type
            }
            
            return result
            
        except Exception as e:
            logger.error("Failed to calculate PD", model_type=model_type, error=str(e))
            return {'probability_of_default': 0.05}
    
    def calculate_loss_given_default(self, entity_features: Dict[str, Any], 
                                   collateral_value: float = 0) -> float:
        """
        Calculate Loss Given Default (LGD).
        
        Args:
            entity_features: Entity-specific features
            collateral_value: Value of collateral
            
        Returns:
            LGD percentage (0-1)
        """
        # Base LGD based on entity type and seniority
        base_lgd = {
            'corporate': 0.45,
            'individual': 0.75,
            'sovereign': 0.20
        }.get(entity_features.get('entity_type', 'corporate'), 0.45)
        
        # Adjust for collateral
        exposure = entity_features.get('exposure_amount', 1000000)
        collateral_ratio = collateral_value / exposure if exposure > 0 else 0
        collateral_adjustment = -0.5 * collateral_ratio  # Reduce LGD with more collateral
        
        # Adjust for seniority
        seniority_adjustment = {
            'senior_secured': -0.20,
            'senior_unsecured': 0.00,
            'subordinated': 0.15,
            'equity': 0.30
        }.get(entity_features.get('seniority', 'senior_unsecured'), 0.00)
        
        # Adjust for industry
        industry_adjustment = {
            'utilities': -0.10,
            'real_estate': -0.05,
            'technology': 0.05,
            'retail': 0.10
        }.get(entity_features.get('industry', ''), 0.00)
        
        lgd = base_lgd + collateral_adjustment + seniority_adjustment + industry_adjustment
        
        return np.clip(lgd, 0.05, 0.95)  # LGD between 5% and 95%
    
    def calculate_exposure_at_default(self, entity_features: Dict[str, Any]) -> float:
        """
        Calculate Exposure at Default (EAD).
        
        Args:
            entity_features: Entity-specific features
            
        Returns:
            EAD amount
        """
        current_exposure = entity_features.get('current_exposure', 0)
        credit_limit = entity_features.get('credit_limit', current_exposure)
        utilization_rate = entity_features.get('credit_utilization', 0.5)
        
        # For revolving credit, account for potential drawdown
        if entity_features.get('facility_type') == 'revolving':
            # Assume higher utilization at default
            stress_utilization = min(1.0, utilization_rate * 1.5)
            ead = credit_limit * stress_utilization
        else:
            # Term loans - current exposure plus accrued interest
            ead = current_exposure * 1.02  # 2% for accrued interest/fees
        
        return ead
    
    def calculate_expected_loss(self, pd: float, lgd: float, ead: float) -> float:
        """
        Calculate Expected Loss.
        
        Args:
            pd: Probability of Default
            lgd: Loss Given Default
            ead: Exposure at Default
            
        Returns:
            Expected Loss amount
        """
        return pd * lgd * ead
    
    def score_entity(self, entity_id: str, entity_features: Dict[str, Any],
                    as_of_date: datetime = None) -> Dict[str, Any]:
        """
        Complete credit scoring for an entity.
        
        Args:
            entity_id: Entity identifier
            entity_features: Entity-specific features
            as_of_date: Scoring date
            
        Returns:
            Complete risk assessment
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        try:
            # Get risk factors snapshot
            risk_factors = self.get_risk_factors_snapshot(as_of_date)
            
            # Prepare features
            features = self.prepare_features(entity_features, risk_factors)
            
            # Calculate PD
            pd_result = self.calculate_probability_of_default(features, 'xgboost')
            
            # Calculate LGD
            lgd = self.calculate_loss_given_default(entity_features)
            
            # Calculate EAD
            ead = self.calculate_exposure_at_default(entity_features)
            
            # Calculate Expected Loss
            expected_loss = self.calculate_expected_loss(
                pd_result['probability_of_default'], lgd, ead
            )
            
            # Determine risk rating
            risk_rating = self._determine_risk_rating(pd_result['probability_of_default'])
            
            result = {
                'entity_id': entity_id,
                'score_date': as_of_date,
                'probability_of_default': pd_result['probability_of_default'],
                'confidence_interval_lower': pd_result.get('confidence_interval_lower'),
                'confidence_interval_upper': pd_result.get('confidence_interval_upper'),
                'loss_given_default': lgd,
                'exposure_at_default': ead,
                'expected_loss': expected_loss,
                'risk_rating': risk_rating,
                'model_version': 'v1.0',
                'model_type': pd_result.get('model_type', 'xgboost'),
                'risk_factors_snapshot': risk_factors
            }
            
            logger.info("Credit scoring completed",
                       entity_id=entity_id,
                       pd=pd_result['probability_of_default'],
                       risk_rating=risk_rating)
            
            return result
            
        except Exception as e:
            logger.error("Credit scoring failed",
                        entity_id=entity_id,
                        error=str(e))
            raise
    
    def _determine_risk_rating(self, pd: float) -> str:
        """
        Determine risk rating based on PD.
        
        Args:
            pd: Probability of Default
            
        Returns:
            Risk rating
        """
        if pd < 0.001:
            return 'AAA'
        elif pd < 0.002:
            return 'AA'
        elif pd < 0.005:
            return 'A'
        elif pd < 0.01:
            return 'BBB'
        elif pd < 0.02:
            return 'BB'
        elif pd < 0.05:
            return 'B'
        elif pd < 0.10:
            return 'CCC'
        else:
            return 'D'
    
    def _train_default_model(self, model_type: str):
        """
        Train default prediction model (placeholder implementation).
        In production, this would use historical data.
        
        Args:
            model_type: Type of model to train
        """
        logger.info("Training default model", model_type=model_type)
        
        # Generate synthetic training data for demonstration
        np.random.seed(42)
        n_samples = 10000
        
        # Features
        X = np.random.randn(n_samples, 20)
        
        # Target (default events)
        # Create realistic default rate (~3%)
        y = (np.random.randn(n_samples) + X[:, 0] * 0.5 + X[:, 1] * -0.3) > 1.5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42)
        elif model_type == 'xgboost':
            if HAS_XGBOOST:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                logger.warning("XGBoost requested but not available, using Random Forest")
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.models[model_type] = model
        self.scalers[model_type] = scaler
        
        logger.info("Model training completed", model_type=model_type) 