"""
Digital Twin Environment for Loan Portfolio Management

This module implements the core digital twin environment that serves as both:
1. A real-time simulation of loan portfolio operations
2. A reinforcement learning environment for training agents

The environment integrates loan business management, portfolio optimization,
stress testing, and performance prediction capabilities.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoanStatus(Enum):
    """Loan status enumeration"""
    CURRENT = "current"
    DELINQUENT_30 = "delinquent_30"
    DELINQUENT_60 = "delinquent_60"
    DELINQUENT_90 = "delinquent_90"
    DEFAULT = "default"
    PREPAID = "prepaid"


class LoanType(Enum):
    """Loan product types"""
    CONSUMER = "consumer"
    AUTO = "auto"
    MORTGAGE = "mortgage"
    BUSINESS = "business"
    CREDIT_CARD = "credit_card"


@dataclass
class Loan:
    """Individual loan entity"""
    loan_id: str
    loan_type: LoanType
    principal: float
    interest_rate: float
    term_months: int
    origination_date: datetime
    borrower_score: float
    monthly_payment: float
    current_balance: float
    status: LoanStatus = LoanStatus.CURRENT
    days_past_due: int = 0
    probability_default: float = 0.0
    ltv_ratio: float = 0.0
    dti_ratio: float = 0.0


@dataclass
class Portfolio:
    """Loan portfolio representation"""
    loans: List[Loan] = field(default_factory=list)
    total_value: float = 0.0
    total_exposure: float = 0.0
    expected_loss: float = 0.0
    var_95: float = 0.0
    return_on_equity: float = 0.0
    net_interest_margin: float = 0.0
    charge_off_rate: float = 0.0
    delinquency_rate: float = 0.0


@dataclass
class MarketConditions:
    """Market and economic conditions"""
    base_interest_rate: float = 0.05
    unemployment_rate: float = 0.04
    gdp_growth: float = 0.02
    inflation_rate: float = 0.02
    housing_price_index: float = 100.0
    credit_spread: float = 0.01
    volatility_index: float = 15.0


@dataclass
class BusinessPolicies:
    """Business policies and constraints"""
    max_ltv: float = 0.80
    min_credit_score: float = 600
    max_dti: float = 0.43
    max_concentration_by_type: float = 0.40
    target_roe: float = 0.15
    max_var_limit: float = 0.05
    min_capital_ratio: float = 0.08


class LoanPortfolioTwinEnv(gym.Env):
    """
    Digital Twin Environment for Loan Portfolio Management
    
    This environment simulates a loan portfolio and allows agents to make
    decisions about credit policies, pricing, and portfolio composition.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        initial_portfolio_size: int = 1000,
        max_portfolio_size: int = 10000,
        simulation_days: int = 365,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.initial_portfolio_size = initial_portfolio_size
        self.max_portfolio_size = max_portfolio_size
        self.simulation_days = simulation_days
        self.render_mode = render_mode
        
        # Initialize environment state
        self.current_day = 0
        self.portfolio = Portfolio()
        self.market_conditions = MarketConditions()
        self.policies = BusinessPolicies()
        
        # Performance tracking
        self.performance_history = []
        self.decision_history = []
        
        # Define action space
        # Actions: [credit_policy_adjustment, pricing_adjustment, portfolio_rebalancing]
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.02, -0.1]),  # Max 10% policy change, 2% rate change, 10% rebalancing
            high=np.array([0.1, 0.02, 0.1]),
            dtype=np.float32
        )
        
        # Define observation space
        # Observations: portfolio metrics, market conditions, policy settings
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,    # portfolio_value
                0.0,    # expected_loss_rate
                0.0,    # delinquency_rate
                0.0,    # roe
                0.0,    # var_95
                0.0,    # concentration_risk
                0.01,   # base_interest_rate
                0.0,    # unemployment_rate
                -0.1,   # gdp_growth
                0.0,    # inflation_rate
                50.0,   # housing_price_index
                0.0,    # credit_spread
                5.0,    # volatility_index
            ]),
            high=np.array([
                1e9,    # portfolio_value
                0.5,    # expected_loss_rate
                0.3,    # delinquency_rate
                1.0,    # roe
                0.2,    # var_95
                1.0,    # concentration_risk
                0.2,    # base_interest_rate
                0.2,    # unemployment_rate
                0.1,    # gdp_growth
                0.1,    # inflation_rate
                200.0,  # housing_price_index
                0.1,    # credit_spread
                50.0,   # volatility_index
            ]),
            dtype=np.float32
        )
        
        # Initialize synthetic loan generator
        self._initialize_loan_generator()
    
    def _initialize_loan_generator(self):
        """Initialize synthetic loan data generation"""
        np.random.seed(42)  # For reproducible results
        
    def _generate_synthetic_loan(self, loan_type: LoanType = None) -> Loan:
        """Generate a synthetic loan with realistic characteristics"""
        if loan_type is None:
            loan_type = np.random.choice(list(LoanType))
        
        # Loan characteristics based on type
        type_params = {
            LoanType.CONSUMER: {
                'principal_range': (5000, 50000),
                'rate_range': (0.08, 0.20),
                'term_range': (24, 60),
                'score_range': (600, 800)
            },
            LoanType.AUTO: {
                'principal_range': (15000, 80000),
                'rate_range': (0.04, 0.12),
                'term_range': (36, 84),
                'score_range': (650, 850)
            },
            LoanType.MORTGAGE: {
                'principal_range': (200000, 800000),
                'rate_range': (0.03, 0.08),
                'term_range': (180, 360),
                'score_range': (700, 850)
            },
            LoanType.BUSINESS: {
                'principal_range': (50000, 500000),
                'rate_range': (0.06, 0.15),
                'term_range': (12, 120),
                'score_range': (650, 800)
            },
            LoanType.CREDIT_CARD: {
                'principal_range': (1000, 25000),
                'rate_range': (0.15, 0.25),
                'term_range': (12, 13),  # Revolving (12 to 12 inclusive)
                'score_range': (600, 800)
            }
        }
        
        params = type_params[loan_type]
        
        # Generate loan attributes
        principal = np.random.uniform(*params['principal_range'])
        interest_rate = np.random.uniform(*params['rate_range'])
        term_months = np.random.randint(*params['term_range'])
        borrower_score = np.random.uniform(*params['score_range'])
        
        # Calculate monthly payment
        if loan_type != LoanType.CREDIT_CARD:
            monthly_rate = interest_rate / 12
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
                            ((1 + monthly_rate)**term_months - 1)
        else:
            monthly_payment = principal * 0.02  # Minimum payment approximation
        
        # Generate risk factors
        ltv_ratio = np.random.uniform(0.5, 0.9) if loan_type in [LoanType.AUTO, LoanType.MORTGAGE] else 0.0
        dti_ratio = np.random.uniform(0.2, 0.45)
        
        # Calculate default probability based on risk factors
        prob_default = self._calculate_default_probability(borrower_score, ltv_ratio, dti_ratio, loan_type)
        
        return Loan(
            loan_id=f"{loan_type.value}_{np.random.randint(100000, 999999)}",
            loan_type=loan_type,
            principal=principal,
            interest_rate=interest_rate,
            term_months=term_months,
            origination_date=datetime.now() - timedelta(days=np.random.randint(0, 365)),
            borrower_score=borrower_score,
            monthly_payment=monthly_payment,
            current_balance=principal * np.random.uniform(0.5, 1.0),  # Some amortization
            ltv_ratio=ltv_ratio,
            dti_ratio=dti_ratio,
            probability_default=prob_default
        )
    
    def _calculate_default_probability(
        self, 
        credit_score: float, 
        ltv: float, 
        dti: float, 
        loan_type: LoanType
    ) -> float:
        """Calculate probability of default based on loan characteristics"""
        # Simple logistic model for demonstration
        base_risk = 0.02
        
        # Credit score impact (higher score = lower risk)
        score_factor = np.exp(-(credit_score - 600) / 100)
        
        # LTV impact (higher LTV = higher risk)
        ltv_factor = ltv ** 2 if ltv > 0 else 1.0
        
        # DTI impact (higher DTI = higher risk)
        dti_factor = dti ** 1.5
        
        # Loan type risk multiplier
        type_multipliers = {
            LoanType.MORTGAGE: 0.5,
            LoanType.AUTO: 0.7,
            LoanType.BUSINESS: 1.2,
            LoanType.CONSUMER: 1.0,
            LoanType.CREDIT_CARD: 1.5
        }
        
        prob = base_risk * score_factor * ltv_factor * dti_factor * type_multipliers[loan_type]
        return min(prob, 0.5)  # Cap at 50%
    
    def _update_market_conditions(self):
        """Simulate market condition changes"""
        # Add some randomness to market conditions
        self.market_conditions.base_interest_rate += np.random.normal(0, 0.001)
        self.market_conditions.unemployment_rate += np.random.normal(0, 0.0005)
        self.market_conditions.gdp_growth += np.random.normal(0, 0.001)
        self.market_conditions.inflation_rate += np.random.normal(0, 0.0005)
        self.market_conditions.housing_price_index += np.random.normal(0, 0.5)
        self.market_conditions.volatility_index += np.random.normal(0, 1.0)
        
        # Keep within reasonable bounds
        self.market_conditions.base_interest_rate = np.clip(
            self.market_conditions.base_interest_rate, 0.01, 0.15
        )
        self.market_conditions.unemployment_rate = np.clip(
            self.market_conditions.unemployment_rate, 0.02, 0.15
        )
        self.market_conditions.volatility_index = np.clip(
            self.market_conditions.volatility_index, 5.0, 40.0
        )
    
    def _calculate_portfolio_metrics(self):
        """Calculate portfolio performance metrics"""
        if not self.portfolio.loans:
            return
        
        # Calculate total values
        self.portfolio.total_value = sum(loan.current_balance for loan in self.portfolio.loans)
        self.portfolio.total_exposure = sum(loan.principal for loan in self.portfolio.loans)
        
        # Calculate expected loss
        self.portfolio.expected_loss = sum(
            loan.current_balance * loan.probability_default 
            for loan in self.portfolio.loans
        )
        
        # Calculate delinquency rate
        delinquent_loans = [loan for loan in self.portfolio.loans if loan.status != LoanStatus.CURRENT]
        self.portfolio.delinquency_rate = len(delinquent_loans) / len(self.portfolio.loans)
        
        # Calculate charge-off rate (simplified)
        defaulted_loans = [loan for loan in self.portfolio.loans if loan.status == LoanStatus.DEFAULT]
        self.portfolio.charge_off_rate = len(defaulted_loans) / len(self.portfolio.loans)
        
        # Calculate ROE (simplified)
        total_interest = sum(
            loan.current_balance * loan.interest_rate / 12 
            for loan in self.portfolio.loans
        )
        total_equity = self.portfolio.total_value * 0.08  # 8% capital ratio
        self.portfolio.return_on_equity = (total_interest - self.portfolio.expected_loss) / total_equity
        
        # Calculate VaR (simplified Monte Carlo)
        self.portfolio.var_95 = self._calculate_var()
    
    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk using Monte Carlo simulation"""
        if not self.portfolio.loans:
            return 0.0
        
        # Simple VaR calculation - in practice would be more sophisticated
        expected_loss_rate = self.portfolio.expected_loss / self.portfolio.total_value
        volatility = np.sqrt(expected_loss_rate * (1 - expected_loss_rate))
        
        # Normal approximation for demonstration
        from scipy import stats
        var_95 = stats.norm.ppf(1 - confidence, expected_loss_rate, volatility)
        return abs(var_95)
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        # Calculate concentration risk
        type_counts = {}
        for loan in self.portfolio.loans:
            type_counts[loan.loan_type] = type_counts.get(loan.loan_type, 0) + 1
        
        concentration_risk = max(type_counts.values()) / len(self.portfolio.loans) if self.portfolio.loans else 0
        
        return np.array([
            self.portfolio.total_value / 1e6,  # Scale to millions
            self.portfolio.expected_loss / self.portfolio.total_value if self.portfolio.total_value > 0 else 0,
            self.portfolio.delinquency_rate,
            self.portfolio.return_on_equity,
            self.portfolio.var_95,
            concentration_risk,
            self.market_conditions.base_interest_rate,
            self.market_conditions.unemployment_rate,
            self.market_conditions.gdp_growth,
            self.market_conditions.inflation_rate,
            self.market_conditions.housing_price_index,
            self.market_conditions.credit_spread,
            self.market_conditions.volatility_index,
        ], dtype=np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for the current state and action"""
        # Multi-objective reward function
        
        # Financial performance component
        roe_reward = min(self.portfolio.return_on_equity / self.policies.target_roe, 2.0)
        
        # Risk management component
        var_penalty = -max(0, self.portfolio.var_95 - self.policies.max_var_limit) * 10
        
        # Delinquency penalty
        delinq_penalty = -self.portfolio.delinquency_rate * 5
        
        # Regulatory compliance
        compliance_bonus = 1.0 if self.portfolio.var_95 <= self.policies.max_var_limit else 0.0
        
        # Portfolio growth reward
        growth_reward = len(self.portfolio.loans) / self.max_portfolio_size
        
        # Combine components
        total_reward = roe_reward + var_penalty + delinq_penalty + compliance_bonus + growth_reward
        
        return total_reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        
        # Apply actions to policies (simplified)
        credit_policy_adj, pricing_adj, portfolio_rebal = action
        
        # Update business policies based on actions
        self.policies.min_credit_score += credit_policy_adj * 50
        self.policies.min_credit_score = np.clip(self.policies.min_credit_score, 550, 800)
        
        # Update market conditions
        self._update_market_conditions()
        
        # Simulate loan performance updates
        self._simulate_loan_performance()
        
        # Add new loans (simplified origination)
        if len(self.portfolio.loans) < self.max_portfolio_size:
            for _ in range(np.random.randint(5, 20)):  # Add 5-20 new loans per day
                new_loan = self._generate_synthetic_loan()
                if new_loan.borrower_score >= self.policies.min_credit_score:
                    self.portfolio.loans.append(new_loan)
        
        # Calculate portfolio metrics
        self._calculate_portfolio_metrics()
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        self.current_day += 1
        terminated = self.current_day >= self.simulation_days
        truncated = False
        
        # Store performance data
        self.performance_history.append({
            'day': self.current_day,
            'portfolio_value': self.portfolio.total_value,
            'roe': self.portfolio.return_on_equity,
            'var_95': self.portfolio.var_95,
            'delinquency_rate': self.portfolio.delinquency_rate,
            'reward': reward
        })
        
        info = {
            'portfolio_size': len(self.portfolio.loans),
            'total_value': self.portfolio.total_value,
            'expected_loss': self.portfolio.expected_loss,
            'market_conditions': self.market_conditions.__dict__
        }
        
        return observation, reward, terminated, truncated, info
    
    def _simulate_loan_performance(self):
        """Simulate loan performance changes over time"""
        for loan in self.portfolio.loans:
            # Simulate payment behavior
            if np.random.random() < loan.probability_default * 0.01:  # Daily default probability
                if loan.status == LoanStatus.CURRENT:
                    loan.status = LoanStatus.DELINQUENT_30
                    loan.days_past_due = 30
                elif loan.status == LoanStatus.DELINQUENT_30:
                    loan.status = LoanStatus.DELINQUENT_60
                    loan.days_past_due = 60
                elif loan.status == LoanStatus.DELINQUENT_60:
                    loan.status = LoanStatus.DELINQUENT_90
                    loan.days_past_due = 90
                elif loan.status == LoanStatus.DELINQUENT_90:
                    loan.status = LoanStatus.DEFAULT
                    loan.days_past_due = 120
            
            # Simulate prepayments
            if np.random.random() < 0.001:  # 0.1% daily prepayment probability
                loan.status = LoanStatus.PREPAID
                loan.current_balance = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_day = 0
        self.portfolio = Portfolio()
        self.market_conditions = MarketConditions()
        self.policies = BusinessPolicies()
        
        # Clear history
        self.performance_history = []
        self.decision_history = []
        
        # Generate initial portfolio
        for _ in range(self.initial_portfolio_size):
            loan = self._generate_synthetic_loan()
            self.portfolio.loans.append(loan)
        
        # Calculate initial metrics
        self._calculate_portfolio_metrics()
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'portfolio_size': len(self.portfolio.loans),
            'total_value': self.portfolio.total_value
        }
        
        return observation, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            print(f"\n=== Day {self.current_day} ===")
            print(f"Portfolio Size: {len(self.portfolio.loans)}")
            print(f"Total Value: ${self.portfolio.total_value:,.2f}")
            print(f"ROE: {self.portfolio.return_on_equity:.2%}")
            print(f"VaR 95%: {self.portfolio.var_95:.2%}")
            print(f"Delinquency Rate: {self.portfolio.delinquency_rate:.2%}")
            print(f"Base Interest Rate: {self.market_conditions.base_interest_rate:.2%}")
            print(f"Unemployment Rate: {self.market_conditions.unemployment_rate:.2%}")
    
    def close(self):
        """Close the environment"""
        pass
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as DataFrame"""
        return pd.DataFrame(self.performance_history)


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = LoanPortfolioTwinEnv(
        initial_portfolio_size=100,
        max_portfolio_size=1000,
        simulation_days=30,
        render_mode="human"
    )
    
    # Test environment
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few steps
    for step in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    # Get performance summary
    performance_df = env.get_performance_summary()
    print("\nPerformance Summary:")
    print(performance_df.head())
