"""
Sample Simulation Module for Consumer Loan Data

This module handles the generation of synthetic loan data with realistic features
and ground truth default probabilities, including the simulation of loan approval
processes that create sample selection bias.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LoanDataSimulator:
    """
    Simulates consumer loan data with realistic features and selection bias
    """
    
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Ground truth coefficients for default probability calculation
        self.true_coefficients = {
            'intercept': -3.0,
            'credit_score_norm': -4.0,
            'debt_to_income': 3.0,
            'income_norm': -0.5,
            'loan_amount_scaled': 0.3,
            'employment_scaled': -0.2,
            'excess_credit_lines': 0.1
        }
        
    def generate_population_data(self):
        """
        Generate synthetic population data with realistic loan applicant features
        """
        np.random.seed(self.random_state)
        
        # Credit score (300-850, higher is better)
        credit_score = np.random.normal(650, 100, self.n_samples)
        credit_score = np.clip(credit_score, 300, 850)
        
        # Annual income (log-normal distribution)
        log_income = np.random.normal(11, 0.8, self.n_samples)  # ~$60k median
        annual_income = np.exp(log_income)
        
        # Debt-to-income ratio (affected by income)
        base_dti = np.random.normal(0.3, 0.15, self.n_samples)
        # Higher income tends to have lower DTI
        income_effect = -0.1 * (np.log(annual_income) - 11)
        debt_to_income = base_dti + income_effect
        debt_to_income = np.clip(debt_to_income, 0.05, 0.8)
        
        # Employment length (years)
        employment_length = np.random.exponential(3, self.n_samples)
        employment_length = np.clip(employment_length, 0, 25)
        
        # Loan amount requested
        loan_amount = np.random.normal(15000, 8000, self.n_samples)
        loan_amount = np.clip(loan_amount, 1000, 50000)
        
        # Number of open credit lines
        open_credit_lines = np.random.poisson(8, self.n_samples)
        
        # Age
        age = np.random.normal(40, 15, self.n_samples)
        age = np.clip(age, 18, 80)
        
        return pd.DataFrame({
            'credit_score': credit_score,
            'annual_income': annual_income,
            'debt_to_income': debt_to_income,
            'employment_length': employment_length,
            'loan_amount': loan_amount,
            'open_credit_lines': open_credit_lines,
            'age': age
        })
    
    def calculate_true_default_probability(self, df):
        """
        Calculate the TRUE probability of default for each applicant
        This represents the ground truth that we want our model to predict
        """
        # Normalize features for probability calculation
        credit_score_norm = (df['credit_score'] - 300) / (850 - 300)
        income_norm = np.log(df['annual_income'] / 30000) / 3
        
        # Create logistic function for default probability using true coefficients
        logit = (
            self.true_coefficients['intercept'] +
            self.true_coefficients['credit_score_norm'] * credit_score_norm +
            self.true_coefficients['debt_to_income'] * df['debt_to_income'] +
            self.true_coefficients['income_norm'] * income_norm +
            self.true_coefficients['loan_amount_scaled'] * (df['loan_amount'] / 10000) +
            self.true_coefficients['employment_scaled'] * (df['employment_length'] / 10) +
            self.true_coefficients['excess_credit_lines'] * np.maximum(0, df['open_credit_lines'] - 10)
        )
        
        true_default_prob = 1 / (1 + np.exp(-logit))
        return true_default_prob
    
    def simulate_loan_approval_process(self, df, rejection_rate=0.4):
        """
        Simulate loan approval process that creates selection bias
        Banks typically reject high-risk applicants, creating sample selection bias
        """
        np.random.seed(self.random_state)
        
        # Calculate approval probability based on similar factors as default risk
        # but with some noise to make it realistic
        credit_score_norm = (df['credit_score'] - 300) / (850 - 300)
        income_norm = np.log(df['annual_income'] / 30000) / 3
        
        # Approval score (higher = more likely to approve)
        approval_score = (
            2.0 * credit_score_norm +
            0.5 * income_norm +
            -2.0 * df['debt_to_income'] +
            0.3 * (df['employment_length'] / 10) +
            -0.2 * (df['loan_amount'] / 10000) +
            np.random.normal(0, 0.3, len(df))  # Add noise
        )
        
        # Set approval threshold to achieve desired rejection rate
        threshold = np.percentile(approval_score, rejection_rate * 100)
        approved = approval_score > threshold
        
        return approved, approval_score
    
    def generate_observed_outcomes(self, df, true_default_prob, approved):
        """
        Generate observed outcomes - we only see defaults for approved loans
        """
        np.random.seed(self.random_state)
        
        # Generate actual defaults based on true probabilities
        actual_defaults = np.random.binomial(1, true_default_prob, len(df))
        
        # Create observed dataset (only approved loans)
        observed_defaults = actual_defaults.astype(float)
        observed_defaults[~approved] = np.nan  # We don't observe rejected loan outcomes
        
        return actual_defaults, observed_defaults
    
    def create_normalized_features(self, df):
        """
        Create normalized features matching the ground truth model structure
        """
        df_normalized = df.copy()
        df_normalized['credit_score_norm'] = (df['credit_score'] - 300) / (850 - 300)
        df_normalized['income_norm'] = np.log(df['annual_income'] / 30000) / 3
        df_normalized['loan_amount_scaled'] = df['loan_amount'] / 10000
        df_normalized['employment_scaled'] = df['employment_length'] / 10
        df_normalized['excess_credit_lines'] = np.maximum(0, df['open_credit_lines'] - 10)
        
        return df_normalized
    
    def simulate_external_predictor(self, df, true_default_prob, correlation_strength=0.6):
        """
        Simulate an external predictor (e.g., alternative credit data)
        that's correlated with true default risk but has some independence
        """
        np.random.seed(self.random_state)
        
        # Create external predictor that's:
        # 1. Correlated with true default probability
        # 2. Partially independent of approval factors
        # 3. Available for both approved and rejected samples
        
        credit_score_norm = (df['credit_score'] - 300) / (850 - 300)
        
        external_signal = (
            correlation_strength * true_default_prob +  # Correlation with true default risk
            0.2 * (1 - credit_score_norm) +  # Some correlation with credit score
            0.1 * df['debt_to_income'] +  # Some correlation with DTI
            (1 - correlation_strength - 0.3) * np.random.normal(0, 0.2, len(df))  # Random noise
        )
        
        # Normalize to 0-1 scale
        external_predictor = (external_signal - external_signal.min()) / (external_signal.max() - external_signal.min())
        
        return external_predictor
    
    def get_feature_columns(self):
        """Return the standard feature columns used in modeling"""
        return ['credit_score', 'annual_income', 'debt_to_income', 
                'employment_length', 'loan_amount', 'open_credit_lines', 'age']
    
    def get_normalized_feature_columns(self):
        """Return the normalized feature columns used in modeling"""
        return ['credit_score_norm', 'income_norm', 'debt_to_income', 
                'employment_scaled', 'loan_amount_scaled', 'excess_credit_lines']
    
    def get_true_coefficients_array(self):
        """Return true coefficients as array for comparison"""
        return [
            self.true_coefficients['intercept'],
            self.true_coefficients['credit_score_norm'],
            self.true_coefficients['income_norm'],
            self.true_coefficients['debt_to_income'],
            self.true_coefficients['employment_scaled'],
            self.true_coefficients['loan_amount_scaled'],
            self.true_coefficients['excess_credit_lines']
        ]
    
    def get_feature_names(self):
        """Return feature names for coefficient comparison"""
        return ['Intercept'] + self.get_normalized_feature_columns()


class SimulationScenario:
    """
    Defines a complete simulation scenario with all parameters
    """
    
    def __init__(self, name, n_samples=10000, rejection_rate=0.5, 
                 external_predictor_strength=0.6, random_state=42):
        self.name = name
        self.n_samples = n_samples
        self.rejection_rate = rejection_rate
        self.external_predictor_strength = external_predictor_strength
        self.random_state = random_state
    
    def generate_complete_dataset(self):
        """
        Generate a complete dataset for the scenario
        """
        simulator = LoanDataSimulator(self.n_samples, self.random_state)
        
        # Generate base population data
        df = simulator.generate_population_data()
        
        # Calculate true default probabilities
        true_default_prob = simulator.calculate_true_default_probability(df)
        
        # Simulate approval process
        approved, approval_score = simulator.simulate_loan_approval_process(df, self.rejection_rate)
        
        # Generate observed outcomes
        actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
            df, true_default_prob, approved
        )
        
        # Create normalized features
        df_normalized = simulator.create_normalized_features(df)
        
        # Add outcomes to dataset
        df_normalized['true_default_prob'] = true_default_prob
        df_normalized['actual_default'] = actual_defaults
        df_normalized['approved'] = approved
        df_normalized['observed_default'] = observed_defaults
        df_normalized['approval_score'] = approval_score
        
        # Add external predictor if specified
        if self.external_predictor_strength > 0:
            external_predictor = simulator.simulate_external_predictor(
                df, true_default_prob, self.external_predictor_strength
            )
            df_normalized['external_predictor'] = external_predictor
        
        return df_normalized, simulator


# Predefined scenarios for common use cases
PREDEFINED_SCENARIOS = {
    'low_rejection': SimulationScenario(
        name='Low Rejection Rate',
        rejection_rate=0.2,
        n_samples=15000
    ),
    'moderate_rejection': SimulationScenario(
        name='Moderate Rejection Rate',
        rejection_rate=0.5,
        n_samples=15000
    ),
    'high_rejection': SimulationScenario(
        name='High Rejection Rate',
        rejection_rate=0.8,
        n_samples=15000
    ),
    'external_data_available': SimulationScenario(
        name='External Data Available',
        rejection_rate=0.5,
        external_predictor_strength=0.7,
        n_samples=15000
    ),
    'weak_external_data': SimulationScenario(
        name='Weak External Data',
        rejection_rate=0.5,
        external_predictor_strength=0.3,
        n_samples=15000
    )
} 