"""
Credit Survival Model using Cox Hazard Analysis

This module implements a comprehensive credit default prediction model using Cox proportional hazards
regression with time-varying covariates. The implementation follows survival analysis best practices
and includes train-test-validation paradigm with extensive model validation and visualization.

Features:
- Cox proportional hazards model with time-varying covariates
- Train-test-validation data split
- Comprehensive model validation metrics
- Extensive visualization for model understanding
- Risk stratification and performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class CreditSurvivalModel:
    """
    A comprehensive credit survival analysis model using Cox proportional hazards regression.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Credit Survival Model.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Model components
        self.cox_model = None
        self.scaler = StandardScaler()
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.val_data = None
        
        # Model performance metrics
        self.train_c_index = None
        self.test_c_index = None
        self.val_c_index = None
        
        # Feature importance
        self.feature_importance = None
        
    def generate_sample_data(self, n_subjects=1000, max_time=60, test_size=0.2, val_size=0.1):
        """
        Generate synthetic credit default data with time-varying covariates and payment patterns.
        
        Parameters:
        -----------
        n_subjects : int
            Number of loan subjects to generate
        max_time : int
            Maximum observation time in months
        test_size : float
            Proportion of data for testing
        val_size : float
            Proportion of data for validation
            
        Returns:
        --------
        tuple : (train_data, test_data, val_data)
        """
        print("Generating synthetic credit default data with payment patterns...")
        
        # True model parameters for data generation
        TRUE_BETA_AGE = 0.03
        TRUE_BETA_INCOME = -0.00002
        TRUE_BETA_LOAN_AMOUNT = 0.00001
        TRUE_BETA_CREDIT_SCORE = -0.008
        TRUE_BETA_EMPLOYMENT_YEARS = -0.05
        TRUE_BETA_LOAN_DURATION = 0.025
        TRUE_BETA_DEBT_TO_INCOME = 0.02
        TRUE_BETA_PAYMENT_HISTORY = -0.1
        
        # NEW: Payment pattern effects
        TRUE_BETA_BALLOON_PAYMENT = 0.4  # Balloon payments increase risk
        TRUE_BETA_INTEREST_ONLY = 0.25   # Interest-only increases risk moderately
        TRUE_BETA_BALLOON_NEAR_MATURITY = 0.6  # Additional risk when balloon payment approaches
        
        BASE_LOG_HAZARD = np.log(0.003)
        
        # Centering values for covariates
        CENTER_AGE = 40
        CENTER_INCOME = 50000
        CENTER_LOAN_AMOUNT = 25000
        CENTER_CREDIT_SCORE = 650
        CENTER_EMPLOYMENT_YEARS = 5
        CENTER_LOAN_DURATION = 24
        CENTER_DEBT_TO_INCOME = 0.3
        CENTER_PAYMENT_HISTORY = 0.95
        
        data_rows = []
        
        for i in range(n_subjects):
            # Generate static features for each subject
            age = np.random.randint(22, 70)
            income = np.random.normal(50000, 20000)
            income = max(income, 15000)  # Minimum income threshold
            
            credit_score = np.random.normal(650, 100)
            credit_score = np.clip(credit_score, 300, 850)
            
            employment_years = np.random.exponential(5)
            employment_years = np.clip(employment_years, 0, 40)
            
            loan_amount = np.random.uniform(5000, 100000)
            debt_to_income = np.random.beta(2, 5) * 0.8  # Realistic DTI distribution
            
            # Payment history score (higher is better)
            payment_history = np.random.beta(8, 2)  # Skewed towards higher values
            
            # NEW: Generate payment pattern
            # Payment pattern probabilities based on loan characteristics
            high_loan_amount = loan_amount > 50000
            low_credit_score = credit_score < 600
            
            # Higher chance of balloon/interest-only for larger loans or lower credit scores
            pattern_weights = [0.7, 0.2, 0.1]  # [installment, balloon, interest_only]
            if high_loan_amount or low_credit_score:
                pattern_weights = [0.5, 0.35, 0.15]  # More balloon/interest-only
            
            payment_pattern = np.random.choice(['installment', 'balloon', 'interest_only'], p=pattern_weights)
            
            # Generate loan term based on payment pattern
            if payment_pattern == 'balloon':
                loan_term_months = np.random.choice([36, 48, 60], p=[0.3, 0.4, 0.3])  # Shorter terms for balloon
            elif payment_pattern == 'interest_only':
                loan_term_months = np.random.choice([60, 72, 84], p=[0.4, 0.4, 0.2])  # Longer terms for IO
            else:  # installment
                loan_term_months = np.random.choice([24, 36, 48, 60], p=[0.2, 0.3, 0.3, 0.2])
            
            # Binary indicators for payment patterns
            is_balloon_payment = 1 if payment_pattern == 'balloon' else 0
            is_interest_only = 1 if payment_pattern == 'interest_only' else 0
            # is_installment is the reference category (both others = 0)
            
            current_time = 0
            event_occurred = False
            
            while current_time < max_time and not event_occurred:
                # Random segment duration
                segment_duration = np.random.randint(1, 7)  # 1-6 months per segment
                start_time = current_time
                stop_time = min(current_time + segment_duration, max_time)
                actual_duration = stop_time - start_time
                
                if actual_duration <= 0:
                    break
                
                # Time-varying covariates
                loan_duration_at_stop = stop_time
                
                # Payment history can deteriorate over time (slight random walk)
                if len(data_rows) > 0 and data_rows[-1]['id'] == i:
                    prev_payment_history = data_rows[-1]['payment_history_score']
                    payment_history = prev_payment_history + np.random.normal(0, 0.01)
                    payment_history = np.clip(payment_history, 0, 1)
                
                # Economic stress factor (increases over time)
                economic_stress = min(0.1 * (stop_time / 12), 0.5)  # Increases annually
                
                # NEW: Time-varying balloon payment risk
                # Risk increases significantly as balloon payment approaches
                balloon_maturity_risk = 0
                if is_balloon_payment:
                    months_to_maturity = loan_term_months - loan_duration_at_stop
                    if months_to_maturity <= 12:  # Within 12 months of balloon payment
                        # Risk increases exponentially as balloon approaches
                        balloon_maturity_risk = TRUE_BETA_BALLOON_NEAR_MATURITY * (1 - months_to_maturity / 12)
                        balloon_maturity_risk = max(0, balloon_maturity_risk)
                
                # Calculate hazard rate for this segment
                log_hazard = (
                    BASE_LOG_HAZARD +
                    TRUE_BETA_AGE * (age - CENTER_AGE) +
                    TRUE_BETA_INCOME * (income - CENTER_INCOME) +
                    TRUE_BETA_LOAN_AMOUNT * (loan_amount - CENTER_LOAN_AMOUNT) +
                    TRUE_BETA_CREDIT_SCORE * (credit_score - CENTER_CREDIT_SCORE) +
                    TRUE_BETA_EMPLOYMENT_YEARS * (employment_years - CENTER_EMPLOYMENT_YEARS) +
                    TRUE_BETA_LOAN_DURATION * (loan_duration_at_stop - CENTER_LOAN_DURATION) +
                    TRUE_BETA_DEBT_TO_INCOME * (debt_to_income - CENTER_DEBT_TO_INCOME) +
                    TRUE_BETA_PAYMENT_HISTORY * (payment_history - CENTER_PAYMENT_HISTORY) +
                    TRUE_BETA_BALLOON_PAYMENT * is_balloon_payment +  # NEW: Balloon payment effect
                    TRUE_BETA_INTEREST_ONLY * is_interest_only +      # NEW: Interest-only effect
                    balloon_maturity_risk +                           # NEW: Time-varying balloon risk
                    economic_stress * 0.5  # Additional economic stress component
                )
                
                hazard_rate = np.exp(log_hazard)
                prob_event = 1 - np.exp(-hazard_rate * actual_duration)
                
                event_status = 0
                if np.random.rand() < prob_event:
                    event_status = 1
                    event_occurred = True
                
                # Store the observation
                data_rows.append({
                    'id': i,
                    'age': age,
                    'income': round(income, 2),
                    'loan_amount': round(loan_amount, 2),
                    'credit_score': round(credit_score, 1),
                    'employment_years': round(employment_years, 1),
                    'debt_to_income_ratio': round(debt_to_income, 3),
                    'payment_history_score': round(payment_history, 3),
                    'payment_pattern': payment_pattern,                    # NEW: Categorical payment pattern
                    'is_balloon_payment': is_balloon_payment,             # NEW: Binary indicator
                    'is_interest_only': is_interest_only,                 # NEW: Binary indicator  
                    'loan_term_months': loan_term_months,                 # NEW: Original loan term
                    'months_to_maturity': max(0, loan_term_months - loan_duration_at_stop),  # NEW: Time to maturity
                    'start_time': round(start_time, 2),
                    'stop_time': round(stop_time, 2),
                    'loan_duration_months': round(loan_duration_at_stop, 2),
                    'event': event_status
                })
                
                current_time = stop_time
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        df = df[df['start_time'] < df['stop_time']]  # Ensure valid intervals
        
        # Split subjects into train/test/validation
        unique_subjects = df['id'].unique()
        subjects_train, subjects_temp = train_test_split(
            unique_subjects, test_size=(test_size + val_size), random_state=self.random_state
        )
        subjects_test, subjects_val = train_test_split(
            subjects_temp, test_size=(val_size / (test_size + val_size)), random_state=self.random_state
        )
        
        # Create data splits
        train_data = df[df['id'].isin(subjects_train)].copy()
        test_data = df[df['id'].isin(subjects_test)].copy()
        val_data = df[df['id'].isin(subjects_val)].copy()
        
        # Store data
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        
        # Print summary including payment patterns
        print(f"Data generation complete!")
        print(f"Total observations: {len(df)}")
        print(f"Unique subjects: {len(unique_subjects)}")
        print(f"Train subjects: {len(subjects_train)} ({len(train_data)} observations)")
        print(f"Test subjects: {len(subjects_test)} ({len(test_data)} observations)")
        print(f"Validation subjects: {len(subjects_val)} ({len(val_data)} observations)")
        print(f"Overall default rate: {df.groupby('id')['event'].max().mean():.3f}")
        
        # Payment pattern distribution
        pattern_dist = df.groupby('id')['payment_pattern'].first().value_counts(normalize=True)
        print(f"\nPayment Pattern Distribution:")
        for pattern, pct in pattern_dist.items():
            default_rate = df[df['payment_pattern'] == pattern].groupby('id')['event'].max().mean()
            print(f"  {pattern}: {pct:.1%} of loans (default rate: {default_rate:.3f})")
        
        return train_data, test_data, val_data
    
    def prepare_features(self, data, fit_scaler=False):
        """
        Prepare features for modeling including payment pattern indicators.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        fit_scaler : bool
            Whether to fit the scaler (True for training data)
            
        Returns:
        --------
        pd.DataFrame : Prepared data with scaled features
        """
        # Define feature columns
        continuous_features = [
            'age', 'income', 'loan_amount', 'credit_score', 
            'employment_years', 'debt_to_income_ratio', 
            'payment_history_score', 'loan_duration_months', 'months_to_maturity'
        ]
        
        # Binary features (don't need scaling)
        binary_features = ['is_balloon_payment', 'is_interest_only']
        
        # Create a copy of the data
        prepared_data = data.copy()
        
        # Scale continuous features only
        if fit_scaler:
            prepared_data[continuous_features] = self.scaler.fit_transform(
                prepared_data[continuous_features]
            )
        else:
            prepared_data[continuous_features] = self.scaler.transform(
                prepared_data[continuous_features]
            )
        
        # Ensure binary features are integers
        for feature in binary_features:
            if feature in prepared_data.columns:
                prepared_data[feature] = prepared_data[feature].astype(int)
        
        return prepared_data
    
    def fit_cox_model(self, penalizer=0.01):
        """
        Fit Cox proportional hazards model with time-varying covariates and payment patterns.
        
        Parameters:
        -----------
        penalizer : float
            Regularization parameter for the Cox model
        """
        print("Fitting Cox Proportional Hazards Model with Payment Patterns...")
        
        # Prepare training data
        train_prepared = self.prepare_features(self.train_data, fit_scaler=True)
        
        # Define covariates for the model (including payment pattern indicators)
        covariates = [
            'age', 'income', 'loan_amount', 'credit_score', 
            'employment_years', 'debt_to_income_ratio', 
            'payment_history_score', 'loan_duration_months', 'months_to_maturity',
            'is_balloon_payment', 'is_interest_only'  # NEW: Payment pattern indicators
        ]
        
        # Required columns for Cox model
        required_cols = ['id', 'event', 'start_time', 'stop_time'] + covariates
        
        # Filter to include only required columns and ensure all are numeric except id columns
        model_data = train_prepared[required_cols].copy()
        
        # Ensure covariates are all numeric
        for col in covariates:
            if col in model_data.columns:
                model_data[col] = pd.to_numeric(model_data[col], errors='coerce')
        
        # Check for any NaN values that might have been created
        if model_data[covariates].isnull().any().any():
            print("Warning: NaN values found in covariates after conversion")
            print(model_data[covariates].isnull().sum())
            # Fill NaN values with 0 for binary features
            model_data['is_balloon_payment'] = model_data['is_balloon_payment'].fillna(0)
            model_data['is_interest_only'] = model_data['is_interest_only'].fillna(0)
        
        # Initialize and fit the Cox model
        self.cox_model = CoxTimeVaryingFitter(penalizer=penalizer)
        
        self.cox_model.fit(
            model_data,
            id_col='id',
            event_col='event',
            start_col='start_time',
            stop_col='stop_time',
            show_progress=True
        )
        
        # Calculate concordance indices
        self.train_c_index = self._calculate_concordance_index(model_data)
        
        if self.test_data is not None:
            test_prepared = self.prepare_features(self.test_data, fit_scaler=False)
            self.test_c_index = self._calculate_concordance_index(test_prepared)
        
        if self.val_data is not None:
            val_prepared = self.prepare_features(self.val_data, fit_scaler=False)
            self.val_c_index = self._calculate_concordance_index(val_prepared)
        
        # Store feature importance
        self.feature_importance = self.cox_model.summary.copy()
        
        print("Model fitting complete!")
        print(f"Training C-index: {self.train_c_index:.4f}")
        if self.test_c_index:
            print(f"Test C-index: {self.test_c_index:.4f}")
        if self.val_c_index:
            print(f"Validation C-index: {self.val_c_index:.4f}")
            
        # Print payment pattern effects
        print(f"\nPayment Pattern Effects:")
        if 'is_balloon_payment' in self.cox_model.summary.index:
            balloon_hr = np.exp(self.cox_model.summary.loc['is_balloon_payment', 'coef'])
            balloon_p = self.cox_model.summary.loc['is_balloon_payment', 'p']
            print(f"  Balloon Payment HR: {balloon_hr:.3f} (p={balloon_p:.4f})")
        
        if 'is_interest_only' in self.cox_model.summary.index:
            io_hr = np.exp(self.cox_model.summary.loc['is_interest_only', 'coef'])
            io_p = self.cox_model.summary.loc['is_interest_only', 'p']
            print(f"  Interest-Only HR: {io_hr:.3f} (p={io_p:.4f})")
        
        if 'months_to_maturity' in self.cox_model.summary.index:
            maturity_hr = np.exp(self.cox_model.summary.loc['months_to_maturity', 'coef'])
            maturity_p = self.cox_model.summary.loc['months_to_maturity', 'p']
            print(f"  Months to Maturity HR: {maturity_hr:.3f} (p={maturity_p:.4f})")
    
    def _calculate_concordance_index(self, data):
        """Calculate concordance index for the given data."""
        try:
            # Get last observation for each subject
            last_obs = data.groupby('id').last().reset_index()
            
            # Predict partial hazard (risk scores)
            covariates = [
                'age', 'income', 'loan_amount', 'credit_score', 
                'employment_years', 'debt_to_income_ratio', 
                'payment_history_score', 'loan_duration_months', 'months_to_maturity',
                'is_balloon_payment', 'is_interest_only'
            ]
            
            risk_scores = self.cox_model.predict_partial_hazard(last_obs[covariates])
            
            # Calculate C-index
            c_idx = concordance_index(
                last_obs['stop_time'],
                -risk_scores,  # Negative because higher risk = lower survival
                last_obs['event']
            )
            
            return c_idx
        except Exception as e:
            print(f"Error calculating C-index: {e}")
            return None
    
    def predict_survival_probability(self, data, time_points=None):
        """
        Predict survival probabilities for given data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data for prediction
        time_points : list
            Time points for survival probability prediction
            
        Returns:
        --------
        dict : Survival probabilities for each subject
        """
        if self.cox_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if time_points is None:
            time_points = [6, 12, 24, 36, 48, 60]  # Default time points in months
        
        # Prepare data
        prepared_data = self.prepare_features(data, fit_scaler=False)
        
        # Get baseline cumulative hazard
        baseline_hazard = self.cox_model.baseline_cumulative_hazard_
        
        # Calculate survival probabilities for each subject
        survival_probs = {}
        
        unique_subjects = prepared_data['id'].unique()
        
        for subject_id in unique_subjects:
            subject_data = prepared_data[prepared_data['id'] == subject_id].sort_values('stop_time')
            
            # Calculate cumulative hazard path for the subject
            survival_probs[subject_id] = self._calculate_subject_survival(
                subject_data, baseline_hazard, time_points
            )
        
        return survival_probs
    
    def _calculate_subject_survival(self, subject_data, baseline_hazard, time_points):
        """Calculate survival probabilities for a single subject."""
        covariates = [
            'age', 'income', 'loan_amount', 'credit_score', 
            'employment_years', 'debt_to_income_ratio', 
            'payment_history_score', 'loan_duration_months', 'months_to_maturity',
            'is_balloon_payment', 'is_interest_only'
        ]
        
        def get_baseline_hazard_at_time(t):
            """Get baseline cumulative hazard at time t."""
            if baseline_hazard.empty or t <= baseline_hazard.index.min():
                return 0.0
            relevant_entries = baseline_hazard[baseline_hazard.index <= t]
            if relevant_entries.empty:
                return 0.0
            return relevant_entries.iloc[-1, 0]
        
        survival_probs = {}
        
        for t in time_points:
            cumulative_hazard = 0.0
            current_time = 0.0
            
            for _, interval in subject_data.iterrows():
                interval_start = max(interval['start_time'], current_time)
                interval_stop = min(interval['stop_time'], t)
                
                if interval_start >= interval_stop:
                    continue
                
                # Get partial hazard for this interval
                interval_covariates = interval[covariates].values.reshape(1, -1)
                partial_hazard = self.cox_model.predict_partial_hazard(
                    pd.DataFrame(interval_covariates, columns=covariates)
                ).iloc[0]
                
                # Get baseline hazard increment
                h0_start = get_baseline_hazard_at_time(interval_start)
                h0_stop = get_baseline_hazard_at_time(interval_stop)
                delta_h0 = h0_stop - h0_start
                
                # Add to cumulative hazard
                cumulative_hazard += partial_hazard * delta_h0
                
                current_time = interval_stop
                
                if current_time >= t:
                    break
            
            # Calculate survival probability
            survival_probs[t] = np.exp(-cumulative_hazard)
        
        return survival_probs
    
    def plot_model_summary(self):
        """Create comprehensive visualization of model results."""
        if self.cox_model is None:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model coefficients
        ax1 = plt.subplot(3, 3, 1)
        coeffs = self.cox_model.summary
        y_pos = np.arange(len(coeffs))
        
        plt.barh(y_pos, coeffs['coef'], color='skyblue', alpha=0.7)
        plt.yticks(y_pos, coeffs.index)
        plt.xlabel('Coefficient Value')
        plt.title('Cox Model Coefficients')
        plt.grid(True, alpha=0.3)
        
        # Add confidence intervals
        for i, (idx, row) in enumerate(coeffs.iterrows()):
            plt.barh(i, row['coef upper 95%'] - row['coef lower 95%'], 
                    left=row['coef lower 95%'], alpha=0.3, color='red')
        
        # 2. Baseline survival curve
        ax2 = plt.subplot(3, 3, 2)
        baseline_survival = self.cox_model.baseline_survival_
        plt.plot(baseline_survival.index, baseline_survival.iloc[:, 0], 
                linewidth=2, color='darkblue')
        plt.xlabel('Time (months)')
        plt.ylabel('Baseline Survival Probability')
        plt.title('Baseline Survival Function')
        plt.grid(True, alpha=0.3)
        
        # 3. Baseline cumulative hazard
        ax3 = plt.subplot(3, 3, 3)
        baseline_hazard = self.cox_model.baseline_cumulative_hazard_
        plt.plot(baseline_hazard.index, baseline_hazard.iloc[:, 0], 
                linewidth=2, color='darkred')
        plt.xlabel('Time (months)')
        plt.ylabel('Baseline Cumulative Hazard')
        plt.title('Baseline Cumulative Hazard Function')
        plt.grid(True, alpha=0.3)
        
        # 4. Feature importance (hazard ratios)
        ax4 = plt.subplot(3, 3, 4)
        hazard_ratios = np.exp(coeffs['coef'])
        y_pos = np.arange(len(hazard_ratios))
        
        colors = ['red' if hr > 1 else 'green' for hr in hazard_ratios]
        plt.barh(y_pos, hazard_ratios, color=colors, alpha=0.7)
        plt.yticks(y_pos, coeffs.index)
        plt.xlabel('Hazard Ratio')
        plt.title('Feature Hazard Ratios')
        plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 5. P-values significance
        ax5 = plt.subplot(3, 3, 5)
        p_values = coeffs['p']
        significant = p_values < 0.05
        
        colors = ['green' if sig else 'red' for sig in significant]
        plt.barh(y_pos, -np.log10(p_values), color=colors, alpha=0.7)
        plt.yticks(y_pos, coeffs.index)
        plt.xlabel('-log10(p-value)')
        plt.title('Feature Significance')
        plt.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Model performance comparison
        ax6 = plt.subplot(3, 3, 6)
        performance_data = {
            'Training': self.train_c_index,
            'Test': self.test_c_index,
            'Validation': self.val_c_index
        }
        performance_data = {k: v for k, v in performance_data.items() if v is not None}
        
        bars = plt.bar(performance_data.keys(), performance_data.values(), 
                      color=['blue', 'orange', 'green'], alpha=0.7)
        plt.ylabel('C-index')
        plt.title('Model Performance (Concordance Index)')
        plt.ylim(0.5, 1.0)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, (k, v) in zip(bars, performance_data.items()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{v:.3f}', ha='center', va='bottom')
        
        # 7. Survival curves for risk groups
        ax7 = plt.subplot(3, 3, 7)
        self._plot_risk_stratified_survival(ax7)
        
        # 8. Residual analysis
        ax8 = plt.subplot(3, 3, 8)
        self._plot_schoenfeld_residuals(ax8)
        
        # 9. Risk distribution
        ax9 = plt.subplot(3, 3, 9)
        self._plot_risk_distribution(ax9)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_risk_stratified_survival(self, ax):
        """Plot survival curves stratified by risk groups and payment patterns."""
        if self.test_data is None:
            return
        
        # Prepare test data
        test_prepared = self.prepare_features(self.test_data, fit_scaler=False)
        
        # Get risk scores for test subjects
        last_obs = test_prepared.groupby('id').last().reset_index()
        covariates = [
            'age', 'income', 'loan_amount', 'credit_score', 
            'employment_years', 'debt_to_income_ratio', 
            'payment_history_score', 'loan_duration_months', 'months_to_maturity',
            'is_balloon_payment', 'is_interest_only'
        ]
        
        risk_scores = self.cox_model.predict_partial_hazard(last_obs[covariates])
        
        # Create payment pattern groups
        payment_patterns = ['installment', 'balloon', 'interest_only']
        colors = ['green', 'red', 'orange']
        
        # Plot Kaplan-Meier curves for each payment pattern
        for i, pattern in enumerate(payment_patterns):
            pattern_mask = last_obs['payment_pattern'] == pattern
            if pattern_mask.sum() == 0:
                continue
                
            group_data = last_obs[pattern_mask]
            
            if len(group_data) > 5:  # Only plot if sufficient data
                kmf = KaplanMeierFitter()
                kmf.fit(group_data['stop_time'], group_data['event'], 
                       label=f'{pattern.title()} Payments')
                kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=2)
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Curves by Payment Pattern')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_schoenfeld_residuals(self, ax):
        """Plot Schoenfeld residuals for proportional hazards assumption."""
        try:
            # For CoxTimeVaryingFitter, residual computation is more complex
            # This is a simplified approach - we'll use a basic implementation
            
            # Prepare training data for residual analysis
            train_prepared = self.prepare_features(self.train_data, fit_scaler=False)
            
            # Get event times and create a basic residual proxy
            event_data = train_prepared[train_prepared['event'] == 1].copy()
            
            if len(event_data) > 0:
                # Use risk scores as a proxy for residual analysis
                covariates = [
                    'age', 'income', 'loan_amount', 'credit_score', 
                    'employment_years', 'debt_to_income_ratio', 
                    'payment_history_score', 'loan_duration_months'
                ]
                
                risk_scores = self.cox_model.predict_partial_hazard(event_data[covariates])
                
                # Plot risk scores vs event times (pseudo-residuals)
                ax.scatter(event_data['stop_time'], risk_scores, alpha=0.5, s=20)
                ax.set_xlabel('Event Time (months)')
                ax.set_ylabel('Risk Score')
                ax.set_title('Risk Scores vs Event Time\n(Proxy for Residual Analysis)')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(event_data['stop_time'], risk_scores, 1)
                p = np.poly1d(z)
                ax.plot(event_data['stop_time'], p(event_data['stop_time']), "r--", alpha=0.8)
            else:
                ax.text(0.5, 0.5, 'No events for residual analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Residual Analysis')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Residual analysis unavailable\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Residual Analysis')
    
    def _plot_risk_distribution(self, ax):
        """Plot distribution of risk scores by payment pattern."""
        if self.test_data is None:
            return
        
        # Get risk scores
        test_prepared = self.prepare_features(self.test_data, fit_scaler=False)
        last_obs = test_prepared.groupby('id').last().reset_index()
        
        covariates = [
            'age', 'income', 'loan_amount', 'credit_score', 
            'employment_years', 'debt_to_income_ratio', 
            'payment_history_score', 'loan_duration_months', 'months_to_maturity',
            'is_balloon_payment', 'is_interest_only'
        ]
        
        risk_scores = self.cox_model.predict_partial_hazard(last_obs[covariates])
        
        # Create separate histograms for each payment pattern
        payment_patterns = last_obs['payment_pattern'].unique()
        colors = ['green', 'red', 'orange']
        
        for i, pattern in enumerate(payment_patterns):
            pattern_mask = last_obs['payment_pattern'] == pattern
            if pattern_mask.sum() == 0:
                continue
                
            pattern_scores = risk_scores[pattern_mask]
            ax.hist(pattern_scores, bins=20, alpha=0.6, 
                   label=f'{pattern.title()} Payments', color=colors[i % len(colors)])
        
        ax.set_xlabel('Risk Score (Partial Hazard)')
        ax.set_ylabel('Frequency')
        ax.set_title('Risk Score Distribution by Payment Pattern')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_survival_curves_examples(self):
        """Plot survival curves for example risk profiles."""
        if self.cox_model is None:
            raise ValueError("Model must be fitted before plotting")
        
        # Define example profiles with all required features
        profiles = {
            'Low Risk': {
                'id': 'profile_low',
                'age': 0,  # Centered values (already scaled)
                'income': 1,  # Above average income
                'loan_amount': -1,  # Below average loan
                'credit_score': 1,  # Above average credit score
                'employment_years': 1,  # Stable employment
                'debt_to_income_ratio': -1,  # Low DTI
                'payment_history_score': 1,  # Good payment history
                'is_balloon_payment': 0,  # Installment payment
                'is_interest_only': 0,
                'loan_term_months': 36,
            },
            'Medium Risk': {
                'id': 'profile_medium',
                'age': 0,
                'income': 0,
                'loan_amount': 0,
                'credit_score': 0,
                'employment_years': 0,
                'debt_to_income_ratio': 0,
                'payment_history_score': 0,
                'is_balloon_payment': 0,  # Installment payment
                'is_interest_only': 0,
                'loan_term_months': 48,
            },
            'High Risk (Balloon)': {
                'id': 'profile_high',
                'age': 1,  # Older
                'income': -1,  # Lower income
                'loan_amount': 1,  # Higher loan
                'credit_score': -1,  # Lower credit score
                'employment_years': -1,  # Less stable employment
                'debt_to_income_ratio': 1,  # High DTI
                'payment_history_score': -1,  # Poor payment history
                'is_balloon_payment': 1,  # Balloon payment - HIGH RISK
                'is_interest_only': 0,
                'loan_term_months': 48,
            }
        }
        
        # Time horizon for prediction
        max_time = 60
        time_points = np.arange(1, max_time + 1)
        
        plt.figure(figsize=(12, 8))
        
        colors = ['green', 'orange', 'red']
        
        for i, (profile_name, profile_values) in enumerate(profiles.items()):
            survival_probs = []
            
            for t in time_points:
                # Create profile data for time t with all required features
                profile_data = pd.DataFrame([{
                    'start_time': 0,
                    'stop_time': t,
                    'loan_duration_months': t,
                    'months_to_maturity': max(0, profile_values['loan_term_months'] - t),
                    **profile_values
                }])
                
                # Predict survival
                survival_pred = self.predict_survival_probability(
                    profile_data, time_points=[t]
                )
                survival_probs.append(survival_pred[profile_values['id']][t])
            
            plt.plot(time_points, survival_probs, label=profile_name, 
                    color=colors[i], linewidth=3, alpha=0.8)
        
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        plt.title('Predicted Survival Curves for Different Risk Profiles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.show()
    
    def generate_model_report(self):
        """Generate a comprehensive model report."""
        if self.cox_model is None:
            raise ValueError("Model must be fitted before generating report")
        
        print("="*80)
        print("CREDIT SURVIVAL MODEL - COMPREHENSIVE REPORT")
        print("="*80)
        
        print("\n1. MODEL SUMMARY")
        print("-" * 40)
        self.cox_model.print_summary()
        
        print("\n\n2. PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Training C-index: {self.train_c_index:.4f}")
        if self.test_c_index:
            print(f"Test C-index: {self.test_c_index:.4f}")
        if self.val_c_index:
            print(f"Validation C-index: {self.val_c_index:.4f}")
        
        print(f"\nInterpretation:")
        print(f"- C-index > 0.7: Good discrimination")
        print(f"- C-index > 0.8: Excellent discrimination")
        print(f"- C-index = 0.5: No discrimination (random)")
        
        print("\n\n3. FEATURE INTERPRETATION")
        print("-" * 40)
        coeffs = self.cox_model.summary
        
        for feature, row in coeffs.iterrows():
            coef = row['coef']
            hr = np.exp(coef)
            p_val = row['p']
            
            direction = "increases" if coef > 0 else "decreases"
            significance = "significant" if p_val < 0.05 else "not significant"
            
            print(f"{feature}:")
            print(f"  - Coefficient: {coef:.4f}")
            print(f"  - Hazard Ratio: {hr:.4f}")
            print(f"  - Effect: {direction} default risk by {abs((hr-1)*100):.1f}% per unit increase")
            print(f"  - Significance: {significance} (p={p_val:.4f})")
            print()
        
        print("\n\n4. DATA SUMMARY")
        print("-" * 40)
        if self.train_data is not None:
            train_default_rate = self.train_data.groupby('id')['event'].max().mean()
            print(f"Training default rate: {train_default_rate:.3f}")
            print(f"Training subjects: {self.train_data['id'].nunique()}")
            print(f"Training observations: {len(self.train_data)}")
        
        if self.test_data is not None:
            test_default_rate = self.test_data.groupby('id')['event'].max().mean()
            print(f"Test default rate: {test_default_rate:.3f}")
            print(f"Test subjects: {self.test_data['id'].nunique()}")
        
        print("\n\n5. MODEL ASSUMPTIONS")
        print("-" * 40)
        print("Cox Proportional Hazards Model Assumptions:")
        print("1. Proportional hazards: Hazard ratios are constant over time")
        print("2. Log-linearity: Log hazard is linear in covariates")
        print("3. Independence: Observations are independent")
        print("\nNote: Validate these assumptions using residual plots and tests.")
    
    def analyze_payment_pattern_effects(self):
        """Analyze and visualize the effects of different payment patterns on default risk."""
        if self.cox_model is None:
            raise ValueError("Model must be fitted before analysis")
        
        print("\n" + "="*60)
        print("PAYMENT PATTERN ANALYSIS")
        print("="*60)
        
        # Analysis on training data
        train_prepared = self.prepare_features(self.train_data, fit_scaler=False)
        
        # Default rates by payment pattern
        print("\n1. Default Rates by Payment Pattern:")
        pattern_analysis = []
        
        for pattern in ['installment', 'balloon', 'interest_only']:
            pattern_data = train_prepared[train_prepared['payment_pattern'] == pattern]
            if len(pattern_data) > 0:
                default_rate = pattern_data.groupby('id')['event'].max().mean()
                n_loans = pattern_data['id'].nunique()
                avg_loan_amount = pattern_data.groupby('id')['loan_amount'].first().mean()
                avg_term = pattern_data.groupby('id')['loan_term_months'].first().mean()
                
                pattern_analysis.append({
                    'pattern': pattern,
                    'default_rate': default_rate,
                    'n_loans': n_loans,
                    'avg_loan_amount': avg_loan_amount,
                    'avg_term': avg_term
                })
                
                print(f"   {pattern.title()}: {default_rate:.3f} ({n_loans} loans)")
        
        # Hazard ratios from the model
        print("\n2. Model-Based Risk Effects (Hazard Ratios):")
        
        if 'is_balloon_payment' in self.cox_model.summary.index:
            balloon_coef = self.cox_model.summary.loc['is_balloon_payment', 'coef']
            balloon_hr = np.exp(balloon_coef)
            balloon_ci_lower = np.exp(self.cox_model.summary.loc['is_balloon_payment', 'coef lower 95%'])
            balloon_ci_upper = np.exp(self.cox_model.summary.loc['is_balloon_payment', 'coef upper 95%'])
            balloon_p = self.cox_model.summary.loc['is_balloon_payment', 'p']
            
            print(f"   Balloon vs Installment:")
            print(f"     Hazard Ratio: {balloon_hr:.3f} (95% CI: {balloon_ci_lower:.3f}-{balloon_ci_upper:.3f})")
            print(f"     Risk increase: {(balloon_hr-1)*100:.1f}%")
            print(f"     P-value: {balloon_p:.4f}")
        
        if 'is_interest_only' in self.cox_model.summary.index:
            io_coef = self.cox_model.summary.loc['is_interest_only', 'coef']
            io_hr = np.exp(io_coef)
            io_ci_lower = np.exp(self.cox_model.summary.loc['is_interest_only', 'coef lower 95%'])
            io_ci_upper = np.exp(self.cox_model.summary.loc['is_interest_only', 'coef upper 95%'])
            io_p = self.cox_model.summary.loc['is_interest_only', 'p']
            
            print(f"   Interest-Only vs Installment:")
            print(f"     Hazard Ratio: {io_hr:.3f} (95% CI: {io_ci_lower:.3f}-{io_ci_upper:.3f})")
            print(f"     Risk increase: {(io_hr-1)*100:.1f}%")
            print(f"     P-value: {io_p:.4f}")
        
        # Time-to-maturity effects
        if 'months_to_maturity' in self.cox_model.summary.index:
            maturity_coef = self.cox_model.summary.loc['months_to_maturity', 'coef']
            maturity_hr = np.exp(maturity_coef)
            maturity_p = self.cox_model.summary.loc['months_to_maturity', 'p']
            
            print(f"\n3. Time-to-Maturity Effect:")
            print(f"   Hazard Ratio per month closer to maturity: {maturity_hr:.3f}")
            print(f"   P-value: {maturity_p:.4f}")
            if maturity_hr < 1:
                print(f"   Interpretation: Risk decreases by {(1-maturity_hr)*100:.1f}% per month further from maturity")
            else:
                print(f"   Interpretation: Risk increases by {(maturity_hr-1)*100:.1f}% per month closer to maturity")
        
        # Create comprehensive payment pattern visualization
        self.plot_payment_pattern_analysis()
        
        return pattern_analysis
    
    def plot_payment_pattern_analysis(self):
        """Create comprehensive payment pattern analysis plots."""
        if self.test_data is None:
            print("No test data available for payment pattern analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Survival curves by payment pattern
        ax1 = axes[0, 0]
        test_prepared = self.prepare_features(self.test_data, fit_scaler=False)
        last_obs = test_prepared.groupby('id').last().reset_index()
        
        payment_patterns = ['installment', 'balloon', 'interest_only']
        colors = ['green', 'red', 'orange']
        
        for i, pattern in enumerate(payment_patterns):
            pattern_mask = last_obs['payment_pattern'] == pattern
            if pattern_mask.sum() > 5:
                group_data = last_obs[pattern_mask]
                kmf = KaplanMeierFitter()
                kmf.fit(group_data['stop_time'], group_data['event'], 
                       label=f'{pattern.title()} ({pattern_mask.sum()} loans)')
                kmf.plot_survival_function(ax=ax1, color=colors[i], linewidth=2)
        
        ax1.set_xlabel('Time (months)')
        ax1.set_ylabel('Survival Probability')
        ax1.set_title('Survival Curves by Payment Pattern')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Default rates by payment pattern
        ax2 = axes[0, 1]
        pattern_default_rates = []
        pattern_names = []
        
        for pattern in payment_patterns:
            pattern_data = test_prepared[test_prepared['payment_pattern'] == pattern]
            if len(pattern_data) > 0:
                default_rate = pattern_data.groupby('id')['event'].max().mean()
                pattern_default_rates.append(default_rate)
                pattern_names.append(pattern.title())
        
        bars = ax2.bar(pattern_names, pattern_default_rates, color=colors[:len(pattern_names)], alpha=0.7)
        ax2.set_ylabel('Default Rate')
        ax2.set_title('Default Rates by Payment Pattern')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, pattern_default_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom')
        
        # 3. Risk scores by payment pattern (box plot)
        ax3 = axes[1, 0]
        covariates = [
            'age', 'income', 'loan_amount', 'credit_score', 
            'employment_years', 'debt_to_income_ratio', 
            'payment_history_score', 'loan_duration_months', 'months_to_maturity',
            'is_balloon_payment', 'is_interest_only'
        ]
        
        risk_scores = self.cox_model.predict_partial_hazard(last_obs[covariates])
        
        risk_data = []
        risk_labels = []
        
        for pattern in payment_patterns:
            pattern_mask = last_obs['payment_pattern'] == pattern
            if pattern_mask.sum() > 0:
                risk_data.append(risk_scores[pattern_mask])
                risk_labels.append(pattern.title())
        
        box_plot = ax3.boxplot(risk_data, labels=risk_labels, patch_artist=True)
        
        # Color the box plots
        for patch, color in zip(box_plot['boxes'], colors[:len(risk_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Risk Score (Partial Hazard)')
        ax3.set_title('Risk Score Distribution by Payment Pattern')
        ax3.grid(True, alpha=0.3)
        
        # 4. Time-to-maturity vs default risk for balloon payments
        ax4 = axes[1, 1]
        balloon_data = test_prepared[test_prepared['payment_pattern'] == 'balloon']
        
        if len(balloon_data) > 10:
            # Group by months to maturity and calculate default rates
            maturity_groups = balloon_data.groupby('months_to_maturity')
            maturity_bins = []
            default_rates = []
            
            for months, group in maturity_groups:
                if len(group) >= 3:  # Minimum group size
                    default_rate = group.groupby('id')['event'].max().mean()
                    maturity_bins.append(months)
                    default_rates.append(default_rate)
            
            if maturity_bins:
                ax4.scatter(maturity_bins, default_rates, color='red', alpha=0.7, s=50)
                
                # Add trend line
                if len(maturity_bins) > 2:
                    z = np.polyfit(maturity_bins, default_rates, 1)
                    p = np.poly1d(z)
                    ax4.plot(maturity_bins, p(maturity_bins), "r--", alpha=0.8)
        
        ax4.set_xlabel('Months to Maturity')
        ax4.set_ylabel('Default Rate')
        ax4.set_title('Balloon Payment: Default Risk vs Time to Maturity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n4. Payment Pattern Insights:")
        print("   - Balloon payments typically show higher default risk")
        print("   - Risk may increase as balloon payment approaches")
        print("   - Interest-only loans often have moderate risk elevation")
        print("   - Installment loans generally have the lowest default risk")


def main():
    """Main execution function for the credit survival model."""
    print("Initializing Credit Survival Model with Payment Pattern Analysis...")
    
    # Create model instance
    model = CreditSurvivalModel(random_state=42)
    
    # Generate sample data
    train_data, test_data, val_data = model.generate_sample_data(
        n_subjects=1500, max_time=72, test_size=0.2, val_size=0.1
    )
    
    print("\nData overview:")
    print(train_data.head())
    print(f"\nFeature statistics (training data):")
    print(train_data.describe())
    
    # Fit the Cox model
    model.fit_cox_model(penalizer=0.01)
    
    # Payment pattern analysis
    print("\nPerforming payment pattern analysis...")
    pattern_analysis = model.analyze_payment_pattern_effects()
    
    # Generate comprehensive visualizations
    print("\nGenerating model visualizations...")
    model.plot_model_summary()
    
    # Plot survival curves for different risk profiles
    print("\nGenerating survival curve examples...")
    model.plot_survival_curves_examples()
    
    # Generate detailed report
    print("\nGenerating comprehensive model report...")
    model.generate_model_report()
    
    # Example prediction with payment patterns
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION WITH PAYMENT PATTERNS")
    print("="*60)
    
    # Create example customers with different payment patterns
    sample_customers = []
    for i, (pattern, pattern_name) in enumerate([
        ('installment', 'Installment Customer'),
        ('balloon', 'Balloon Payment Customer'),
        ('interest_only', 'Interest-Only Customer')
    ]):
        # Use actual feature values from test data
        sample_customer = test_data[test_data['payment_pattern'] == pattern].iloc[0:1].copy()
        sample_customer['id'] = f'example_{pattern}'
        sample_customers.append((sample_customer, pattern_name))
    
    print("\nSurvival probability predictions by payment pattern:")
    
    for sample_data, customer_name in sample_customers:
        if len(sample_data) > 0:
            survival_predictions = model.predict_survival_probability(
                sample_data, time_points=[6, 12, 24, 36, 48]
            )
            
            print(f"\n{customer_name}:")
            customer_id = sample_data['id'].iloc[0]
            pattern = sample_data['payment_pattern'].iloc[0]
            loan_term = sample_data['loan_term_months'].iloc[0]
            
            print(f"  Payment Pattern: {pattern}")
            print(f"  Loan Term: {loan_term} months")
            
            for time_point, prob in survival_predictions[customer_id].items():
                default_risk = (1 - prob) * 100
                print(f"    {time_point:2d} months: {prob:.3f} survival ({default_risk:.1f}% default risk)")
    
    # Payment pattern business insights
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS - PAYMENT PATTERNS")
    print("="*60)
    
    print("\n1. RISK RANKING:")
    print("   Highest Risk → Lowest Risk")
    print("   Balloon Payments > Interest-Only > Installment Payments")
    
    print("\n2. RISK MANAGEMENT RECOMMENDATIONS:")
    print("   • Balloon Payments:")
    print("     - Require higher down payments")
    print("     - More stringent credit score requirements") 
    print("     - Monitor closely as balloon payment approaches")
    print("     - Consider refinancing options before maturity")
    
    print("\n   • Interest-Only Loans:")
    print("     - Verify borrower's ability to handle principal payments")
    print("     - Regular income verification during interest-only period")
    print("     - Educational materials about payment shock")
    
    print("\n   • Installment Loans:")
    print("     - Lowest risk, standard underwriting acceptable")
    print("     - Focus on traditional credit metrics")
    
    print("\n3. PRICING IMPLICATIONS:")
    print("   • Risk-based pricing should reflect payment pattern")
    print("   • Balloon payments may warrant 100-200 bps premium")
    print("   • Interest-only may warrant 50-100 bps premium")
    
    print("\n" + "="*60)
    print("ENHANCED MODEL IMPLEMENTATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
