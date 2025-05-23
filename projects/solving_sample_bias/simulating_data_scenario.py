"""
Sample Selection Bias in Consumer Loan Decision Making

This script simulates how sample selection bias occurs in credit modeling when
loan rejection rates are high, and demonstrates techniques to improve model
performance by addressing rejected samples.

Key concepts demonstrated:
1. How selection bias affects model training when we only observe outcomes for approved loans
2. The impact of high rejection rates on model generalizability
3. Techniques to handle rejected samples and improve model power
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class LoanDataSimulator:
    """
    Simulates consumer loan data with realistic features and selection bias
    """
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
    def generate_population_data(self):
        """
        Generate synthetic population data with realistic loan applicant features
        """
        # Generate correlated features that affect loan default probability
        
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
        income_norm = np.log(df['annual_income'] / 30000) / 3  # Normalize around $30k
        
        # Create logistic function for default probability
        # Lower credit score, higher DTI, higher loan amount = higher default risk
        logit = (
            -3.0 +  # Base intercept (low default rate)
            -4.0 * credit_score_norm +  # Credit score effect (higher score = lower risk)
            3.0 * df['debt_to_income'] +  # DTI effect (higher DTI = higher risk)
            -0.5 * income_norm +  # Income effect (higher income = lower risk)
            0.3 * (df['loan_amount'] / 10000) +  # Loan amount effect
            -0.2 * (df['employment_length'] / 10) +  # Employment stability
            0.1 * np.maximum(0, df['open_credit_lines'] - 10)  # Too many credit lines
        )
        
        true_default_prob = 1 / (1 + np.exp(-logit))
        return true_default_prob
    
    def simulate_loan_approval_process(self, df, rejection_rate=0.4):
        """
        Simulate loan approval process that creates selection bias
        Banks typically reject high-risk applicants, creating sample selection bias
        """
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
        # Generate actual defaults based on true probabilities
        actual_defaults = np.random.binomial(1, true_default_prob, len(df))
        
        # Create observed dataset (only approved loans)
        observed_defaults = actual_defaults.astype(float)  # Convert to float to handle NaN
        observed_defaults[~approved] = np.nan  # We don't observe rejected loan outcomes
        
        return actual_defaults, observed_defaults

def demonstrate_selection_bias():
    """
    Main function to demonstrate sample selection bias and its solutions
    """
    print("=" * 80)
    print("SAMPLE SELECTION BIAS IN CONSUMER LOAN DECISIONS")
    print("=" * 80)
    
    # Generate data
    simulator = LoanDataSimulator(n_samples=10000)
    df = simulator.generate_population_data()
    
    # Calculate true default probabilities
    true_default_prob = simulator.calculate_true_default_probability(df)
    
    # Simulate different rejection rate scenarios
    rejection_rates = [0.2, 0.4, 0.6, 0.8]
    results = {}
    
    for rejection_rate in rejection_rates:
        print(f"\nAnalyzing scenario with {rejection_rate*100:.0f}% rejection rate...")
        
        # Simulate loan approval process
        approved, approval_score = simulator.simulate_loan_approval_process(df, rejection_rate)
        
        # Generate observed outcomes
        actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
            df, true_default_prob, approved
        )
        
        # Create datasets
        df_with_outcomes = df.copy()
        df_with_outcomes['true_default_prob'] = true_default_prob
        df_with_outcomes['actual_default'] = actual_defaults
        df_with_outcomes['approved'] = approved
        df_with_outcomes['observed_default'] = observed_defaults
        df_with_outcomes['approval_score'] = approval_score
        
        # Approved samples only (traditional approach)
        approved_data = df_with_outcomes[df_with_outcomes['approved']].copy()
        
        # Prepare features
        feature_cols = ['credit_score', 'annual_income', 'debt_to_income', 
                       'employment_length', 'loan_amount', 'open_credit_lines', 'age']
        
        X_full = df_with_outcomes[feature_cols]
        y_full = df_with_outcomes['actual_default']
        
        X_approved = approved_data[feature_cols]
        y_approved = approved_data['observed_default']
        
        # Split data
        X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
            X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
        )
        
        X_approved_train, X_approved_test, y_approved_train, y_approved_test = train_test_split(
            X_approved, y_approved, test_size=0.3, random_state=42, stratify=y_approved
        )
        
        # Scale features
        scaler_full = StandardScaler()
        scaler_approved = StandardScaler()
        
        X_full_train_scaled = scaler_full.fit_transform(X_full_train)
        X_full_test_scaled = scaler_full.transform(X_full_test)
        
        X_approved_train_scaled = scaler_approved.fit_transform(X_approved_train)
        X_approved_test_scaled = scaler_approved.transform(X_approved_test)
        
        # Train models
        # Model 1: Biased model (trained only on approved loans)
        model_biased = LogisticRegression(random_state=42)
        model_biased.fit(X_approved_train_scaled, y_approved_train)
        
        # Model 2: Unbiased model (trained on full population - hypothetical)
        model_unbiased = LogisticRegression(random_state=42)
        model_unbiased.fit(X_full_train_scaled, y_full_train)
        
        # Evaluate on full population (this is what we really care about)
        pred_biased_full = model_biased.predict_proba(
            scaler_approved.transform(X_full_test)
        )[:, 1]
        pred_unbiased_full = model_unbiased.predict_proba(X_full_test_scaled)[:, 1]
        
        # Calculate metrics
        auc_biased = roc_auc_score(y_full_test, pred_biased_full)
        auc_unbiased = roc_auc_score(y_full_test, pred_unbiased_full)
        
        # Store results
        results[rejection_rate] = {
            'approved_samples': approved.sum(),
            'total_samples': len(df),
            'approval_rate': approved.mean(),
            'default_rate_approved': approved_data['observed_default'].mean(),
            'default_rate_population': actual_defaults.mean(),
            'auc_biased': auc_biased,
            'auc_unbiased': auc_unbiased,
            'performance_gap': auc_unbiased - auc_biased
        }
        
        print(f"  Approval rate: {approved.mean():.1%}")
        print(f"  Default rate (approved only): {approved_data['observed_default'].mean():.1%}")
        print(f"  Default rate (full population): {actual_defaults.mean():.1%}")
        print(f"  Model AUC (biased): {auc_biased:.3f}")
        print(f"  Model AUC (unbiased): {auc_unbiased:.3f}")
        print(f"  Performance gap: {auc_unbiased - auc_biased:.3f}")
    
    return results, df_with_outcomes

def demonstrate_rejection_inference():
    """
    Demonstrate techniques to handle rejected samples and improve model performance
    """
    print("\n" + "=" * 80)
    print("TECHNIQUES TO HANDLE REJECTED SAMPLES")
    print("=" * 80)
    
    # Use moderate rejection rate scenario
    simulator = LoanDataSimulator(n_samples=10000)
    df = simulator.generate_population_data()
    true_default_prob = simulator.calculate_true_default_probability(df)
    approved, approval_score = simulator.simulate_loan_approval_process(df, rejection_rate=0.5)
    actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
        df, true_default_prob, approved
    )
    
    # Create full dataset
    df_full = df.copy()
    df_full['true_default_prob'] = true_default_prob
    df_full['actual_default'] = actual_defaults
    df_full['approved'] = approved
    df_full['observed_default'] = observed_defaults
    df_full['approval_score'] = approval_score
    
    feature_cols = ['credit_score', 'annual_income', 'debt_to_income', 
                   'employment_length', 'loan_amount', 'open_credit_lines', 'age']
    
    # Approved data only
    approved_data = df_full[df_full['approved']].copy()
    rejected_data = df_full[~df_full['approved']].copy()
    
    print(f"Total samples: {len(df_full)}")
    print(f"Approved samples: {len(approved_data)} ({len(approved_data)/len(df_full):.1%})")
    print(f"Rejected samples: {len(rejected_data)} ({len(rejected_data)/len(df_full):.1%})")
    
    # Method 1: Traditional approach (approved only)
    X_approved = approved_data[feature_cols]
    y_approved = approved_data['observed_default']
    
    # Method 2: Rejection inference - assign labels to rejected samples
    # Simple approach: assume rejected samples would have high default rate
    df_augmented = df_full.copy()
    
    # For rejected samples, we'll use a simple heuristic:
    # Those with very low approval scores get default=1, others get default=0
    rejection_threshold = np.percentile(rejected_data['approval_score'], 30)
    
    # Create augmented labels
    augmented_defaults = df_augmented['observed_default'].copy()
    rejected_mask = ~df_augmented['approved']
    
    # Assign high default probability to worst rejected applicants
    worst_rejected = (rejected_mask & 
                     (df_augmented['approval_score'] < rejection_threshold))
    augmented_defaults[worst_rejected] = 1
    
    # Assign low default probability to better rejected applicants
    better_rejected = (rejected_mask & 
                      (df_augmented['approval_score'] >= rejection_threshold))
    augmented_defaults[better_rejected] = 0
    
    df_augmented['augmented_default'] = augmented_defaults
    
    # Method 3: Propensity score weighting
    # Weight approved samples by inverse probability of approval
    approval_model = LogisticRegression(random_state=42)
    scaler_approval = StandardScaler()
    X_for_approval = scaler_approval.fit_transform(df_full[feature_cols])
    approval_model.fit(X_for_approval, df_full['approved'])
    
    # Calculate propensity scores (probability of approval)
    propensity_scores = approval_model.predict_proba(X_for_approval)[:, 1]
    df_full['propensity_score'] = propensity_scores
    
    # Create weights for approved samples (inverse propensity weighting)
    weights = np.where(df_full['approved'], 1/propensity_scores, 0)
    weights = weights / weights[df_full['approved']].mean()  # Normalize
    
    # Train models using different approaches
    X_full = df_full[feature_cols]
    y_full = df_full['actual_default']  # True outcomes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
    )
    
    # Get corresponding approved masks and augmented data for training
    train_indices = X_train.index
    test_indices = X_test.index
    
    approved_train_mask = df_full.loc[train_indices, 'approved']
    weights_train = weights[train_indices]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Biased (approved only)
    X_train_approved = X_train_scaled[approved_train_mask]
    y_train_approved = df_full.loc[train_indices[approved_train_mask], 'observed_default']
    
    model_biased = LogisticRegression(random_state=42)
    model_biased.fit(X_train_approved, y_train_approved)
    
    # Model 2: Rejection inference
    y_train_augmented = df_augmented.loc[train_indices, 'augmented_default']
    model_augmented = LogisticRegression(random_state=42)
    model_augmented.fit(X_train_scaled, y_train_augmented)
    
    # Model 3: Propensity score weighted (approved only but weighted)
    weights_approved = weights_train[approved_train_mask]
    model_weighted = LogisticRegression(random_state=42)
    model_weighted.fit(X_train_approved, y_train_approved, sample_weight=weights_approved)
    
    # Model 4: Oracle (full data - for comparison)
    model_oracle = LogisticRegression(random_state=42)
    model_oracle.fit(X_train_scaled, y_train)
    
    # Evaluate all models
    models = {
        'Biased (Approved Only)': model_biased,
        'Rejection Inference': model_augmented,
        'Propensity Weighted': model_weighted,
        'Oracle (Full Data)': model_oracle
    }
    
    print("\nModel Performance Comparison:")
    print("-" * 60)
    
    for name, model in models.items():
        if name == 'Biased (Approved Only)' or name == 'Propensity Weighted':
            # These models were trained on approved data, so we need to use 
            # the scaler fitted on approved data
            scaler_subset = StandardScaler()
            X_train_subset = X_train[approved_train_mask]
            scaler_subset.fit(X_train_subset)
            X_test_for_pred = scaler_subset.transform(X_test)
        else:
            X_test_for_pred = X_test_scaled
            
        pred_proba = model.predict_proba(X_test_for_pred)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        print(f"{name:25s}: AUC = {auc:.3f}")
    
    return df_full, models

def create_visualizations(results, df_with_outcomes):
    """
    Create visualizations to illustrate the sample selection bias problem
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sample Selection Bias in Consumer Loan Decisions', fontsize=16, fontweight='bold')
    
    # Plot 1: Rejection rate vs Performance gap
    rejection_rates = list(results.keys())
    performance_gaps = [results[r]['performance_gap'] for r in rejection_rates]
    
    axes[0, 0].plot(rejection_rates, performance_gaps, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Rejection Rate')
    axes[0, 0].set_ylabel('Performance Gap (AUC)')
    axes[0, 0].set_title('Impact of Rejection Rate on Model Bias')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Default rates comparison
    default_rates_approved = [results[r]['default_rate_approved'] for r in rejection_rates]
    default_rates_population = [results[r]['default_rate_population'] for r in rejection_rates]
    
    x = np.arange(len(rejection_rates))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, default_rates_approved, width, label='Approved Only', alpha=0.7)
    axes[0, 1].bar(x + width/2, default_rates_population, width, label='Full Population', alpha=0.7)
    axes[0, 1].set_xlabel('Rejection Rate')
    axes[0, 1].set_ylabel('Default Rate')
    axes[0, 1].set_title('Default Rates: Approved vs Population')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'{r:.0%}' for r in rejection_rates])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature distribution by approval status
    approved_data = df_with_outcomes[df_with_outcomes['approved']]
    rejected_data = df_with_outcomes[~df_with_outcomes['approved']]
    
    axes[0, 2].hist(approved_data['credit_score'], bins=30, alpha=0.7, label='Approved', density=True)
    axes[0, 2].hist(rejected_data['credit_score'], bins=30, alpha=0.7, label='Rejected', density=True)
    axes[0, 2].set_xlabel('Credit Score')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Credit Score Distribution by Approval Status')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: DTI distribution by approval status
    axes[1, 0].hist(approved_data['debt_to_income'], bins=30, alpha=0.7, label='Approved', density=True)
    axes[1, 0].hist(rejected_data['debt_to_income'], bins=30, alpha=0.7, label='Rejected', density=True)
    axes[1, 0].set_xlabel('Debt-to-Income Ratio')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('DTI Distribution by Approval Status')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: True default rate by approval score
    df_plot = df_with_outcomes.copy()
    df_plot['approval_score_bin'] = pd.qcut(df_plot['approval_score'], q=10, labels=False)
    bin_stats = df_plot.groupby('approval_score_bin').agg({
        'actual_default': 'mean',
        'approved': 'mean'
    }).reset_index()
    
    ax5 = axes[1, 1]
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(bin_stats['approval_score_bin'], bin_stats['actual_default'], 
                     'b-o', label='Default Rate')
    line2 = ax5_twin.plot(bin_stats['approval_score_bin'], bin_stats['approved'], 
                          'r-s', label='Approval Rate')
    
    ax5.set_xlabel('Approval Score Decile')
    ax5.set_ylabel('Default Rate', color='b')
    ax5_twin.set_ylabel('Approval Rate', color='r')
    ax5.set_title('Default vs Approval Rate by Score')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Plot 6: Model AUC comparison
    auc_biased = [results[r]['auc_biased'] for r in rejection_rates]
    auc_unbiased = [results[r]['auc_unbiased'] for r in rejection_rates]
    
    axes[1, 2].plot(rejection_rates, auc_biased, 'r-o', label='Biased Model', linewidth=2)
    axes[1, 2].plot(rejection_rates, auc_unbiased, 'g-s', label='Unbiased Model', linewidth=2)
    axes[1, 2].set_xlabel('Rejection Rate')
    axes[1, 2].set_ylabel('Model AUC')
    axes[1, 2].set_title('Model Performance vs Rejection Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_selection_bias_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Comment out show() for headless environment
    
    print("Visualizations saved as 'sample_selection_bias_analysis.png'")

def print_summary_insights():
    """
    Print key insights and recommendations
    """
    print("\n" + "=" * 80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 80)
    
    insights = [
        "1. SAMPLE SELECTION BIAS IMPACT:",
        "   • Higher rejection rates lead to larger performance gaps",
        "   • Models trained only on approved loans underestimate population risk",
        "   • Approved samples have systematically lower default rates",
        "",
        "2. WHY REJECTION INFERENCE MATTERS:",
        "   • Traditional models only see 'good' customers who got approved",
        "   • This creates blind spots for higher-risk segments",
        "   • Models may fail when applied to broader populations",
        "",
        "3. TECHNIQUES TO ADDRESS SELECTION BIAS:",
        "   • Rejection Inference: Assign estimated outcomes to rejected applicants",
        "   • Propensity Score Weighting: Weight approved samples by approval probability",
        "   • Semi-supervised Learning: Use features of rejected samples",
        "   • Domain Knowledge: Use business rules to estimate rejected outcomes",
        "",
        "4. BUSINESS IMPLICATIONS:",
        "   • Models may approve too many high-risk applicants if bias isn't addressed",
        "   • Portfolio performance may be worse than expected",
        "   • Regulatory compliance requires fair lending practices",
        "   • Better models lead to better risk management and profitability",
        "",
        "5. RECOMMENDATIONS:",
        "   • Always assess the degree of selection bias in your data",
        "   • Implement rejection inference techniques when rejection rates are high (>30%)",
        "   • Use holdout samples from rejected applicants when possible",
        "   • Monitor model performance on out-of-sample populations",
        "   • Consider the business cost of different types of errors"
    ]
    
    for insight in insights:
        print(insight)

def compare_model_coefficients():
    """
    Compare ground-truth model coefficients with fitted coefficients
    to demonstrate the value of rejection inference
    """
    print("\n" + "=" * 80)
    print("GROUND-TRUTH vs FITTED MODEL COEFFICIENTS COMPARISON")
    print("=" * 80)
    
    # Generate data with moderate rejection rate (50%)
    simulator = LoanDataSimulator(n_samples=20000)  # Larger sample for stability
    df = simulator.generate_population_data()
    true_default_prob = simulator.calculate_true_default_probability(df)
    approved, approval_score = simulator.simulate_loan_approval_process(df, rejection_rate=0.5)
    actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
        df, true_default_prob, approved
    )
    
    # Create full dataset
    df_full = df.copy()
    df_full['true_default_prob'] = true_default_prob
    df_full['actual_default'] = actual_defaults
    df_full['approved'] = approved
    df_full['observed_default'] = observed_defaults
    df_full['approval_score'] = approval_score
    
    # Define feature columns and create normalized features (same as ground truth)
    feature_cols = ['credit_score', 'annual_income', 'debt_to_income', 
                   'employment_length', 'loan_amount', 'open_credit_lines', 'age']
    
    # Create normalized features (matching the ground truth model)
    df_normalized = df_full.copy()
    df_normalized['credit_score_norm'] = (df_full['credit_score'] - 300) / (850 - 300)
    df_normalized['income_norm'] = np.log(df_full['annual_income'] / 30000) / 3
    df_normalized['loan_amount_scaled'] = df_full['loan_amount'] / 10000
    df_normalized['employment_scaled'] = df_full['employment_length'] / 10
    df_normalized['excess_credit_lines'] = np.maximum(0, df_full['open_credit_lines'] - 10)
    
    # Normalized feature columns for modeling
    norm_feature_cols = ['credit_score_norm', 'income_norm', 'debt_to_income', 
                        'employment_scaled', 'loan_amount_scaled', 'excess_credit_lines']
    
    # Ground truth coefficients (from calculate_true_default_probability)
    true_coefficients = {
        'intercept': -3.0,
        'credit_score_norm': -4.0,
        'debt_to_income': 3.0,
        'income_norm': -0.5,
        'loan_amount_scaled': 0.3,
        'employment_scaled': -0.2,
        'excess_credit_lines': 0.1
    }
    
    print(f"Sample size: {len(df_full):,}")
    print(f"Approved samples: {approved.sum():,} ({approved.mean():.1%})")
    print(f"Rejected samples: {(~approved).sum():,} ({(~approved).mean():.1%})")
    
    # Split data for training and testing
    train_idx, test_idx = train_test_split(
        range(len(df_normalized)), test_size=0.3, random_state=42, 
        stratify=df_normalized['actual_default']
    )
    
    # Prepare datasets for different modeling approaches
    # 1. Approved only (traditional biased approach)
    approved_train_idx = [i for i in train_idx if df_normalized.iloc[i]['approved']]
    
    X_approved_train = df_normalized.iloc[approved_train_idx][norm_feature_cols]
    y_approved_train = df_normalized.iloc[approved_train_idx]['observed_default']
    
    # 2. Full population (oracle - ground truth)
    X_full_train = df_normalized.iloc[train_idx][norm_feature_cols]
    y_full_train = df_normalized.iloc[train_idx]['actual_default']
    
    # 3. Rejection inference approach
    df_augmented = df_normalized.copy()
    
    # Simple rejection inference: assign outcomes based on approval score
    rejected_mask = ~df_augmented['approved']
    rejection_threshold = np.percentile(
        df_augmented[rejected_mask]['approval_score'], 30
    )
    
    augmented_defaults = df_augmented['observed_default'].copy()
    worst_rejected = rejected_mask & (df_augmented['approval_score'] < rejection_threshold)
    better_rejected = rejected_mask & (df_augmented['approval_score'] >= rejection_threshold)
    
    augmented_defaults[worst_rejected] = 1
    augmented_defaults[better_rejected] = 0
    
    X_augmented_train = df_augmented.iloc[train_idx][norm_feature_cols]
    y_augmented_train = augmented_defaults.iloc[train_idx]
    
    # 4. Propensity score weighted approach
    # Train approval model for propensity scores
    approval_model = LogisticRegression(random_state=42, max_iter=1000)
    approval_model.fit(X_full_train, df_normalized.iloc[train_idx]['approved'])
    propensity_scores = approval_model.predict_proba(X_full_train)[:, 1]
    
    # Create weights for approved samples only
    approved_train_mask = df_normalized.iloc[train_idx]['approved'].values
    weights_approved = 1 / propensity_scores[approved_train_mask]
    weights_approved = weights_approved / weights_approved.mean()  # Normalize
    
    # Train models
    models = {}
    
    # Model 1: Biased (approved only)
    model_biased = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
    model_biased.fit(X_approved_train, y_approved_train)
    models['Biased (Approved Only)'] = model_biased
    
    # Model 2: Oracle (full data)
    model_oracle = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
    model_oracle.fit(X_full_train, y_full_train)
    models['Oracle (Full Data)'] = model_oracle
    
    # Model 3: Rejection inference
    model_augmented = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
    model_augmented.fit(X_augmented_train, y_augmented_train)
    models['Rejection Inference'] = model_augmented
    
    # Model 4: Propensity score weighted (approved only but weighted)
    model_weighted = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
    model_weighted.fit(X_approved_train, y_approved_train, sample_weight=weights_approved)
    models['Propensity Weighted'] = model_weighted
    
    # Extract and compare coefficients
    print("\n" + "-" * 80)
    print("COEFFICIENT COMPARISON")
    print("-" * 80)
    
    # Create results dataframe
    coef_comparison = pd.DataFrame()
    coef_comparison['True_Coefficients'] = [
        true_coefficients['intercept'],
        true_coefficients['credit_score_norm'],
        true_coefficients['income_norm'],
        true_coefficients['debt_to_income'],
        true_coefficients['employment_scaled'],
        true_coefficients['loan_amount_scaled'],
        true_coefficients['excess_credit_lines']
    ]
    coef_comparison.index = ['Intercept'] + norm_feature_cols
    
    # Add fitted coefficients for each model
    for name, model in models.items():
        fitted_coefs = [model.intercept_[0]] + list(model.coef_[0])
        coef_comparison[name] = fitted_coefs
    
    # Calculate coefficient biases and errors
    bias_metrics = {}
    for name in models.keys():
        bias = coef_comparison[name] - coef_comparison['True_Coefficients']
        mse = np.mean(bias**2)
        mae = np.mean(np.abs(bias))
        max_abs_bias = np.max(np.abs(bias))
        
        bias_metrics[name] = {
            'MSE': mse,
            'MAE': mae,
            'Max_Abs_Bias': max_abs_bias
        }
    
    # Display results
    print("\nCoefficient Values:")
    print(coef_comparison.round(3))
    
    print("\nCoefficient Bias (Fitted - True):")
    bias_df = coef_comparison.copy()
    for name in models.keys():
        bias_df[f'{name}_Bias'] = coef_comparison[name] - coef_comparison['True_Coefficients']
    
    bias_cols = [col for col in bias_df.columns if '_Bias' in col]
    print(bias_df[bias_cols].round(3))
    
    print("\nOverall Bias Metrics:")
    bias_metrics_df = pd.DataFrame(bias_metrics).T
    print(bias_metrics_df.round(4))
    
    # Evaluate model performance on test set
    print("\n" + "-" * 80)
    print("MODEL PERFORMANCE ON TEST SET")
    print("-" * 80)
    
    X_test = df_normalized.iloc[test_idx][norm_feature_cols]
    y_test = df_normalized.iloc[test_idx]['actual_default']
    
    performance_results = {}
    for name, model in models.items():
        pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        performance_results[name] = auc
        print(f"{name:25s}: AUC = {auc:.4f}")
    
    return coef_comparison, bias_metrics_df, performance_results

def visualize_coefficient_comparison(coef_comparison, bias_metrics_df):
    """
    Create visualizations comparing coefficients and bias metrics
    """
    print("\n" + "=" * 80)
    print("CREATING COEFFICIENT COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ground-Truth vs Fitted Model Coefficients Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Coefficient values comparison
    ax1 = axes[0, 0]
    coef_plot_data = coef_comparison.copy()
    x_pos = np.arange(len(coef_plot_data))
    width = 0.15
    
    models_to_plot = ['Biased (Approved Only)', 'Rejection Inference', 'Propensity Weighted', 'Oracle (Full Data)']
    colors = ['red', 'blue', 'green', 'purple']
    
    for i, (model, color) in enumerate(zip(models_to_plot, colors)):
        offset = (i - 1.5) * width
        ax1.bar(x_pos + offset, coef_plot_data[model], width, 
                label=model, alpha=0.7, color=color)
    
    # Add true coefficients as black line
    ax1.plot(x_pos, coef_plot_data['True_Coefficients'], 'ko-', 
             linewidth=2, markersize=8, label='True Coefficients')
    
    ax1.set_xlabel('Coefficient')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('Coefficient Values: True vs Fitted')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(coef_plot_data.index, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient bias (error bars)
    ax2 = axes[0, 1]
    bias_data = {}
    for model in models_to_plot:
        bias_data[model] = coef_comparison[model] - coef_comparison['True_Coefficients']
    
    bias_df_plot = pd.DataFrame(bias_data)
    
    # Create heatmap of bias
    sns.heatmap(bias_df_plot.T, annot=True, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Coefficient Bias'}, ax=ax2)
    ax2.set_title('Coefficient Bias Heatmap (Fitted - True)')
    ax2.set_xlabel('Coefficient')
    ax2.set_ylabel('Model')
    
    # Plot 3: Overall bias metrics
    ax3 = axes[1, 0]
    x_metrics = np.arange(len(bias_metrics_df))
    width = 0.25
    
    metrics = ['MSE', 'MAE', 'Max_Abs_Bias']
    colors_metrics = ['orange', 'green', 'purple']
    
    for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
        offset = (i - 1) * width
        ax3.bar(x_metrics + offset, bias_metrics_df[metric], width, 
                label=metric, alpha=0.7, color=color)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Bias Metric Value')
    ax3.set_title('Overall Coefficient Bias Metrics')
    ax3.set_xticks(x_metrics)
    ax3.set_xticklabels(bias_metrics_df.index, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coefficient recovery comparison (scatter plot)
    ax4 = axes[1, 1]
    
    true_vals = coef_comparison['True_Coefficients']
    
    for model, color in zip(models_to_plot, colors):
        fitted_vals = coef_comparison[model]
        ax4.scatter(true_vals, fitted_vals, alpha=0.7, s=100, 
                   label=model, color=color)
    
    # Add perfect prediction line
    min_val = min(true_vals.min(), coef_comparison[models_to_plot].min().min())
    max_val = max(true_vals.max(), coef_comparison[models_to_plot].max().max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
             linewidth=2, label='Perfect Recovery')
    
    ax4.set_xlabel('True Coefficient Value')
    ax4.set_ylabel('Fitted Coefficient Value')
    ax4.set_title('Coefficient Recovery: True vs Fitted')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coefficient_comparison_analysis.png', dpi=300, bbox_inches='tight')
    
    print("Coefficient comparison visualizations saved as 'coefficient_comparison_analysis.png'")

def diagnose_rejection_inference_bias():
    """
    Diagnostic analysis to understand why rejection inference produces biased coefficients
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS: WHY REJECTION INFERENCE PRODUCES BIASED COEFFICIENTS")
    print("=" * 80)
    
    # Generate data for analysis
    simulator = LoanDataSimulator(n_samples=15000)
    df = simulator.generate_population_data()
    true_default_prob = simulator.calculate_true_default_probability(df)
    approved, approval_score = simulator.simulate_loan_approval_process(df, rejection_rate=0.5)
    actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
        df, true_default_prob, approved
    )
    
    # Create full dataset
    df_full = df.copy()
    df_full['true_default_prob'] = true_default_prob
    df_full['actual_default'] = actual_defaults
    df_full['approved'] = approved
    df_full['observed_default'] = observed_defaults
    df_full['approval_score'] = approval_score
    
    # Analyze the rejection inference label assignment
    rejected_data = df_full[~df_full['approved']].copy()
    
    # Apply the same rejection inference logic as in the main analysis
    rejection_threshold = np.percentile(rejected_data['approval_score'], 30)
    
    print(f"Total rejected samples: {len(rejected_data):,}")
    print(f"Rejection threshold (30th percentile): {rejection_threshold:.3f}")
    
    # Assign labels
    worst_rejected = rejected_data['approval_score'] < rejection_threshold
    better_rejected = rejected_data['approval_score'] >= rejection_threshold
    
    print(f"Assigned default=1 (worst rejected): {worst_rejected.sum():,} ({worst_rejected.mean():.1%})")
    print(f"Assigned default=0 (better rejected): {better_rejected.sum():,} ({better_rejected.mean():.1%})")
    
    # Compare assigned labels with true outcomes
    worst_rejected_true_default_rate = rejected_data[worst_rejected]['actual_default'].mean()
    better_rejected_true_default_rate = rejected_data[better_rejected]['actual_default'].mean()
    overall_rejected_true_default_rate = rejected_data['actual_default'].mean()
    
    print(f"\nTRUE DEFAULT RATES IN REJECTED POPULATION:")
    print(f"Worst rejected (assigned 1): {worst_rejected_true_default_rate:.1%} true default rate")
    print(f"Better rejected (assigned 0): {better_rejected_true_default_rate:.1%} true default rate")
    print(f"Overall rejected: {overall_rejected_true_default_rate:.1%} true default rate")
    
    # Calculate label assignment accuracy
    assigned_labels = np.where(worst_rejected, 1, 0)
    true_labels = rejected_data['actual_default'].values
    
    accuracy = (assigned_labels == true_labels).mean()
    precision_1 = (assigned_labels & true_labels).sum() / assigned_labels.sum() if assigned_labels.sum() > 0 else 0
    recall_1 = (assigned_labels & true_labels).sum() / true_labels.sum() if true_labels.sum() > 0 else 0
    
    print(f"\nLABEL ASSIGNMENT QUALITY:")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"Precision (assigned=1): {precision_1:.1%}")
    print(f"Recall (assigned=1): {recall_1:.1%}")
    
    # Analyze the distribution of assigned vs true labels
    print(f"\nLABEL DISTRIBUTION COMPARISON:")
    print(f"Assigned label distribution: {assigned_labels.mean():.1%} default rate")
    print(f"True label distribution: {true_labels.mean():.1%} default rate")
    print(f"Bias in default rate: {assigned_labels.mean() - true_labels.mean():.1%} percentage points")
    
    # Analyze feature distributions in different groups
    feature_cols = ['credit_score', 'debt_to_income', 'annual_income']
    
    print(f"\nFEATURE DISTRIBUTION ANALYSIS:")
    print("-" * 60)
    
    # Compare approved vs rejected vs artificial groups
    approved_data = df_full[df_full['approved']]
    
    groups = {
        'Approved (Real Labels)': approved_data,
        'Rejected - Assigned Default=1': rejected_data[worst_rejected],
        'Rejected - Assigned Default=0': rejected_data[better_rejected],
        'All Rejected (True)': rejected_data
    }
    
    for feature in feature_cols:
        print(f"\n{feature.upper()}:")
        for group_name, group_data in groups.items():
            mean_val = group_data[feature].mean()
            print(f"  {group_name:25s}: {mean_val:8.1f}")
    
    # Create augmented dataset and analyze coefficient inflation
    df_augmented = df_full.copy()
    augmented_defaults = df_augmented['observed_default'].copy()
    
    rejected_mask = ~df_augmented['approved']
    worst_rejected_mask = rejected_mask & (df_augmented['approval_score'] < rejection_threshold)
    better_rejected_mask = rejected_mask & (df_augmented['approval_score'] >= rejection_threshold)
    
    augmented_defaults[worst_rejected_mask] = 1
    augmented_defaults[better_rejected_mask] = 0
    
    # Analyze the extreme cases created by rejection inference
    print(f"\nEXTREME CASES ANALYSIS:")
    print("-" * 60)
    
    # Find samples with most extreme features
    worst_rejected_samples = df_augmented[worst_rejected_mask]
    if len(worst_rejected_samples) > 0:
        print(f"Samples assigned default=1 (worst rejected): {len(worst_rejected_samples):,}")
        print(f"  Mean credit score: {worst_rejected_samples['credit_score'].mean():.1f}")
        print(f"  Mean DTI: {worst_rejected_samples['debt_to_income'].mean():.3f}")
        print(f"  Mean income: ${worst_rejected_samples['annual_income'].mean():,.0f}")
        
        # Compare to approved defaulters
        approved_defaulters = approved_data[approved_data['observed_default'] == 1]
        if len(approved_defaulters) > 0:
            print(f"\nApproved samples that actually defaulted: {len(approved_defaulters):,}")
            print(f"  Mean credit score: {approved_defaulters['credit_score'].mean():.1f}")
            print(f"  Mean DTI: {approved_defaulters['debt_to_income'].mean():.3f}")
            print(f"  Mean income: ${approved_defaulters['annual_income'].mean():,.0f}")
            
            print(f"\nComparison (Worst Rejected vs Approved Defaulters):")
            print(f"  Credit score difference: {worst_rejected_samples['credit_score'].mean() - approved_defaulters['credit_score'].mean():.1f}")
            print(f"  DTI difference: {worst_rejected_samples['debt_to_income'].mean() - approved_defaulters['debt_to_income'].mean():.3f}")
    
    # Explain why coefficients get inflated
    print(f"\nWHY COEFFICIENTS GET INFLATED:")
    print("-" * 60)
    print("1. ARTIFICIAL SEPARATION: Rejection inference creates artificial extreme groups")
    print("2. FEATURE AMPLIFICATION: Assigned defaults have more extreme features than real defaults")
    print("3. MODEL OVERFITTING: Logistic regression tries to perfectly separate these artificial groups")
    print("4. COEFFICIENT INFLATION: Large coefficients needed to achieve artificial separation")
    
    return df_augmented, worst_rejected_mask, better_rejected_mask

def compare_coefficient_stability():
    """
    Compare coefficient stability across different rejection inference strategies
    """
    print("\n" + "=" * 80)
    print("COEFFICIENT STABILITY ACROSS REJECTION INFERENCE STRATEGIES")
    print("=" * 80)
    
    # Test different rejection inference strategies
    strategies = {
        'Conservative (40% default)': 0.4,  # Assign default to top 40% worst rejected
        'Moderate (30% default)': 0.3,      # Current approach
        'Aggressive (20% default)': 0.2,    # Assign default to top 20% worst rejected
        'Random (50% default)': 0.5         # Random assignment for comparison
    }
    
    # Generate base data
    simulator = LoanDataSimulator(n_samples=15000)
    df = simulator.generate_population_data()
    true_default_prob = simulator.calculate_true_default_probability(df)
    approved, approval_score = simulator.simulate_loan_approval_process(df, rejection_rate=0.5)
    actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
        df, true_default_prob, approved
    )
    
    df_full = df.copy()
    df_full['true_default_prob'] = true_default_prob
    df_full['actual_default'] = actual_defaults
    df_full['approved'] = approved
    df_full['observed_default'] = observed_defaults
    df_full['approval_score'] = approval_score
    
    # Normalized features
    df_normalized = df_full.copy()
    df_normalized['credit_score_norm'] = (df_full['credit_score'] - 300) / (850 - 300)
    df_normalized['income_norm'] = np.log(df_full['annual_income'] / 30000) / 3
    df_normalized['loan_amount_scaled'] = df_full['loan_amount'] / 10000
    df_normalized['employment_scaled'] = df_full['employment_length'] / 10
    df_normalized['excess_credit_lines'] = np.maximum(0, df_full['open_credit_lines'] - 10)
    
    norm_feature_cols = ['credit_score_norm', 'income_norm', 'debt_to_income', 
                        'employment_scaled', 'loan_amount_scaled', 'excess_credit_lines']
    
    # True coefficients
    true_coefficients = [-3.0, -4.0, -0.5, 3.0, -0.2, 0.3, 0.1]
    feature_names = ['Intercept'] + norm_feature_cols
    
    results = {}
    
    for strategy_name, threshold_percentile in strategies.items():
        print(f"\nTesting strategy: {strategy_name}")
        
        # Create augmented dataset
        df_augmented = df_normalized.copy()
        rejected_mask = ~df_augmented['approved']
        rejected_data = df_augmented[rejected_mask]
        
        if strategy_name == 'Random (50% default)':
            # Random assignment
            np.random.seed(42)
            assigned_defaults = np.random.binomial(1, 0.5, len(rejected_data))
        else:
            # Threshold-based assignment
            threshold = np.percentile(rejected_data['approval_score'], threshold_percentile * 100)
            assigned_defaults = (rejected_data['approval_score'] < threshold).astype(int)
        
        # Update augmented labels
        augmented_defaults = df_augmented['observed_default'].copy()
        augmented_defaults[rejected_mask] = assigned_defaults
        
        # Train model
        train_mask = np.random.choice(len(df_augmented), size=int(0.7 * len(df_augmented)), replace=False)
        X_train = df_augmented.iloc[train_mask][norm_feature_cols]
        y_train = augmented_defaults.iloc[train_mask]
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Extract coefficients
        fitted_coefs = [model.intercept_[0]] + list(model.coef_[0])
        
        # Calculate bias metrics
        bias = np.array(fitted_coefs) - np.array(true_coefficients)
        mse = np.mean(bias**2)
        mae = np.mean(np.abs(bias))
        max_abs_bias = np.max(np.abs(bias))
        
        results[strategy_name] = {
            'coefficients': fitted_coefs,
            'bias': bias,
            'mse': mse,
            'mae': mae,
            'max_abs_bias': max_abs_bias,
            'assigned_default_rate': assigned_defaults.mean()
        }
        
        print(f"  Assigned default rate: {assigned_defaults.mean():.1%}")
        print(f"  Coefficient MAE: {mae:.3f}")
        print(f"  Max absolute bias: {max_abs_bias:.3f}")
    
    # Summary comparison
    print(f"\nSTRATEGY COMPARISON:")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Assigned Default Rate':<20} {'MAE':<10} {'Max Bias':<10}")
    print("-" * 80)
    
    for strategy_name, result in results.items():
        print(f"{strategy_name:<25} {result['assigned_default_rate']:<20.1%} {result['mae']:<10.3f} {result['max_abs_bias']:<10.3f}")
    
    return results

def interpretability_first_analysis():
    """
    Analysis focused on model interpretability rather than prediction performance
    """
    print("\n" + "=" * 80)
    print("INTERPRETABILITY-FIRST APPROACH TO SAMPLE SELECTION BIAS")
    print("=" * 80)
    
    # Generate data
    simulator = LoanDataSimulator(n_samples=20000)
    df = simulator.generate_population_data()
    true_default_prob = simulator.calculate_true_default_probability(df)
    approved, approval_score = simulator.simulate_loan_approval_process(df, rejection_rate=0.5)
    actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
        df, true_default_prob, approved
    )
    
    # Create full dataset
    df_full = df.copy()
    df_full['true_default_prob'] = true_default_prob
    df_full['actual_default'] = actual_defaults
    df_full['approved'] = approved
    df_full['observed_default'] = observed_defaults
    df_full['approval_score'] = approval_score
    
    # Normalized features
    df_normalized = df_full.copy()
    df_normalized['credit_score_norm'] = (df_full['credit_score'] - 300) / (850 - 300)
    df_normalized['income_norm'] = np.log(df_full['annual_income'] / 30000) / 3
    df_normalized['loan_amount_scaled'] = df_full['loan_amount'] / 10000
    df_normalized['employment_scaled'] = df_full['employment_length'] / 10
    df_normalized['excess_credit_lines'] = np.maximum(0, df_full['open_credit_lines'] - 10)
    
    norm_feature_cols = ['credit_score_norm', 'income_norm', 'debt_to_income', 
                        'employment_scaled', 'loan_amount_scaled', 'excess_credit_lines']
    
    # True coefficients for comparison
    true_coefficients = [-3.0, -4.0, -0.5, 3.0, -0.2, 0.3, 0.1]
    feature_names = ['Intercept'] + norm_feature_cols
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(df_normalized)), test_size=0.3, random_state=42, 
        stratify=df_normalized['actual_default']
    )
    
    # Interpretability-focused approaches
    approaches = {}
    
    print("Testing interpretability-focused approaches...")
    
    # 1. Approved Only (Baseline - traditional approach)
    approved_train_idx = [i for i in train_idx if df_normalized.iloc[i]['approved']]
    X_approved_train = df_normalized.iloc[approved_train_idx][norm_feature_cols]
    y_approved_train = df_normalized.iloc[approved_train_idx]['observed_default']
    
    model_baseline = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    model_baseline.fit(X_approved_train, y_approved_train)
    approaches['Baseline (Approved Only)'] = model_baseline
    
    # 2. Regularized Approved Only (reduce overfitting)
    model_regularized = LogisticRegression(random_state=42, max_iter=1000, C=0.1)  # Strong regularization
    model_regularized.fit(X_approved_train, y_approved_train)
    approaches['Regularized (C=0.1)'] = model_regularized
    
    # 3. Conservative Propensity Weighting
    X_full_train = df_normalized.iloc[train_idx][norm_feature_cols]
    approval_model = LogisticRegression(random_state=42, max_iter=1000)
    approval_model.fit(X_full_train, df_normalized.iloc[train_idx]['approved'])
    propensity_scores = approval_model.predict_proba(X_full_train)[:, 1]
    
    # Conservative weighting: cap extreme weights
    raw_weights = 1 / propensity_scores[df_normalized.iloc[train_idx]['approved']]
    # Cap weights at 95th percentile to avoid extreme cases
    weight_cap = np.percentile(raw_weights, 95)
    capped_weights = np.minimum(raw_weights, weight_cap)
    capped_weights = capped_weights / capped_weights.mean()
    
    model_conservative_weighted = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    model_conservative_weighted.fit(X_approved_train, y_approved_train, sample_weight=capped_weights)
    approaches['Conservative Propensity (Capped)'] = model_conservative_weighted
    
    # 4. Stability-focused approach: Ensemble of subsamples
    # Train multiple models on different subsamples and average coefficients
    n_models = 10
    subsample_models = []
    
    for i in range(n_models):
        # Random subsample of approved data
        subsample_idx = np.random.choice(
            approved_train_idx, 
            size=int(0.8 * len(approved_train_idx)), 
            replace=False
        )
        X_subsample = df_normalized.iloc[subsample_idx][norm_feature_cols]
        y_subsample = df_normalized.iloc[subsample_idx]['observed_default']
        
        model_sub = LogisticRegression(random_state=42+i, max_iter=1000, C=0.5)
        model_sub.fit(X_subsample, y_subsample)
        subsample_models.append(model_sub)
    
    # Average coefficients
    avg_intercept = np.mean([m.intercept_[0] for m in subsample_models])
    avg_coefs = np.mean([m.coef_[0] for m in subsample_models], axis=0)
    
    # Create ensemble model (for evaluation purposes)
    model_ensemble = LogisticRegression(random_state=42, max_iter=1000)
    model_ensemble.fit(X_approved_train, y_approved_train)  # Fit on full data
    # Manually set averaged coefficients
    model_ensemble.intercept_ = np.array([avg_intercept])
    model_ensemble.coef_ = np.array([avg_coefs])
    approaches['Ensemble Averaging'] = model_ensemble
    
    # 5. Domain-constrained approach: Add coefficient bounds
    # This would typically require specialized optimization, but we'll simulate with regularization
    model_constrained = LogisticRegression(random_state=42, max_iter=1000, C=0.5, penalty='l1', solver='liblinear')
    model_constrained.fit(X_approved_train, y_approved_train)
    approaches['L1 Regularized (Sparse)'] = model_constrained
    
    # Evaluate all approaches
    print("\n" + "-" * 80)
    print("INTERPRETABILITY ANALYSIS RESULTS")
    print("-" * 80)
    
    results = {}
    
    # Test set evaluation
    X_test = df_normalized.iloc[test_idx][norm_feature_cols]
    y_test = df_normalized.iloc[test_idx]['actual_default']
    
    print(f"\n{'Approach':<30} {'Coef MAE':<12} {'Max |Bias|':<12} {'AUC':<8} {'Interpretability Score':<20}")
    print("-" * 90)
    
    for name, model in approaches.items():
        # Extract coefficients
        fitted_coefs = [model.intercept_[0]] + list(model.coef_[0])
        
        # Calculate coefficient bias
        bias = np.array(fitted_coefs) - np.array(true_coefficients)
        mae = np.mean(np.abs(bias))
        max_abs_bias = np.max(np.abs(bias))
        
        # Calculate AUC
        pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        
        # Interpretability score (lower coefficient bias = higher interpretability)
        # Score from 0-100, where 100 = perfect coefficient recovery
        max_possible_mae = 10.0  # Reasonable upper bound
        interpretability_score = max(0, 100 * (1 - mae / max_possible_mae))
        
        results[name] = {
            'coefficients': fitted_coefs,
            'mae': mae,
            'max_abs_bias': max_abs_bias,
            'auc': auc,
            'interpretability_score': interpretability_score
        }
        
        print(f"{name:<30} {mae:<12.3f} {max_abs_bias:<12.3f} {auc:<8.3f} {interpretability_score:<20.1f}")
    
    # Detailed coefficient comparison
    print(f"\nDETAILED COEFFICIENT COMPARISON:")
    print("-" * 80)
    
    coef_df = pd.DataFrame()
    coef_df['True_Coefficients'] = true_coefficients
    coef_df.index = feature_names
    
    for name, result in results.items():
        coef_df[name] = result['coefficients']
    
    print(coef_df.round(3))
    
    # Recommendations based on interpretability priority
    print(f"\nINTERPRETABILITY-FIRST RECOMMENDATIONS:")
    print("-" * 80)
    
    # Find best interpretability approach
    best_interpretability = min(results.items(), key=lambda x: x[1]['mae'])
    best_balance = min(results.items(), key=lambda x: x[1]['mae'] + (0.1 / x[1]['auc']))  # Weighted score
    
    print(f"1. BEST COEFFICIENT ACCURACY: {best_interpretability[0]}")
    print(f"   - Coefficient MAE: {best_interpretability[1]['mae']:.3f}")
    print(f"   - AUC: {best_interpretability[1]['auc']:.3f}")
    print(f"   - Interpretability Score: {best_interpretability[1]['interpretability_score']:.1f}/100")
    
    print(f"\n2. BEST BALANCE (Accuracy + Performance): {best_balance[0]}")
    print(f"   - Coefficient MAE: {best_balance[1]['mae']:.3f}")
    print(f"   - AUC: {best_balance[1]['auc']:.3f}")
    print(f"   - Interpretability Score: {best_balance[1]['interpretability_score']:.1f}/100")
    
    # Practical guidance
    print(f"\nPRACTICAL GUIDANCE FOR INTERPRETABILITY-FIRST MODELING:")
    print("-" * 80)
    guidance = [
        "1. ACCEPT LOWER AUC: Prioritize coefficient accuracy over prediction performance",
        "2. USE REGULARIZATION: L1/L2 regularization prevents extreme coefficient values",
        "3. CAP PROPENSITY WEIGHTS: Avoid extreme weights that distort coefficients",
        "4. ENSEMBLE AVERAGING: Average coefficients across multiple subsamples for stability",
        "5. VALIDATE COEFFICIENTS: Always compare fitted coefficients to business intuition",
        "6. DOCUMENT TRADE-OFFS: Clearly communicate the interpretability vs performance trade-off",
        "7. CONSIDER DOMAIN CONSTRAINTS: Use business knowledge to constrain coefficient ranges",
        "8. MONITOR COEFFICIENT DRIFT: Track coefficient stability over time"
    ]
    
    for point in guidance:
        print(f"   {point}")
    
    return results, coef_df

def create_interpretability_recommendations():
    """
    Create specific recommendations for interpretability-focused modeling
    """
    print("\n" + "=" * 80)
    print("INTERPRETABILITY-FOCUSED MODELING STRATEGY")
    print("=" * 80)
    
    strategies = {
        "CONSERVATIVE APPROACH": {
            "description": "Minimize coefficient bias at the cost of some performance",
            "methods": [
                "Use only approved samples with strong regularization (C=0.1-0.5)",
                "Cap propensity weights at 95th percentile",
                "Ensemble averaging across subsamples",
                "L1 regularization for feature selection"
            ],
            "pros": [
                "Coefficients closest to true relationships",
                "Stable and interpretable model",
                "Regulatory-friendly explanations"
            ],
            "cons": [
                "Lower AUC performance",
                "May miss some predictive power",
                "Conservative risk assessment"
            ],
            "use_when": "Regulatory compliance, model explainability, and stakeholder trust are critical"
        },
        
        "BALANCED APPROACH": {
            "description": "Moderate trade-off between interpretability and performance",
            "methods": [
                "Conservative propensity weighting with capped weights",
                "Moderate regularization (C=0.5-1.0)",
                "Cross-validation for coefficient stability",
                "Feature importance validation"
            ],
            "pros": [
                "Reasonable coefficient accuracy",
                "Acceptable performance",
                "Good business interpretability"
            ],
            "cons": [
                "Some coefficient bias remains",
                "Requires careful tuning",
                "May need frequent recalibration"
            ],
            "use_when": "Need both interpretability and competitive performance"
        },
        
        "HYBRID APPROACH": {
            "description": "Use different models for different purposes",
            "methods": [
                "Interpretable model for coefficient analysis and reporting",
                "High-performance model for actual scoring",
                "Regular coefficient comparison and validation",
                "Separate model governance processes"
            ],
            "pros": [
                "Best of both worlds",
                "Flexibility in model usage",
                "Clear separation of concerns"
            ],
            "cons": [
                "Complexity in model management",
                "Potential stakeholder confusion",
                "Higher maintenance overhead"
            ],
            "use_when": "Organization needs both interpretable insights and competitive performance"
        }
    }
    
    for strategy_name, strategy_info in strategies.items():
        print(f"\n{strategy_name}:")
        print(f"Description: {strategy_info['description']}")
        
        print(f"\nMethods:")
        for method in strategy_info['methods']:
            print(f"  • {method}")
        
        print(f"\nPros:")
        for pro in strategy_info['pros']:
            print(f"  ✓ {pro}")
        
        print(f"\nCons:")
        for con in strategy_info['cons']:
            print(f"  ✗ {con}")
        
        print(f"\nUse When: {strategy_info['use_when']}")
        print("-" * 60)

def simulate_external_predictor_rejection_inference():
    """
    Simulate the case where an external powerful predictor is available
    to help reclassify rejected samples and approximate ground truth labels
    """
    print("\n" + "=" * 80)
    print("EXTERNAL PREDICTOR-ENHANCED REJECTION INFERENCE")
    print("=" * 80)
    
    # Generate base data
    simulator = LoanDataSimulator(n_samples=20000)
    df = simulator.generate_population_data()
    true_default_prob = simulator.calculate_true_default_probability(df)
    approved, approval_score = simulator.simulate_loan_approval_process(df, rejection_rate=0.5)
    actual_defaults, observed_defaults = simulator.generate_observed_outcomes(
        df, true_default_prob, approved
    )
    
    # Create full dataset
    df_full = df.copy()
    df_full['true_default_prob'] = true_default_prob
    df_full['actual_default'] = actual_defaults
    df_full['approved'] = approved
    df_full['observed_default'] = observed_defaults
    df_full['approval_score'] = approval_score
    
    # SIMULATE EXTERNAL PREDICTOR
    # This represents external data sources like:
    # - Alternative credit data (utility payments, rent history)
    # - Banking transaction patterns
    # - Employment verification data
    # - Social media/digital footprint scores
    print("Simulating external predictor (e.g., alternative credit data)...")
    
    # External predictor that's:
    # 1. Correlated with true default probability
    # 2. Partially independent of approval score (has unique information)
    # 3. Available for both approved and rejected samples
    
    # Create external predictor score
    # Base it on true default probability with some noise and different weighting
    external_signal = (
        0.6 * true_default_prob +  # 60% correlation with true default risk
        0.2 * (1 - (df_full['credit_score'] - 300) / 550) +  # Some correlation with credit score
        0.1 * df_full['debt_to_income'] +  # Some correlation with DTI
        0.1 * np.random.normal(0, 0.2, len(df_full))  # 10% random noise
    )
    
    # Normalize to 0-1 scale
    external_predictor = (external_signal - external_signal.min()) / (external_signal.max() - external_signal.min())
    df_full['external_predictor'] = external_predictor
    
    print(f"External predictor correlation with true default: {np.corrcoef(external_predictor, actual_defaults)[0,1]:.3f}")
    print(f"External predictor correlation with approval score: {np.corrcoef(external_predictor, approval_score)[0,1]:.3f}")
    
    # Normalized features for modeling
    df_normalized = df_full.copy()
    df_normalized['credit_score_norm'] = (df_full['credit_score'] - 300) / (850 - 300)
    df_normalized['income_norm'] = np.log(df_full['annual_income'] / 30000) / 3
    df_normalized['loan_amount_scaled'] = df_full['loan_amount'] / 10000
    df_normalized['employment_scaled'] = df_full['employment_length'] / 10
    df_normalized['excess_credit_lines'] = np.maximum(0, df_full['open_credit_lines'] - 10)
    
    norm_feature_cols = ['credit_score_norm', 'income_norm', 'debt_to_income', 
                        'employment_scaled', 'loan_amount_scaled', 'excess_credit_lines']
    
    # True coefficients
    true_coefficients = [-3.0, -4.0, -0.5, 3.0, -0.2, 0.3, 0.1]
    feature_names = ['Intercept'] + norm_feature_cols
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(df_normalized)), test_size=0.3, random_state=42, 
        stratify=df_normalized['actual_default']
    )
    
    # Test different rejection inference strategies
    strategies = {}
    
    print("\nTesting rejection inference strategies...")
    
    # Strategy 1: Baseline (Approved Only)
    approved_train_idx = [i for i in train_idx if df_normalized.iloc[i]['approved']]
    X_approved_train = df_normalized.iloc[approved_train_idx][norm_feature_cols]
    y_approved_train = df_normalized.iloc[approved_train_idx]['observed_default']
    
    model_baseline = LogisticRegression(random_state=42, max_iter=1000)
    model_baseline.fit(X_approved_train, y_approved_train)
    strategies['Baseline (Approved Only)'] = model_baseline
    
    # Strategy 2: Naive Rejection Inference (Approval Score Based)
    rejected_data = df_normalized[~df_normalized['approved']]
    rejection_threshold = np.percentile(rejected_data['approval_score'], 30)
    
    df_naive = df_normalized.copy()
    augmented_defaults_naive = df_naive['observed_default'].copy()
    rejected_mask = ~df_naive['approved']
    worst_rejected = rejected_mask & (df_naive['approval_score'] < rejection_threshold)
    better_rejected = rejected_mask & (df_naive['approval_score'] >= rejection_threshold)
    
    augmented_defaults_naive[worst_rejected] = 1
    augmented_defaults_naive[better_rejected] = 0
    
    X_train_naive = df_naive.iloc[train_idx][norm_feature_cols]
    y_train_naive = augmented_defaults_naive.iloc[train_idx]
    
    model_naive = LogisticRegression(random_state=42, max_iter=1000)
    model_naive.fit(X_train_naive, y_train_naive)
    strategies['Naive Rejection Inference'] = model_naive
    
    # Strategy 3: External Predictor Enhanced Rejection Inference
    print("Implementing external predictor enhanced rejection inference...")
    
    # Use external predictor to assign labels to rejected samples
    rejected_external_scores = df_normalized[rejected_mask]['external_predictor']
    
    # Set threshold based on external predictor to match true default rate better
    # Use multiple thresholds to create more nuanced labels
    
    # Method 3a: External predictor threshold
    external_threshold_high = np.percentile(rejected_external_scores, 85)  # Top 15% get default=1
    external_threshold_low = np.percentile(rejected_external_scores, 50)   # Bottom 50% get default=0
    
    df_external = df_normalized.copy()
    augmented_defaults_external = df_external['observed_default'].copy()
    
    # High risk rejected samples (top 15% by external predictor)
    high_risk_rejected = rejected_mask & (df_external['external_predictor'] > external_threshold_high)
    # Low risk rejected samples (bottom 50% by external predictor)
    low_risk_rejected = rejected_mask & (df_external['external_predictor'] <= external_threshold_low)
    # Medium risk rejected samples get probabilistic assignment
    medium_risk_rejected = rejected_mask & (df_external['external_predictor'] > external_threshold_low) & (df_external['external_predictor'] <= external_threshold_high)
    
    augmented_defaults_external[high_risk_rejected] = 1
    augmented_defaults_external[low_risk_rejected] = 0
    
    # For medium risk, use probability proportional to external predictor score
    medium_scores = df_external[medium_risk_rejected]['external_predictor']
    medium_probs = (medium_scores - external_threshold_low) / (external_threshold_high - external_threshold_low)
    np.random.seed(42)
    medium_assignments = np.random.binomial(1, medium_probs)
    augmented_defaults_external[medium_risk_rejected] = medium_assignments
    
    X_train_external = df_external.iloc[train_idx][norm_feature_cols]
    y_train_external = augmented_defaults_external.iloc[train_idx]
    
    model_external = LogisticRegression(random_state=42, max_iter=1000)
    model_external.fit(X_train_external, y_train_external)
    strategies['External Predictor Enhanced'] = model_external
    
    # Strategy 4: Hybrid Approach (External + Propensity)
    print("Implementing hybrid approach (external predictor + propensity weighting)...")
    
    # Combine external predictor with propensity weighting
    # Use external predictor for label assignment, then apply propensity weights
    
    # Train propensity model
    X_full_train = df_normalized.iloc[train_idx][norm_feature_cols]
    approval_model = LogisticRegression(random_state=42, max_iter=1000)
    approval_model.fit(X_full_train, df_normalized.iloc[train_idx]['approved'])
    propensity_scores = approval_model.predict_proba(X_full_train)[:, 1]
    
    # Conservative weighting for approved samples
    approved_train_mask = df_normalized.iloc[train_idx]['approved'].values
    raw_weights = 1 / propensity_scores[approved_train_mask]
    weight_cap = np.percentile(raw_weights, 95)
    capped_weights = np.minimum(raw_weights, weight_cap)
    capped_weights = capped_weights / capped_weights.mean()
    
    # Combine external predictor labels with propensity weighting for approved samples
    model_hybrid = LogisticRegression(random_state=42, max_iter=1000)
    
    # For this hybrid approach, we'll use external predictor labels but weight everything by propensity
    # Create weights for the full augmented dataset
    all_weights = np.ones(len(train_idx))
    all_weights[approved_train_mask] = capped_weights
    # Give lower weight to artificially labeled rejected samples
    rejected_train_mask = ~approved_train_mask
    all_weights[rejected_train_mask] = 0.5  # Lower confidence in artificial labels
    
    model_hybrid.fit(X_train_external, y_train_external, sample_weight=all_weights)
    strategies['Hybrid (External + Propensity)'] = model_hybrid
    
    # Strategy 5: Oracle (for comparison)
    X_full_train = df_normalized.iloc[train_idx][norm_feature_cols]
    y_full_train = df_normalized.iloc[train_idx]['actual_default']
    
    model_oracle = LogisticRegression(random_state=42, max_iter=1000)
    model_oracle.fit(X_full_train, y_full_train)
    strategies['Oracle (Full Data)'] = model_oracle
    
    # Evaluate all strategies
    print("\n" + "-" * 80)
    print("EXTERNAL PREDICTOR REJECTION INFERENCE RESULTS")
    print("-" * 80)
    
    results = {}
    X_test = df_normalized.iloc[test_idx][norm_feature_cols]
    y_test = df_normalized.iloc[test_idx]['actual_default']
    
    print(f"\n{'Strategy':<35} {'Coef MAE':<12} {'Max |Bias|':<12} {'AUC':<8} {'Interp Score':<20}")
    print("-" * 95)
    
    for name, model in strategies.items():
        # Extract coefficients
        fitted_coefs = [model.intercept_[0]] + list(model.coef_[0])
        
        # Calculate coefficient bias
        bias = np.array(fitted_coefs) - np.array(true_coefficients)
        mae = np.mean(np.abs(bias))
        max_abs_bias = np.max(np.abs(bias))
        
        # Calculate AUC
        pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        
        # Interpretability score
        interpretability_score = max(0, 100 * (1 - mae / 10.0))
        
        results[name] = {
            'coefficients': fitted_coefs,
            'mae': mae,
            'max_abs_bias': max_abs_bias,
            'auc': auc,
            'interpretability_score': interpretability_score
        }
        
        print(f"{name:<35} {mae:<12.3f} {max_abs_bias:<12.3f} {auc:<8.3f} {interpretability_score:<20.1f}")
    
    # Analyze label assignment quality for different methods
    print(f"\nLABEL ASSIGNMENT QUALITY ANALYSIS:")
    print("-" * 80)
    
    rejected_mask_full = ~df_normalized['approved']
    rejected_true_labels = df_normalized[rejected_mask_full]['actual_default'].values
    
    # Naive method labels
    naive_labels = np.where(
        df_normalized[rejected_mask_full]['approval_score'] < rejection_threshold, 1, 0
    )
    
    # External predictor method labels
    external_labels = augmented_defaults_external[rejected_mask_full].values
    
    # Calculate accuracy metrics
    naive_accuracy = (naive_labels == rejected_true_labels).mean()
    external_accuracy = (external_labels == rejected_true_labels).mean()
    
    naive_precision = ((naive_labels == 1) & (rejected_true_labels == 1)).sum() / (naive_labels == 1).sum() if (naive_labels == 1).sum() > 0 else 0
    external_precision = ((external_labels == 1) & (rejected_true_labels == 1)).sum() / (external_labels == 1).sum() if (external_labels == 1).sum() > 0 else 0
    
    naive_recall = ((naive_labels == 1) & (rejected_true_labels == 1)).sum() / (rejected_true_labels == 1).sum() if (rejected_true_labels == 1).sum() > 0 else 0
    external_recall = ((external_labels == 1) & (rejected_true_labels == 1)).sum() / (rejected_true_labels == 1).sum() if (rejected_true_labels == 1).sum() > 0 else 0
    
    print(f"NAIVE REJECTION INFERENCE:")
    print(f"  Accuracy: {naive_accuracy:.1%}")
    print(f"  Precision: {naive_precision:.1%}")
    print(f"  Recall: {naive_recall:.1%}")
    print(f"  Assigned default rate: {naive_labels.mean():.1%}")
    print(f"  True default rate: {rejected_true_labels.mean():.1%}")
    
    print(f"\nEXTERNAL PREDICTOR METHOD:")
    print(f"  Accuracy: {external_accuracy:.1%}")
    print(f"  Precision: {external_precision:.1%}")
    print(f"  Recall: {external_recall:.1%}")
    print(f"  Assigned default rate: {external_labels.mean():.1%}")
    print(f"  True default rate: {rejected_true_labels.mean():.1%}")
    
    print(f"\nIMPROVEMENT FROM EXTERNAL PREDICTOR:")
    print(f"  Accuracy improvement: +{(external_accuracy - naive_accuracy)*100:.1f} percentage points")
    print(f"  Precision improvement: +{(external_precision - naive_precision)*100:.1f} percentage points")
    print(f"  Recall improvement: +{(external_recall - naive_recall)*100:.1f} percentage points")
    
    # Detailed coefficient comparison
    print(f"\nDETAILED COEFFICIENT COMPARISON:")
    print("-" * 80)
    
    coef_df = pd.DataFrame()
    coef_df['True_Coefficients'] = true_coefficients
    coef_df.index = feature_names
    
    for name, result in results.items():
        coef_df[name] = result['coefficients']
    
    print(coef_df.round(3))
    
    # Best strategy recommendations
    print(f"\nRECOMMENDATIONS BASED ON EXTERNAL PREDICTOR AVAILABILITY:")
    print("-" * 80)
    
    best_performance = max(results.items(), key=lambda x: x[1]['auc'])
    best_interpretability = min(results.items(), key=lambda x: x[1]['mae'])
    best_balance = min(results.items(), key=lambda x: x[1]['mae'] + (0.1 / max(x[1]['auc'], 0.01)))
    
    print(f"1. BEST PERFORMANCE: {best_performance[0]}")
    print(f"   - AUC: {best_performance[1]['auc']:.3f}")
    print(f"   - Coefficient MAE: {best_performance[1]['mae']:.3f}")
    
    print(f"\n2. BEST INTERPRETABILITY: {best_interpretability[0]}")
    print(f"   - Coefficient MAE: {best_interpretability[1]['mae']:.3f}")
    print(f"   - AUC: {best_interpretability[1]['auc']:.3f}")
    
    print(f"\n3. BEST BALANCE: {best_balance[0]}")
    print(f"   - Coefficient MAE: {best_balance[1]['mae']:.3f}")
    print(f"   - AUC: {best_balance[1]['auc']:.3f}")
    
    return results, coef_df, df_normalized

def create_external_predictor_recommendations():
    """
    Create recommendations for using external predictors in rejection inference
    """
    print("\n" + "=" * 80)
    print("EXTERNAL PREDICTOR STRATEGY RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = {
        "DATA REQUIREMENTS": [
            "External predictor should be available for both approved AND rejected samples",
            "Predictor should be correlated with true default risk (correlation > 0.3)",
            "Predictor should have some independence from approval criteria",
            "Data quality and timeliness must be consistent across populations"
        ],
        
        "IMPLEMENTATION BEST PRACTICES": [
            "Use multiple thresholds for nuanced label assignment (high/medium/low risk)",
            "Combine external predictor with propensity weighting for optimal results",
            "Validate external predictor predictive power on approved samples first",
            "Monitor external predictor stability and drift over time",
            "Apply lower weights to artificially labeled samples in training"
        ],
        
        "EXAMPLES OF USEFUL EXTERNAL PREDICTORS": [
            "Alternative credit data (utility payments, rent history, telecom bills)",
            "Banking transaction patterns and cash flow analysis",
            "Employment verification and income stability metrics",
            "Property ownership and asset verification",
            "Digital footprint and behavioral analytics",
            "Industry-specific risk scores (e.g., FICO, VantageScore alternatives)"
        ],
        
        "VALIDATION FRAMEWORK": [
            "Test external predictor on known outcomes (approved samples)",
            "Compare label assignment accuracy between naive and external methods",
            "Monitor coefficient stability when including external predictor",
            "Validate that external predictor doesn't introduce new biases",
            "Establish performance benchmarks and monitoring thresholds"
        ],
        
        "BUSINESS CONSIDERATIONS": [
            "Cost-benefit analysis of acquiring external data",
            "Regulatory compliance and data privacy requirements",
            "Integration complexity and operational overhead",
            "Vendor risk management and data quality guarantees",
            "Model explainability and audit trail requirements"
        ]
    }
    
    for category, items in recommendations.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
        print("-" * 60)

if __name__ == "__main__":
    # Run the complete analysis
    print("Starting Sample Selection Bias Analysis...")
    
    # Demonstrate the bias problem
    results, df_with_outcomes = demonstrate_selection_bias()
    
    # Show techniques to address it
    df_full, models = demonstrate_rejection_inference()
    
    # Create visualizations
    create_visualizations(results, df_with_outcomes)
    
    # Print insights
    print_summary_insights()
    
    # Compare model coefficients
    coef_comparison, bias_metrics_df, performance_results = compare_model_coefficients()
    
    # Visualize coefficient comparison
    visualize_coefficient_comparison(coef_comparison, bias_metrics_df)
    
    # Diagnose rejection inference bias
    df_augmented, worst_rejected_mask, better_rejected_mask = diagnose_rejection_inference_bias()
    
    # Compare coefficient stability
    results_stability = compare_coefficient_stability()
    
    # Interpretability-first analysis
    results_interpretability, coef_df_interpretability = interpretability_first_analysis()
    
    # Create interpretability recommendations
    create_interpretability_recommendations()
    
    # Simulate external predictor enhanced rejection inference
    results_external, coef_df_external, df_normalized_external = simulate_external_predictor_rejection_inference()
    
    # Create external predictor recommendations
    create_external_predictor_recommendations()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nThis simulation demonstrates how sample selection bias affects")
    print("credit modeling and provides practical techniques to address it.")
    print("The key takeaway: addressing rejected samples can significantly")
    print("improve model performance for the target population.")
