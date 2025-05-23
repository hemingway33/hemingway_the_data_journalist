"""
Simple demonstration of the Credit Survival Model

This script provides a focused demonstration of the key features
of the Cox hazard survival model for credit default prediction.
"""

from credit_survival_model import CreditSurvivalModel
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Focused demonstration of the credit survival model."""
    
    print("=" * 60)
    print("CREDIT SURVIVAL MODEL DEMONSTRATION")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing the model...")
    model = CreditSurvivalModel(random_state=42)
    
    # Generate data with fewer subjects for faster demonstration
    print("\n2. Generating sample credit data...")
    train_data, test_data, val_data = model.generate_sample_data(
        n_subjects=800,  # Reduced for faster execution
        max_time=48,     # 4 years observation period
        test_size=0.2,
        val_size=0.1
    )
    
    print(f"\nData Summary:")
    print(f"- Training subjects: {train_data['id'].nunique()}")
    print(f"- Test subjects: {test_data['id'].nunique()}")
    print(f"- Validation subjects: {val_data['id'].nunique()}")
    print(f"- Training default rate: {train_data.groupby('id')['event'].max().mean():.3f}")
    
    # Fit the model
    print("\n3. Fitting Cox proportional hazards model...")
    model.fit_cox_model(penalizer=0.01)
    
    # Display key results
    print("\n4. Model Performance:")
    print(f"- Training C-index: {model.train_c_index:.4f}")
    print(f"- Test C-index: {model.test_c_index:.4f}")
    print(f"- Validation C-index: {model.val_c_index:.4f}")
    
    # Show feature importance
    print("\n5. Top Risk Factors (Hazard Ratios):")
    feature_importance = model.cox_model.summary
    hazard_ratios = np.exp(feature_importance['coef'])
    sorted_features = hazard_ratios.sort_values(ascending=False)
    
    for feature, hr in sorted_features.items():
        direction = "↑ Increases" if hr > 1 else "↓ Decreases"
        effect = abs((hr - 1) * 100)
        print(f"   {feature}: {direction} risk by {effect:.1f}% (HR: {hr:.3f})")
    
    # Predict survival for different risk profiles
    print("\n6. Survival Predictions for Risk Profiles:")
    
    # Create example profiles (using scaled values)
    profiles = {
        'Low Risk Installment Customer': {
            'id': 'low_risk_installment',
            'age': -1.0,      # Younger than average
            'income': 1.0,    # Higher income
            'loan_amount': -0.5,  # Smaller loan
            'credit_score': 1.0,  # Better credit score
            'employment_years': 1.0,  # More stable employment
            'debt_to_income_ratio': -1.0,  # Lower DTI
            'payment_history_score': 1.0,  # Better payment history
            'is_balloon_payment': 0,  # Installment payment
            'is_interest_only': 0,
            'loan_term_months': 36,
            'months_to_maturity': 12,  # Not critical for installment
            'start_time': 0,
            'stop_time': 24,
            'loan_duration_months': 24
        },
        'High Risk Balloon Customer': {
            'id': 'high_risk_balloon',
            'age': 1.0,       # Older
            'income': -1.0,   # Lower income
            'loan_amount': 1.0,   # Larger loan
            'credit_score': -1.0, # Poor credit score
            'employment_years': -1.0, # Less stable employment
            'debt_to_income_ratio': 1.0,  # Higher DTI
            'payment_history_score': -1.0, # Poor payment history
            'is_balloon_payment': 1,  # Balloon payment - HIGH RISK
            'is_interest_only': 0,
            'loan_term_months': 48,
            'months_to_maturity': 6,   # Near balloon payment - VERY HIGH RISK
            'start_time': 0,
            'stop_time': 24,
            'loan_duration_months': 24
        },
        'Medium Risk Interest-Only Customer': {
            'id': 'medium_risk_io',
            'age': 0.0,       # Average age
            'income': 0.0,    # Average income
            'loan_amount': 0.5,   # Slightly higher loan
            'credit_score': 0.0,  # Average credit score
            'employment_years': 0.0, # Average employment
            'debt_to_income_ratio': 0.5,  # Moderate DTI
            'payment_history_score': 0.0, # Average payment history
            'is_balloon_payment': 0,
            'is_interest_only': 1,  # Interest-only payment - MEDIUM RISK
            'loan_term_months': 60,
            'months_to_maturity': 36,
            'start_time': 0,
            'stop_time': 24,
            'loan_duration_months': 24
        }
    }
    
    time_points = [6, 12, 24, 36]
    
    for profile_name, profile_data in profiles.items():
        print(f"\n   {profile_name}:")
        
        # Convert to DataFrame
        profile_df = pd.DataFrame([profile_data])
        
        # Predict survival probabilities
        survival_pred = model.predict_survival_probability(
            profile_df, time_points=time_points
        )
        
        for t in time_points:
            prob = survival_pred[profile_data['id']][t]
            print(f"     {t:2d} months: {prob:.3f} ({(1-prob)*100:.1f}% default risk)")
    
    # Create a simple visualization
    print("\n7. Creating visualization...")
    
    # Plot survival curves for the example profiles
    plt.figure(figsize=(14, 8))
    
    time_range = np.arange(1, 49)  # 48 months
    colors = ['green', 'red', 'orange']
    
    for i, (profile_name, profile_data) in enumerate(profiles.items()):
        survival_probs = []
        
        for t in time_range:
            # Update profile for current time
            current_profile = profile_data.copy()
            current_profile['stop_time'] = t
            current_profile['loan_duration_months'] = t
            # Update months to maturity
            current_profile['months_to_maturity'] = max(0, current_profile['loan_term_months'] - t)
            
            profile_df = pd.DataFrame([current_profile])
            survival_pred = model.predict_survival_probability(
                profile_df, time_points=[t]
            )
            survival_probs.append(survival_pred[profile_data['id']][t])
        
        # Shorten label for legend
        short_label = profile_name.replace(' Customer', '').replace('Risk ', '')
        plt.plot(time_range, survival_probs, 
                label=short_label, color=colors[i], linewidth=3, alpha=0.8)
    
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability (No Default)')
    plt.title('Credit Survival Curves: Payment Pattern Impact on Default Risk')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add annotations for payment patterns
    plt.annotate('Installment:\nLowest risk', 
                xy=(35, 0.85), xytext=(25, 0.95),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green', ha='center')
    
    plt.annotate('Balloon Payment:\nHighest risk\n(esp. near maturity)', 
                xy=(35, 0.2), xytext=(25, 0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red', ha='center')
    
    plt.annotate('Interest-Only:\nModerate risk', 
                xy=(35, 0.55), xytext=(42, 0.75),
                arrowprops=dict(arrowstyle='->', color='orange'),
                fontsize=9, color='orange', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print("\n8. Model Interpretation:")
    print("   - C-index > 0.8 indicates excellent model discrimination")
    print("   - Payment patterns significantly affect default risk:")
    print("     • Balloon payments: Highest risk (especially near maturity)")
    print("     • Interest-only: Moderate risk elevation")
    print("     • Installment: Lowest risk (reference category)")
    print("   - Key risk factors: Credit Score, Age, Payment Pattern")
    print("   - Time-varying effects: Balloon risk increases near maturity")
    
    # NEW: Payment pattern analysis
    print("\n9. Payment Pattern Business Impact:")
    print("   - Risk-based pricing should incorporate payment structure")
    print("   - Enhanced monitoring needed for balloon loans approaching maturity")
    print("   - Portfolio diversification across payment types recommended")
    
    print("\n" + "=" * 60)
    print("ENHANCED DEMONSTRATION COMPLETE!")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    import pandas as pd
    model = main() 