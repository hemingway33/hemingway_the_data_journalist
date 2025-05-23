"""
Loan Profit Optimization Using Survival Model Predictions

This module implements a comprehensive framework to determine the optimal payment pattern 
for each borrower to maximize loan portfolio profit, using survival analysis predictions 
to estimate default probabilities and calculate risk-adjusted expected returns.

Key Features:
- Expected profit calculation for each payment pattern
- Risk-adjusted return metrics
- Portfolio optimization recommendations
- Business constraints and rules
- Comprehensive profit analysis and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from credit_survival_model import CreditSurvivalModel
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LoanProfitOptimizer:
    """
    Optimize loan payment patterns to maximize portfolio profit using survival analysis.
    """
    
    def __init__(self, survival_model: CreditSurvivalModel, 
                 base_interest_rate: float = 0.12,
                 cost_of_funds: float = 0.03,
                 recovery_rate: float = 0.40):
        """
        Initialize the profit optimizer.
        
        Parameters:
        -----------
        survival_model : CreditSurvivalModel
            Fitted survival model for default prediction
        base_interest_rate : float
            Base annual interest rate (before risk adjustments)
        cost_of_funds : float
            Cost of funds (funding rate)
        recovery_rate : float
            Expected recovery rate on defaulted loans
        """
        self.survival_model = survival_model
        self.base_interest_rate = base_interest_rate
        self.cost_of_funds = cost_of_funds
        self.recovery_rate = recovery_rate
        
        # Payment pattern specific parameters
        self.payment_patterns = {
            'installment': {
                'rate_premium': 0.0,  # Base rate (reference)
                'origination_fee': 0.01,  # 1% of loan amount
                'servicing_cost_annual': 0.005,  # 0.5% annually
                'min_credit_score': 600,
                'max_dti': 0.45
            },
            'balloon': {
                'rate_premium': 0.015,  # 150 bps premium for risk
                'origination_fee': 0.015,  # 1.5% of loan amount  
                'servicing_cost_annual': 0.008,  # Higher servicing costs
                'min_credit_score': 650,  # Stricter underwriting
                'max_dti': 0.40
            },
            'interest_only': {
                'rate_premium': 0.008,  # 80 bps premium
                'origination_fee': 0.012,  # 1.2% of loan amount
                'servicing_cost_annual': 0.006,  # Moderate servicing costs
                'min_credit_score': 620,
                'max_dti': 0.35  # Lower DTI for payment shock protection
            }
        }
    
    def calculate_payment_schedule(self, loan_amount: float, annual_rate: float, 
                                 term_months: int, pattern: str) -> Dict:
        """
        Calculate payment schedule based on payment pattern.
        
        Parameters:
        -----------
        loan_amount : float
            Principal loan amount
        annual_rate : float
            Annual interest rate
        term_months : int
            Loan term in months
        pattern : str
            Payment pattern ('installment', 'balloon', 'interest_only')
            
        Returns:
        --------
        Dict : Payment schedule details
        """
        monthly_rate = annual_rate / 12
        
        if pattern == 'installment':
            # Standard amortizing loan
            if monthly_rate > 0:
                monthly_payment = loan_amount * (
                    monthly_rate * (1 + monthly_rate)**term_months
                ) / ((1 + monthly_rate)**term_months - 1)
            else:
                monthly_payment = loan_amount / term_months
            
            return {
                'monthly_payment': monthly_payment,
                'total_payments': monthly_payment * term_months,
                'total_interest': monthly_payment * term_months - loan_amount,
                'balloon_payment': 0
            }
        
        elif pattern == 'balloon':
            # Interest-only payments with balloon at end
            monthly_payment = loan_amount * monthly_rate
            balloon_payment = loan_amount
            
            return {
                'monthly_payment': monthly_payment,
                'total_payments': monthly_payment * term_months + balloon_payment,
                'total_interest': monthly_payment * term_months,
                'balloon_payment': balloon_payment
            }
        
        elif pattern == 'interest_only':
            # Interest-only for part of term, then amortizing
            io_months = min(term_months // 2, 60)  # IO period up to 5 years
            amort_months = term_months - io_months
            
            # Interest-only phase
            io_payment = loan_amount * monthly_rate
            
            # Amortizing phase
            if monthly_rate > 0 and amort_months > 0:
                amort_payment = loan_amount * (
                    monthly_rate * (1 + monthly_rate)**amort_months
                ) / ((1 + monthly_rate)**amort_months - 1)
            else:
                amort_payment = loan_amount / amort_months if amort_months > 0 else loan_amount
            
            total_payments = io_payment * io_months + amort_payment * amort_months
            
            return {
                'monthly_payment': io_payment,  # Initial payment
                'amort_payment': amort_payment,  # Payment after IO period
                'io_months': io_months,
                'total_payments': total_payments,
                'total_interest': total_payments - loan_amount,
                'balloon_payment': 0
            }
    
    def calculate_expected_profit(self, borrower_data: pd.DataFrame, 
                                payment_pattern: str, 
                                loan_amount: float,
                                term_months: int = 48,
                                time_horizon: int = 60) -> Dict:
        """
        Calculate expected profit for a specific payment pattern.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        payment_pattern : str
            Payment pattern to evaluate
        loan_amount : float
            Loan amount
        term_months : int
            Loan term
        time_horizon : int
            Analysis time horizon in months
            
        Returns:
        --------
        Dict : Expected profit metrics
        """
        # Get payment pattern parameters
        pattern_params = self.payment_patterns[payment_pattern]
        
        # Calculate interest rate for this pattern
        interest_rate = self.base_interest_rate + pattern_params['rate_premium']
        
        # Calculate payment schedule
        payment_schedule = self.calculate_payment_schedule(
            loan_amount, interest_rate, term_months, payment_pattern
        )
        
        # Prepare borrower data for survival prediction
        borrower_profile = borrower_data.copy()
        borrower_profile['loan_amount'] = loan_amount
        borrower_profile['loan_term_months'] = term_months
        borrower_profile['is_balloon_payment'] = 1 if payment_pattern == 'balloon' else 0
        borrower_profile['is_interest_only'] = 1 if payment_pattern == 'interest_only' else 0
        
        # Predict survival probabilities
        time_points = list(range(1, min(time_horizon, term_months) + 1))
        
        survival_probs = []
        for t in time_points:
            # Create time-specific profile
            profile_t = borrower_profile.copy()
            profile_t['start_time'] = 0
            profile_t['stop_time'] = t
            profile_t['loan_duration_months'] = t
            profile_t['months_to_maturity'] = max(0, term_months - t)
            
            # Convert to DataFrame if it's a Series
            if isinstance(profile_t, pd.Series):
                profile_t = profile_t.to_frame().T
            
            try:
                survival_pred = self.survival_model.predict_survival_probability(
                    profile_t, time_points=[t]
                )
                
                # Get the subject ID from the prediction
                subject_id = profile_t['id'].iloc[0] if 'id' in profile_t.columns else list(survival_pred.keys())[0]
                survival_prob = survival_pred[subject_id][t]
                survival_probs.append(survival_prob)
            except:
                # Fallback to decreasing survival probability
                survival_probs.append(max(0.5, 1.0 - t * 0.01))
        
        # Calculate expected cash flows
        revenues = []
        costs = []
        defaults = []
        
        # Initial fees
        origination_revenue = loan_amount * pattern_params['origination_fee']
        initial_cost = loan_amount  # Cost of funds for loan origination
        
        monthly_servicing_cost = loan_amount * pattern_params['servicing_cost_annual'] / 12
        
        # Monthly cash flows
        for i, t in enumerate(time_points):
            if i == 0:
                prev_survival = 1.0
            else:
                prev_survival = survival_probs[i-1] if i-1 < len(survival_probs) else survival_probs[-1]
            
            current_survival = survival_probs[i] if i < len(survival_probs) else survival_probs[-1]
            
            # Default probability in this period
            default_prob = prev_survival - current_survival
            defaults.append(default_prob)
            
            # Expected payment (survival weighted)
            if payment_pattern == 'interest_only' and hasattr(payment_schedule, 'io_months'):
                if t <= payment_schedule.get('io_months', term_months // 2):
                    expected_payment = payment_schedule['monthly_payment'] * current_survival
                else:
                    expected_payment = payment_schedule['amort_payment'] * current_survival
            else:
                expected_payment = payment_schedule['monthly_payment'] * current_survival
            
            revenues.append(expected_payment)
            
            # Servicing costs (only for surviving loans)
            costs.append(monthly_servicing_cost * current_survival)
        
        # Balloon payment if applicable
        balloon_recovery = 0
        if payment_pattern == 'balloon' and term_months <= time_horizon:
            final_survival = survival_probs[min(term_months-1, len(survival_probs)-1)]
            balloon_recovery = payment_schedule['balloon_payment'] * final_survival
        
        # Expected loss from defaults
        total_default_prob = 1 - (survival_probs[-1] if survival_probs else 0.5)
        expected_loss = loan_amount * total_default_prob * (1 - self.recovery_rate)
        expected_recovery = loan_amount * total_default_prob * self.recovery_rate
        
        # Calculate NPV
        discount_rate = self.cost_of_funds / 12  # Monthly discount rate
        
        npv_revenues = origination_revenue + balloon_recovery
        npv_costs = initial_cost + expected_loss
        
        for i, (rev, cost) in enumerate(zip(revenues, costs)):
            npv_revenues += rev / (1 + discount_rate)**(i+1)
            npv_costs += cost / (1 + discount_rate)**(i+1)
        
        expected_profit = npv_revenues - npv_costs
        
        # Risk-adjusted metrics
        expected_return = expected_profit / loan_amount if loan_amount > 0 else 0
        
        # Sharpe-like ratio (return per unit of default risk)
        risk_adjusted_return = expected_return / max(total_default_prob, 0.01)
        
        return {
            'payment_pattern': payment_pattern,
            'expected_profit': expected_profit,
            'expected_return': expected_return,
            'risk_adjusted_return': risk_adjusted_return,
            'total_default_prob': total_default_prob,
            'expected_loss': expected_loss,
            'total_revenue': npv_revenues,
            'total_cost': npv_costs,
            'interest_rate': interest_rate,
            'monthly_payment': payment_schedule['monthly_payment'],
            'balloon_payment': payment_schedule.get('balloon_payment', 0)
        }
    
    def check_underwriting_criteria(self, borrower_data: pd.DataFrame, 
                                  payment_pattern: str) -> bool:
        """
        Check if borrower meets underwriting criteria for the payment pattern.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        payment_pattern : str
            Payment pattern to check
            
        Returns:
        --------
        bool : Whether borrower qualifies
        """
        criteria = self.payment_patterns[payment_pattern]
        
        # Extract borrower characteristics
        if isinstance(borrower_data, pd.Series):
            borrower = borrower_data
        else:
            borrower = borrower_data.iloc[0]
        
        # Check credit score (assuming it's standardized, convert back to raw score)
        # This is a simplified conversion - in practice you'd use the actual raw scores
        estimated_credit_score = 650 + borrower.get('credit_score', 0) * 100
        
        if estimated_credit_score < criteria['min_credit_score']:
            return False
        
        # Check debt-to-income ratio
        estimated_dti = 0.3 + borrower.get('debt_to_income_ratio', 0) * 0.2
        if estimated_dti > criteria['max_dti']:
            return False
        
        return True
    
    def optimize_payment_pattern(self, borrower_data: pd.DataFrame,
                               loan_amount: float,
                               term_months: int = 48) -> Dict:
        """
        Determine optimal payment pattern for a borrower to maximize expected profit.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        loan_amount : float
            Requested loan amount
        term_months : int
            Loan term in months
            
        Returns:
        --------
        Dict : Optimization results
        """
        results = {}
        eligible_patterns = []
        
        # Evaluate each payment pattern
        for pattern in ['installment', 'balloon', 'interest_only']:
            # Check underwriting criteria
            if not self.check_underwriting_criteria(borrower_data, pattern):
                results[pattern] = {
                    'eligible': False,
                    'reason': 'Does not meet underwriting criteria'
                }
                continue
            
            # Calculate expected profit
            try:
                profit_analysis = self.calculate_expected_profit(
                    borrower_data, pattern, loan_amount, term_months
                )
                profit_analysis['eligible'] = True
                results[pattern] = profit_analysis
                eligible_patterns.append(pattern)
            except Exception as e:
                results[pattern] = {
                    'eligible': False,
                    'reason': f'Calculation error: {str(e)}'
                }
        
        # Determine optimal pattern
        if eligible_patterns:
            # Find pattern with highest expected profit
            best_pattern = max(eligible_patterns, 
                             key=lambda p: results[p]['expected_profit'])
            
            # Alternative: highest risk-adjusted return
            best_risk_adjusted = max(eligible_patterns,
                                   key=lambda p: results[p]['risk_adjusted_return'])
        else:
            best_pattern = None
            best_risk_adjusted = None
        
        return {
            'borrower_analysis': results,
            'recommended_pattern': best_pattern,
            'recommended_risk_adjusted': best_risk_adjusted,
            'eligible_patterns': eligible_patterns,
            'loan_amount': loan_amount,
            'term_months': term_months
        }
    
    def analyze_portfolio_optimization(self, borrower_portfolio: pd.DataFrame,
                                     loan_amounts: List[float],
                                     term_months: int = 48) -> pd.DataFrame:
        """
        Optimize payment patterns for an entire portfolio of borrowers.
        
        Parameters:
        -----------
        borrower_portfolio : pd.DataFrame
            Portfolio of borrower characteristics
        loan_amounts : List[float]
            Loan amounts for each borrower
        term_months : int
            Standard loan term
            
        Returns:
        --------
        pd.DataFrame : Portfolio optimization results
        """
        portfolio_results = []
        
        print(f"Optimizing payment patterns for {len(borrower_portfolio)} borrowers...")
        
        for idx, (_, borrower) in enumerate(borrower_portfolio.iterrows()):
            if idx < len(loan_amounts):
                loan_amount = loan_amounts[idx]
            else:
                loan_amount = np.mean(loan_amounts)  # Default to average
            
            # Convert Series to DataFrame for optimization
            borrower_df = borrower.to_frame().T
            borrower_df['id'] = f'borrower_{idx}'
            
            # Optimize payment pattern
            optimization_result = self.optimize_payment_pattern(
                borrower_df, loan_amount, term_months
            )
            
            # Extract key results
            result = {
                'borrower_id': idx,
                'loan_amount': loan_amount,
                'recommended_pattern': optimization_result['recommended_pattern'],
                'recommended_risk_adjusted': optimization_result['recommended_risk_adjusted']
            }
            
            # Add profit metrics for each eligible pattern
            for pattern in ['installment', 'balloon', 'interest_only']:
                if pattern in optimization_result['borrower_analysis']:
                    analysis = optimization_result['borrower_analysis'][pattern]
                    if analysis.get('eligible', False):
                        result[f'{pattern}_profit'] = analysis['expected_profit']
                        result[f'{pattern}_return'] = analysis['expected_return']
                        result[f'{pattern}_default_prob'] = analysis['total_default_prob']
                        result[f'{pattern}_risk_adj_return'] = analysis['risk_adjusted_return']
                    else:
                        result[f'{pattern}_profit'] = np.nan
                        result[f'{pattern}_return'] = np.nan
                        result[f'{pattern}_default_prob'] = np.nan
                        result[f'{pattern}_risk_adj_return'] = np.nan
            
            portfolio_results.append(result)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1} borrowers...")
        
        return pd.DataFrame(portfolio_results)
    
    def plot_profit_analysis(self, portfolio_results: pd.DataFrame):
        """
        Create comprehensive visualizations of portfolio profit analysis.
        
        Parameters:
        -----------
        portfolio_results : pd.DataFrame
            Results from analyze_portfolio_optimization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribution of recommended patterns
        ax1 = axes[0, 0]
        pattern_counts = portfolio_results['recommended_pattern'].value_counts()
        colors = ['green', 'red', 'orange']
        bars = ax1.bar(pattern_counts.index, pattern_counts.values, color=colors[:len(pattern_counts)])
        ax1.set_title('Recommended Payment Pattern Distribution')
        ax1.set_ylabel('Number of Borrowers')
        
        # Add percentage labels
        total = len(portfolio_results)
        for bar, count in zip(bars, pattern_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count/total:.1%}', ha='center', va='bottom')
        
        # 2. Expected profit by pattern
        ax2 = axes[0, 1]
        profit_data = []
        patterns = ['installment', 'balloon', 'interest_only']
        
        for pattern in patterns:
            profit_col = f'{pattern}_profit'
            if profit_col in portfolio_results.columns:
                profits = portfolio_results[profit_col].dropna()
                if len(profits) > 0:
                    profit_data.append(profits)
        
        if profit_data:
            ax2.boxplot(profit_data, labels=[p.title() for p in patterns[:len(profit_data)]])
            ax2.set_title('Expected Profit Distribution by Pattern')
            ax2.set_ylabel('Expected Profit ($)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Risk vs Return scatter plot
        ax3 = axes[0, 2]
        for i, pattern in enumerate(patterns):
            return_col = f'{pattern}_return'
            risk_col = f'{pattern}_default_prob'
            
            if return_col in portfolio_results.columns and risk_col in portfolio_results.columns:
                returns = portfolio_results[return_col].dropna()
                risks = portfolio_results[risk_col].dropna()
                
                if len(returns) > 0 and len(risks) > 0:
                    # Align the data
                    valid_idx = portfolio_results[[return_col, risk_col]].dropna().index
                    if len(valid_idx) > 0:
                        returns_aligned = portfolio_results.loc[valid_idx, return_col]
                        risks_aligned = portfolio_results.loc[valid_idx, risk_col]
                        
                        ax3.scatter(risks_aligned, returns_aligned, 
                                  label=pattern.title(), alpha=0.6, 
                                  color=colors[i % len(colors)])
        
        ax3.set_xlabel('Default Probability')
        ax3.set_ylabel('Expected Return')
        ax3.set_title('Risk vs Return by Payment Pattern')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Portfolio value optimization
        ax4 = axes[1, 0]
        # Calculate total portfolio value for each pattern
        portfolio_values = {}
        for pattern in patterns:
            profit_col = f'{pattern}_profit'
            if profit_col in portfolio_results.columns:
                total_profit = portfolio_results[profit_col].fillna(0).sum()
                portfolio_values[pattern] = total_profit
        
        if portfolio_values:
            bars = ax4.bar(portfolio_values.keys(), portfolio_values.values(), 
                          color=['green', 'red', 'orange'][:len(portfolio_values)])
            ax4.set_title('Total Portfolio Value by Pattern')
            ax4.set_ylabel('Total Expected Profit ($)')
            
            # Add value labels
            for bar, value in zip(bars, portfolio_values.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(portfolio_values.values())*0.01,
                        f'${value:,.0f}', ha='center', va='bottom')
        
        # 5. Optimal vs Alternative patterns
        ax5 = axes[1, 1]
        # Compare recommended pattern profit vs best alternative
        profit_comparison = []
        
        for _, row in portfolio_results.iterrows():
            recommended = row['recommended_pattern']
            if pd.notna(recommended):
                recommended_profit = row.get(f'{recommended}_profit', 0)
                
                # Find best alternative
                alternative_profits = []
                for pattern in patterns:
                    if pattern != recommended:
                        alt_profit = row.get(f'{pattern}_profit', np.nan)
                        if pd.notna(alt_profit):
                            alternative_profits.append(alt_profit)
                
                if alternative_profits:
                    best_alternative = max(alternative_profits)
                    profit_lift = recommended_profit - best_alternative
                    profit_comparison.append(profit_lift)
        
        if profit_comparison:
            ax5.hist(profit_comparison, bins=20, alpha=0.7, color='blue')
            ax5.set_title('Profit Improvement vs Best Alternative')
            ax5.set_xlabel('Profit Difference ($)')
            ax5.set_ylabel('Number of Borrowers')
            ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax5.grid(True, alpha=0.3)
        
        # 6. Risk-adjusted return comparison
        ax6 = axes[1, 2]
        risk_adj_data = []
        
        for pattern in patterns:
            risk_adj_col = f'{pattern}_risk_adj_return'
            if risk_adj_col in portfolio_results.columns:
                values = portfolio_results[risk_adj_col].dropna()
                if len(values) > 0:
                    risk_adj_data.append(values)
        
        if risk_adj_data:
            ax6.boxplot(risk_adj_data, labels=[p.title() for p in patterns[:len(risk_adj_data)]])
            ax6.set_title('Risk-Adjusted Return by Pattern')
            ax6.set_ylabel('Risk-Adjusted Return')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_business_recommendations(self, portfolio_results: pd.DataFrame) -> Dict:
        """
        Generate business recommendations based on portfolio optimization.
        
        Parameters:
        -----------
        portfolio_results : pd.DataFrame
            Results from portfolio optimization
            
        Returns:
        --------
        Dict : Business recommendations and insights
        """
        # Calculate key metrics
        total_borrowers = len(portfolio_results)
        
        # Pattern distribution
        pattern_distribution = portfolio_results['recommended_pattern'].value_counts(normalize=True)
        
        # Portfolio-level metrics
        portfolio_metrics = {}
        patterns = ['installment', 'balloon', 'interest_only']
        
        for pattern in patterns:
            profit_col = f'{pattern}_profit'
            if profit_col in portfolio_results.columns:
                profits = portfolio_results[profit_col].dropna()
                if len(profits) > 0:
                    portfolio_metrics[pattern] = {
                        'total_profit': profits.sum(),
                        'avg_profit': profits.mean(),
                        'eligible_borrowers': len(profits),
                        'avg_default_prob': portfolio_results[f'{pattern}_default_prob'].mean()
                    }
        
        # Best performing pattern
        best_pattern = max(portfolio_metrics.keys(), 
                          key=lambda p: portfolio_metrics[p]['total_profit']) if portfolio_metrics else None
        
        recommendations = {
            'executive_summary': {
                'total_borrowers_analyzed': total_borrowers,
                'recommended_patterns': pattern_distribution.to_dict(),
                'best_performing_pattern': best_pattern,
                'total_portfolio_value': sum(m['total_profit'] for m in portfolio_metrics.values())
            },
            
            'pattern_insights': portfolio_metrics,
            
            'business_recommendations': [],
            
            'risk_management': [],
            
            'pricing_strategy': []
        }
        
        # Generate specific recommendations
        if best_pattern:
            recommendations['business_recommendations'].extend([
                f"Focus marketing efforts on {best_pattern} loans - highest portfolio value",
                f"Develop specialized underwriting for {best_pattern} loans",
                "Implement risk-based pricing across all payment patterns"
            ])
        
        # Risk management recommendations
        balloon_share = pattern_distribution.get('balloon', 0)
        if balloon_share > 0.3:
            recommendations['risk_management'].append(
                "HIGH RISK: Balloon payment concentration exceeds 30% - implement concentration limits"
            )
        
        if balloon_share > 0.1:
            recommendations['risk_management'].extend([
                "Develop balloon payment monitoring system",
                "Create refinancing programs for approaching balloon payments",
                "Enhanced capital allocation for balloon payment risk"
            ])
        
        # Pricing strategy
        for pattern in patterns:
            if pattern in portfolio_metrics:
                avg_default = portfolio_metrics[pattern]['avg_default_prob']
                if avg_default > 0.5:
                    recommendations['pricing_strategy'].append(
                        f"Consider higher risk premium for {pattern} loans (avg default prob: {avg_default:.1%})"
                    )
        
        return recommendations


def main():
    """Demonstrate the loan profit optimization framework."""
    print("="*80)
    print("LOAN PROFIT OPTIMIZATION USING SURVIVAL MODEL")
    print("="*80)
    
    # Initialize and fit survival model
    print("\n1. Initializing survival model...")
    survival_model = CreditSurvivalModel(random_state=42)
    
    # Generate training data
    train_data, test_data, val_data = survival_model.generate_sample_data(
        n_subjects=500, max_time=48, test_size=0.3, val_size=0.1
    )
    
    # Fit the model
    survival_model.fit_cox_model(penalizer=0.01)
    
    # Initialize profit optimizer
    print("\n2. Initializing profit optimizer...")
    optimizer = LoanProfitOptimizer(
        survival_model=survival_model,
        base_interest_rate=0.12,  # 12% base rate
        cost_of_funds=0.03,       # 3% cost of funds
        recovery_rate=0.40        # 40% recovery rate
    )
    
    # Analyze portfolio (using test data as loan applicants)
    print("\n3. Analyzing loan portfolio optimization...")
    
    # Prepare test data for optimization
    portfolio_data = test_data.groupby('id').first().reset_index()
    loan_amounts = np.random.uniform(10000, 50000, len(portfolio_data))
    
    # Run portfolio optimization
    portfolio_results = optimizer.analyze_portfolio_optimization(
        portfolio_data, loan_amounts.tolist(), term_months=48
    )
    
    print(f"Optimization complete for {len(portfolio_results)} borrowers!")
    
    # Display results summary
    print("\n4. OPTIMIZATION RESULTS SUMMARY")
    print("-" * 40)
    
    # Pattern distribution
    pattern_dist = portfolio_results['recommended_pattern'].value_counts()
    print("Recommended Payment Pattern Distribution:")
    for pattern, count in pattern_dist.items():
        percentage = count / len(portfolio_results) * 100
        print(f"  {pattern.title()}: {count} borrowers ({percentage:.1f}%)")
    
    # Profit summary
    print(f"\nPortfolio Profit Summary:")
    total_profit = 0
    for pattern in ['installment', 'balloon', 'interest_only']:
        profit_col = f'{pattern}_profit'
        if profit_col in portfolio_results.columns:
            pattern_profit = portfolio_results[profit_col].fillna(0).sum()
            total_profit += pattern_profit
            print(f"  {pattern.title()}: ${pattern_profit:,.0f}")
    
    print(f"  TOTAL PORTFOLIO VALUE: ${total_profit:,.0f}")
    
    # Generate business recommendations
    print("\n5. BUSINESS RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = optimizer.generate_business_recommendations(portfolio_results)
    
    print("Executive Summary:")
    for key, value in recommendations['executive_summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\nKey Recommendations:")
    for i, rec in enumerate(recommendations['business_recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nRisk Management:")
    for i, risk in enumerate(recommendations['risk_management'], 1):
        print(f"  {i}. {risk}")
    
    print(f"\nPricing Strategy:")
    for i, pricing in enumerate(recommendations['pricing_strategy'], 1):
        print(f"  {i}. {pricing}")
    
    # Create visualizations
    print("\n6. Generating profit analysis visualizations...")
    optimizer.plot_profit_analysis(portfolio_results)
    
    # Example individual borrower optimization
    print("\n7. INDIVIDUAL BORROWER EXAMPLE")
    print("-" * 40)
    
    # Select a sample borrower
    sample_borrower = portfolio_data.iloc[0:1].copy()
    sample_borrower['id'] = 'sample_borrower'
    sample_loan_amount = 25000
    
    # Optimize for this borrower
    individual_result = optimizer.optimize_payment_pattern(
        sample_borrower, sample_loan_amount, term_months=48
    )
    
    print(f"Sample Borrower Analysis (Loan Amount: ${sample_loan_amount:,}):")
    print(f"Recommended Pattern: {individual_result['recommended_pattern']}")
    print(f"Risk-Adjusted Optimal: {individual_result['recommended_risk_adjusted']}")
    
    print(f"\nDetailed Analysis:")
    for pattern, analysis in individual_result['borrower_analysis'].items():
        if analysis.get('eligible', False):
            print(f"  {pattern.title()}:")
            print(f"    Expected Profit: ${analysis['expected_profit']:,.0f}")
            print(f"    Expected Return: {analysis['expected_return']:.2%}")
            print(f"    Default Probability: {analysis['total_default_prob']:.1%}")
            print(f"    Interest Rate: {analysis['interest_rate']:.2%}")
            print(f"    Monthly Payment: ${analysis['monthly_payment']:,.2f}")
        else:
            print(f"  {pattern.title()}: Not eligible - {analysis.get('reason', 'Unknown')}")
    
    print("\n" + "="*80)
    print("PROFIT OPTIMIZATION ANALYSIS COMPLETE!")
    print("="*80)
    
    return optimizer, portfolio_results, recommendations


if __name__ == "__main__":
    optimizer, results, recommendations = main() 