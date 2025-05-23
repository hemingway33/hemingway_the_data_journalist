"""
Enhanced Loan Profit Optimizer with Term Duration Optimization

This module extends the profit optimization framework to optimize both payment pattern 
AND loan term duration as decision variables. The optimizer evaluates multiple 
combinations to find the optimal loan structure that maximizes expected profit 
while managing risk exposure.

Key Features:
- Joint optimization of payment pattern and loan term
- Term-specific survival analysis and risk assessment
- Dynamic pricing based on term and payment structure
- Portfolio-level term distribution optimization
- Risk-adjusted return calculations across term durations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from credit_survival_model import CreditSurvivalModel
from loan_profit_optimizer import LoanProfitOptimizer
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LoanTermProfitOptimizer(LoanProfitOptimizer):
    """
    Enhanced profit optimizer that optimizes both payment pattern and loan term duration.
    """
    
    def __init__(self, survival_model: CreditSurvivalModel, 
                 base_interest_rate: float = 0.12,
                 cost_of_funds: float = 0.03,
                 recovery_rate: float = 0.40):
        """
        Initialize the enhanced profit optimizer.
        
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
        super().__init__(survival_model, base_interest_rate, cost_of_funds, recovery_rate)
        
        # Enhanced payment pattern parameters with term-specific constraints
        self.payment_patterns = {
            'installment': {
                'rate_premium': 0.0,  # Base rate (reference)
                'origination_fee': 0.01,  # 1% of loan amount
                'servicing_cost_annual': 0.005,  # 0.5% annually
                'min_credit_score': 600,
                'max_dti': 0.45,
                'min_term_months': 12,  # Minimum term
                'max_term_months': 84,  # Maximum term
                'preferred_terms': [24, 36, 48, 60, 72]  # Standard terms
            },
            'balloon': {
                'rate_premium': 0.015,  # 150 bps premium for risk
                'origination_fee': 0.015,  # 1.5% of loan amount  
                'servicing_cost_annual': 0.008,  # Higher servicing costs
                'min_credit_score': 650,  # Stricter underwriting
                'max_dti': 0.40,
                'min_term_months': 24,  # Minimum term for balloon
                'max_term_months': 60,  # Maximum term for balloon
                'preferred_terms': [36, 48, 60]  # Shorter terms for balloon
            },
            'interest_only': {
                'rate_premium': 0.008,  # 80 bps premium
                'origination_fee': 0.012,  # 1.2% of loan amount
                'servicing_cost_annual': 0.006,  # Moderate servicing costs
                'min_credit_score': 620,
                'max_dti': 0.35,  # Lower DTI for payment shock protection
                'min_term_months': 36,  # Minimum term for IO
                'max_term_months': 120,  # Maximum term for IO
                'preferred_terms': [60, 72, 84, 96, 120]  # Longer terms for IO
            }
        }
        
        # Term-specific risk adjustments
        self.term_risk_adjustments = {
            24: -0.002,   # Short term: -20 bps (lower risk)
            36: -0.001,   # Medium-short: -10 bps
            48: 0.000,    # Standard term: no adjustment
            60: 0.001,    # Medium-long: +10 bps
            72: 0.002,    # Long term: +20 bps
            84: 0.003,    # Longer term: +30 bps
            96: 0.004,    # Very long: +40 bps
            120: 0.005    # Ultra long: +50 bps
        }
    
    def get_term_risk_adjustment(self, term_months: int) -> float:
        """
        Get risk adjustment for specific loan term.
        
        Parameters:
        -----------
        term_months : int
            Loan term in months
            
        Returns:
        --------
        float : Risk adjustment (basis points as decimal)
        """
        # Interpolate for terms not explicitly defined
        if term_months in self.term_risk_adjustments:
            return self.term_risk_adjustments[term_months]
        
        # Linear interpolation for intermediate terms
        sorted_terms = sorted(self.term_risk_adjustments.keys())
        
        if term_months <= sorted_terms[0]:
            return self.term_risk_adjustments[sorted_terms[0]]
        elif term_months >= sorted_terms[-1]:
            return self.term_risk_adjustments[sorted_terms[-1]]
        
        # Find surrounding terms and interpolate
        for i in range(len(sorted_terms) - 1):
            if sorted_terms[i] <= term_months <= sorted_terms[i + 1]:
                lower_term = sorted_terms[i]
                upper_term = sorted_terms[i + 1]
                lower_adj = self.term_risk_adjustments[lower_term]
                upper_adj = self.term_risk_adjustments[upper_term]
                
                # Linear interpolation
                weight = (term_months - lower_term) / (upper_term - lower_term)
                return lower_adj + weight * (upper_adj - lower_adj)
        
        return 0.0  # Fallback
    
    def calculate_expected_profit_with_term(self, borrower_data: pd.DataFrame, 
                                          payment_pattern: str, 
                                          loan_amount: float,
                                          term_months: int,
                                          time_horizon: int = None) -> Dict:
        """
        Calculate expected profit for specific payment pattern and term combination.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        payment_pattern : str
            Payment pattern to evaluate
        loan_amount : float
            Loan amount
        term_months : int
            Loan term in months
        time_horizon : int
            Analysis time horizon (defaults to term_months)
            
        Returns:
        --------
        Dict : Expected profit metrics including term analysis
        """
        if time_horizon is None:
            time_horizon = min(term_months + 12, 120)  # Extend horizon slightly beyond term
        
        # Get payment pattern parameters
        pattern_params = self.payment_patterns[payment_pattern]
        
        # Calculate term-adjusted interest rate
        base_rate_premium = pattern_params['rate_premium']
        term_risk_adjustment = self.get_term_risk_adjustment(term_months)
        total_rate_premium = base_rate_premium + term_risk_adjustment
        
        interest_rate = self.base_interest_rate + total_rate_premium
        
        # Calculate payment schedule with specific term
        payment_schedule = self.calculate_payment_schedule(
            loan_amount, interest_rate, term_months, payment_pattern
        )
        
        # Prepare borrower data for survival prediction
        borrower_profile = borrower_data.copy()
        borrower_profile['loan_amount'] = loan_amount
        borrower_profile['loan_term_months'] = term_months
        borrower_profile['is_balloon_payment'] = 1 if payment_pattern == 'balloon' else 0
        borrower_profile['is_interest_only'] = 1 if payment_pattern == 'interest_only' else 0
        
        # Predict survival probabilities over the term
        time_points = list(range(1, min(time_horizon, term_months) + 1))
        
        survival_probs = []
        cumulative_default_prob = 0
        
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
                # Fallback: longer terms have higher cumulative default risk
                base_monthly_hazard = 0.008  # Base monthly default probability
                term_multiplier = 1 + (term_months - 48) * 0.001  # Increase for longer terms
                monthly_survival = 1 - (base_monthly_hazard * term_multiplier)
                survival_prob = monthly_survival ** t
                survival_probs.append(max(0.1, survival_prob))
        
        # Calculate expected cash flows with term-specific considerations
        revenues = []
        costs = []
        defaults = []
        
        # Initial fees
        origination_revenue = loan_amount * pattern_params['origination_fee']
        initial_cost = loan_amount  # Cost of funds for loan origination
        
        monthly_servicing_cost = loan_amount * pattern_params['servicing_cost_annual'] / 12
        
        # Monthly cash flows over the term
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
            if payment_pattern == 'interest_only':
                io_months = min(term_months // 2, 60)  # IO period
                if t <= io_months:
                    expected_payment = payment_schedule['monthly_payment'] * current_survival
                else:
                    expected_payment = payment_schedule.get('amort_payment', payment_schedule['monthly_payment']) * current_survival
            else:
                expected_payment = payment_schedule['monthly_payment'] * current_survival
            
            revenues.append(expected_payment)
            
            # Servicing costs (only for surviving loans)
            costs.append(monthly_servicing_cost * current_survival)
        
        # Terminal value calculations
        balloon_recovery = 0
        if payment_pattern == 'balloon' and len(survival_probs) > 0:
            final_survival = survival_probs[min(term_months-1, len(survival_probs)-1)]
            balloon_recovery = payment_schedule['balloon_payment'] * final_survival
        
        # Calculate total expected loss with term consideration
        final_survival_prob = survival_probs[-1] if survival_probs else 0.5
        total_default_prob = 1 - final_survival_prob
        
        # Term-adjusted recovery rate (longer terms may have lower recovery)
        term_recovery_adjustment = max(0.8, 1 - (term_months - 48) * 0.002)  # Reduce recovery for very long terms
        adjusted_recovery_rate = self.recovery_rate * term_recovery_adjustment
        
        expected_loss = loan_amount * total_default_prob * (1 - adjusted_recovery_rate)
        expected_recovery = loan_amount * total_default_prob * adjusted_recovery_rate
        
        # Calculate NPV with term-specific discount rate
        monthly_discount_rate = (self.cost_of_funds + term_risk_adjustment) / 12
        
        npv_revenues = origination_revenue + balloon_recovery
        npv_costs = initial_cost + expected_loss
        
        for i, (rev, cost) in enumerate(zip(revenues, costs)):
            discount_factor = (1 + monthly_discount_rate) ** (i + 1)
            npv_revenues += rev / discount_factor
            npv_costs += cost / discount_factor
        
        expected_profit = npv_revenues - npv_costs
        
        # Risk and return metrics
        expected_return = expected_profit / loan_amount if loan_amount > 0 else 0
        risk_adjusted_return = expected_return / max(total_default_prob, 0.01)
        
        # Term-specific metrics
        annualized_return = expected_return * (12 / term_months)  # Annualized return
        profit_per_month = expected_profit / term_months
        
        return {
            'payment_pattern': payment_pattern,
            'term_months': term_months,
            'expected_profit': expected_profit,
            'expected_return': expected_return,
            'annualized_return': annualized_return,
            'profit_per_month': profit_per_month,
            'risk_adjusted_return': risk_adjusted_return,
            'total_default_prob': total_default_prob,
            'expected_loss': expected_loss,
            'total_revenue': npv_revenues,
            'total_cost': npv_costs,
            'interest_rate': interest_rate,
            'rate_premium': total_rate_premium,
            'term_risk_adjustment': term_risk_adjustment,
            'monthly_payment': payment_schedule['monthly_payment'],
            'balloon_payment': payment_schedule.get('balloon_payment', 0),
            'adjusted_recovery_rate': adjusted_recovery_rate
        }
    
    def optimize_payment_pattern_and_term(self, borrower_data: pd.DataFrame,
                                        loan_amount: float,
                                        consider_terms: List[int] = None) -> Dict:
        """
        Optimize both payment pattern and loan term for maximum expected profit.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        loan_amount : float
            Requested loan amount
        consider_terms : List[int]
            Specific terms to evaluate (defaults to standard terms)
            
        Returns:
        --------
        Dict : Optimization results with best pattern and term combination
        """
        if consider_terms is None:
            # Default terms to consider
            consider_terms = [24, 36, 48, 60, 72, 84]
        
        results = {}
        eligible_combinations = []
        
        # Evaluate each payment pattern and term combination
        for pattern in ['installment', 'balloon', 'interest_only']:
            pattern_params = self.payment_patterns[pattern]
            pattern_results = {}
            
            for term in consider_terms:
                # Check if term is valid for this payment pattern
                if (term < pattern_params['min_term_months'] or 
                    term > pattern_params['max_term_months']):
                    pattern_results[term] = {
                        'eligible': False,
                        'reason': f'Term {term} months outside allowed range for {pattern}'
                    }
                    continue
                
                # Check underwriting criteria
                if not self.check_underwriting_criteria(borrower_data, pattern):
                    pattern_results[term] = {
                        'eligible': False,
                        'reason': 'Does not meet underwriting criteria'
                    }
                    continue
                
                # Calculate expected profit for this combination
                try:
                    profit_analysis = self.calculate_expected_profit_with_term(
                        borrower_data, pattern, loan_amount, term
                    )
                    profit_analysis['eligible'] = True
                    pattern_results[term] = profit_analysis
                    eligible_combinations.append((pattern, term, profit_analysis))
                except Exception as e:
                    pattern_results[term] = {
                        'eligible': False,
                        'reason': f'Calculation error: {str(e)}'
                    }
            
            results[pattern] = pattern_results
        
        # Find optimal combinations
        best_combination = None
        best_profit_combination = None
        best_risk_adjusted_combination = None
        best_annualized_combination = None
        
        if eligible_combinations:
            # Highest expected profit
            best_profit_combination = max(eligible_combinations, 
                                        key=lambda x: x[2]['expected_profit'])
            
            # Highest risk-adjusted return
            best_risk_adjusted_combination = max(eligible_combinations,
                                               key=lambda x: x[2]['risk_adjusted_return'])
            
            # Highest annualized return
            best_annualized_combination = max(eligible_combinations,
                                            key=lambda x: x[2]['annualized_return'])
            
            # Overall best (using expected profit as primary criterion)
            best_combination = best_profit_combination
        
        return {
            'borrower_analysis': results,
            'eligible_combinations': len(eligible_combinations),
            'best_combination': best_combination,
            'best_profit_combination': best_profit_combination,
            'best_risk_adjusted_combination': best_risk_adjusted_combination,
            'best_annualized_combination': best_annualized_combination,
            'loan_amount': loan_amount,
            'terms_considered': consider_terms
        }
    
    def analyze_portfolio_with_term_optimization(self, borrower_portfolio: pd.DataFrame,
                                               loan_amounts: List[float],
                                               consider_terms: List[int] = None) -> pd.DataFrame:
        """
        Optimize payment patterns and terms for an entire portfolio.
        
        Parameters:
        -----------
        borrower_portfolio : pd.DataFrame
            Portfolio of borrower characteristics
        loan_amounts : List[float]
            Loan amounts for each borrower
        consider_terms : List[int]
            Terms to consider for optimization
            
        Returns:
        --------
        pd.DataFrame : Portfolio optimization results with term analysis
        """
        if consider_terms is None:
            consider_terms = [24, 36, 48, 60, 72, 84]
        
        portfolio_results = []
        
        print(f"Optimizing payment patterns and terms for {len(borrower_portfolio)} borrowers...")
        print(f"Considering terms: {consider_terms} months")
        
        for idx, (_, borrower) in enumerate(borrower_portfolio.iterrows()):
            if idx < len(loan_amounts):
                loan_amount = loan_amounts[idx]
            else:
                loan_amount = np.mean(loan_amounts)
            
            # Convert Series to DataFrame for optimization
            borrower_df = borrower.to_frame().T
            borrower_df['id'] = f'borrower_{idx}'
            
            # Optimize payment pattern and term
            optimization_result = self.optimize_payment_pattern_and_term(
                borrower_df, loan_amount, consider_terms
            )
            
            # Extract results
            best_combo = optimization_result['best_combination']
            best_profit_combo = optimization_result['best_profit_combination']
            best_risk_adj_combo = optimization_result['best_risk_adjusted_combination']
            best_annual_combo = optimization_result['best_annualized_combination']
            
            result = {
                'borrower_id': idx,
                'loan_amount': loan_amount,
                'eligible_combinations': optimization_result['eligible_combinations']
            }
            
            # Best overall combination
            if best_combo:
                pattern, term, analysis = best_combo
                result.update({
                    'recommended_pattern': pattern,
                    'recommended_term': term,
                    'expected_profit': analysis['expected_profit'],
                    'expected_return': analysis['expected_return'],
                    'annualized_return': analysis['annualized_return'],
                    'profit_per_month': analysis['profit_per_month'],
                    'default_probability': analysis['total_default_prob'],
                    'interest_rate': analysis['interest_rate'],
                    'monthly_payment': analysis['monthly_payment'],
                    'balloon_payment': analysis.get('balloon_payment', 0)
                })
            else:
                result.update({
                    'recommended_pattern': None,
                    'recommended_term': None,
                    'expected_profit': np.nan,
                    'expected_return': np.nan,
                    'annualized_return': np.nan,
                    'profit_per_month': np.nan,
                    'default_probability': np.nan,
                    'interest_rate': np.nan,
                    'monthly_payment': np.nan,
                    'balloon_payment': np.nan
                })
            
            # Alternative optimizations
            if best_profit_combo and best_profit_combo != best_combo:
                pattern, term, analysis = best_profit_combo
                result.update({
                    'alt_profit_pattern': pattern,
                    'alt_profit_term': term,
                    'alt_profit_value': analysis['expected_profit']
                })
            
            if best_risk_adj_combo:
                pattern, term, analysis = best_risk_adj_combo
                result.update({
                    'best_risk_adj_pattern': pattern,
                    'best_risk_adj_term': term,
                    'best_risk_adj_return': analysis['risk_adjusted_return']
                })
            
            if best_annual_combo:
                pattern, term, analysis = best_annual_combo
                result.update({
                    'best_annual_pattern': pattern,
                    'best_annual_term': term,
                    'best_annual_return': analysis['annualized_return']
                })
            
            portfolio_results.append(result)
            
            if (idx + 1) % 25 == 0:
                print(f"Processed {idx + 1} borrowers...")
        
        return pd.DataFrame(portfolio_results)
    
    def plot_term_optimization_analysis(self, portfolio_results: pd.DataFrame):
        """
        Create comprehensive visualizations of term and pattern optimization.
        
        Parameters:
        -----------
        portfolio_results : pd.DataFrame
            Results from portfolio optimization with terms
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Filter to approved loans only
        approved_loans = portfolio_results.dropna(subset=['recommended_pattern'])
        
        if len(approved_loans) == 0:
            print("No approved loans to visualize")
            return
        
        # 1. Term distribution
        ax1 = axes[0, 0]
        term_counts = approved_loans['recommended_term'].value_counts().sort_index()
        bars = ax1.bar(term_counts.index, term_counts.values, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Loan Term (months)')
        ax1.set_ylabel('Number of Loans')
        ax1.set_title('Distribution of Optimal Loan Terms')
        
        # Add count labels
        for bar, count in zip(bars, term_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        # 2. Payment pattern distribution
        ax2 = axes[0, 1]
        pattern_counts = approved_loans['recommended_pattern'].value_counts()
        colors = ['green', 'red', 'orange']
        ax2.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%',
               colors=colors[:len(pattern_counts)])
        ax2.set_title('Optimal Payment Pattern Distribution')
        
        # 3. Term vs Pattern heatmap
        ax3 = axes[0, 2]
        pattern_term_counts = approved_loans.groupby(['recommended_pattern', 'recommended_term']).size().unstack(fill_value=0)
        
        if not pattern_term_counts.empty:
            sns.heatmap(pattern_term_counts, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_title('Payment Pattern vs Term Combinations')
            ax3.set_xlabel('Loan Term (months)')
            ax3.set_ylabel('Payment Pattern')
        
        # 4. Expected profit by term
        ax4 = axes[1, 0]
        profit_by_term = approved_loans.groupby('recommended_term')['expected_profit'].agg(['mean', 'std', 'count'])
        
        if not profit_by_term.empty:
            ax4.errorbar(profit_by_term.index, profit_by_term['mean'], 
                        yerr=profit_by_term['std'], marker='o', capsize=5)
            ax4.set_xlabel('Loan Term (months)')
            ax4.set_ylabel('Average Expected Profit ($)')
            ax4.set_title('Expected Profit by Loan Term')
            ax4.grid(True, alpha=0.3)
        
        # 5. Annualized return by term
        ax5 = axes[1, 1]
        annual_return_by_term = approved_loans.groupby('recommended_term')['annualized_return'].mean()
        
        if not annual_return_by_term.empty:
            bars = ax5.bar(annual_return_by_term.index, annual_return_by_term.values, alpha=0.7, color='lightgreen')
            ax5.set_xlabel('Loan Term (months)')
            ax5.set_ylabel('Annualized Return')
            ax5.set_title('Annualized Return by Loan Term')
            
            # Add value labels
            for bar, value in zip(bars, annual_return_by_term.values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.1%}', ha='center', va='bottom')
        
        # 6. Default risk by term
        ax6 = axes[1, 2]
        risk_by_term = approved_loans.groupby('recommended_term')['default_probability'].mean()
        
        if not risk_by_term.empty:
            ax6.plot(risk_by_term.index, risk_by_term.values, marker='o', linewidth=2, color='red')
            ax6.set_xlabel('Loan Term (months)')
            ax6.set_ylabel('Average Default Probability')
            ax6.set_title('Default Risk by Loan Term')
            ax6.grid(True, alpha=0.3)
        
        # 7. Term vs Profit scatter by pattern
        ax7 = axes[2, 0]
        patterns = approved_loans['recommended_pattern'].unique()
        colors = ['green', 'red', 'orange']
        
        for i, pattern in enumerate(patterns):
            pattern_data = approved_loans[approved_loans['recommended_pattern'] == pattern]
            if len(pattern_data) > 0:
                ax7.scatter(pattern_data['recommended_term'], pattern_data['expected_profit'],
                          label=pattern.title(), alpha=0.7, color=colors[i % len(colors)])
        
        ax7.set_xlabel('Loan Term (months)')
        ax7.set_ylabel('Expected Profit ($)')
        ax7.set_title('Term vs Profit by Payment Pattern')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Portfolio efficiency frontier (Risk vs Return)
        ax8 = axes[2, 1]
        if len(approved_loans) > 0:
            # Color by term
            scatter = ax8.scatter(approved_loans['default_probability'], 
                                approved_loans['annualized_return'],
                                c=approved_loans['recommended_term'], 
                                cmap='viridis', alpha=0.7)
            ax8.set_xlabel('Default Probability')
            ax8.set_ylabel('Annualized Return')
            ax8.set_title('Risk-Return Profile (colored by term)')
            plt.colorbar(scatter, ax=ax8, label='Term (months)')
            ax8.grid(True, alpha=0.3)
        
        # 9. Capital efficiency by term
        ax9 = axes[2, 2]
        capital_efficiency = approved_loans.groupby('recommended_term')['profit_per_month'].mean()
        
        if not capital_efficiency.empty:
            bars = ax9.bar(capital_efficiency.index, capital_efficiency.values, alpha=0.7, color='purple')
            ax9.set_xlabel('Loan Term (months)')
            ax9.set_ylabel('Profit per Month ($)')
            ax9.set_title('Capital Efficiency by Term')
            
            # Add value labels
            for bar, value in zip(bars, capital_efficiency.values):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(capital_efficiency.values)*0.01,
                        f'${value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_term_optimization_insights(self, portfolio_results: pd.DataFrame) -> Dict:
        """
        Generate business insights from term and pattern optimization.
        
        Parameters:
        -----------
        portfolio_results : pd.DataFrame
            Results from portfolio optimization
            
        Returns:
        --------
        Dict : Business insights and recommendations
        """
        approved_loans = portfolio_results.dropna(subset=['recommended_pattern'])
        
        if len(approved_loans) == 0:
            return {"error": "No approved loans to analyze"}
        
        # Term analysis
        avg_term = approved_loans['recommended_term'].mean()
        term_distribution = approved_loans['recommended_term'].value_counts(normalize=True).sort_index()
        
        # Pattern analysis
        pattern_distribution = approved_loans['recommended_pattern'].value_counts(normalize=True)
        
        # Performance metrics
        avg_profit = approved_loans['expected_profit'].mean()
        avg_annual_return = approved_loans['annualized_return'].mean()
        avg_default_risk = approved_loans['default_probability'].mean()
        
        # Term-specific insights
        profit_by_term = approved_loans.groupby('recommended_term')['expected_profit'].mean()
        best_profit_term = profit_by_term.idxmax()
        best_profit_value = profit_by_term.max()
        
        annual_return_by_term = approved_loans.groupby('recommended_term')['annualized_return'].mean()
        best_annual_term = annual_return_by_term.idxmax()
        best_annual_value = annual_return_by_term.max()
        
        risk_by_term = approved_loans.groupby('recommended_term')['default_probability'].mean()
        lowest_risk_term = risk_by_term.idxmin()
        lowest_risk_value = risk_by_term.min()
        
        # Portfolio composition
        total_portfolio_value = approved_loans['expected_profit'].sum()
        
        insights = {
            'portfolio_summary': {
                'total_approved_loans': len(approved_loans),
                'total_portfolio_value': total_portfolio_value,
                'average_loan_term': avg_term,
                'average_expected_profit': avg_profit,
                'average_annualized_return': avg_annual_return,
                'average_default_risk': avg_default_risk
            },
            
            'term_optimization': {
                'optimal_profit_term': best_profit_term,
                'optimal_profit_value': best_profit_value,
                'optimal_annual_return_term': best_annual_term,
                'optimal_annual_return_value': best_annual_value,
                'lowest_risk_term': lowest_risk_term,
                'lowest_risk_value': lowest_risk_value,
                'term_distribution': term_distribution.to_dict()
            },
            
            'pattern_insights': {
                'pattern_distribution': pattern_distribution.to_dict(),
                'dominant_pattern': pattern_distribution.idxmax(),
                'pattern_concentration': pattern_distribution.max()
            },
            
            'strategic_recommendations': []
        }
        
        # Generate recommendations
        recommendations = insights['strategic_recommendations']
        
        # Term-based recommendations
        if avg_term > 60:
            recommendations.append("Portfolio skews toward longer terms - monitor default risk concentration")
        elif avg_term < 36:
            recommendations.append("Portfolio favors shorter terms - consider longer terms for higher profits")
        
        if best_profit_term != best_annual_term:
            recommendations.append(f"Trade-off between profit maximization ({best_profit_term}m) and annual returns ({best_annual_term}m)")
        
        # Concentration risk
        max_term_concentration = term_distribution.max()
        if max_term_concentration > 0.4:
            recommendations.append(f"High concentration in {term_distribution.idxmax()}-month terms ({max_term_concentration:.1%}) - diversify term mix")
        
        # Pattern-term combinations
        pattern_term_analysis = approved_loans.groupby(['recommended_pattern', 'recommended_term']).size()
        if len(pattern_term_analysis) > 0:
            dominant_combo = pattern_term_analysis.idxmax()
            combo_concentration = pattern_term_analysis.max() / len(approved_loans)
            
            if combo_concentration > 0.3:
                recommendations.append(f"High concentration in {dominant_combo[0]} {dominant_combo[1]}-month loans ({combo_concentration:.1%})")
        
        return insights


def main():
    """Demonstrate the enhanced loan term and pattern optimization."""
    print("="*80)
    print("ENHANCED LOAN PROFIT OPTIMIZATION WITH TERM DURATION")
    print("="*80)
    
    # Initialize survival model
    print("\n1. Initializing enhanced survival model...")
    survival_model = CreditSurvivalModel(random_state=42)
    
    # Generate training data
    train_data, test_data, val_data = survival_model.generate_sample_data(
        n_subjects=400, max_time=72, test_size=0.3, val_size=0.1
    )
    
    # Fit the model
    survival_model.fit_cox_model(penalizer=0.01)
    
    # Initialize enhanced optimizer
    print("\n2. Initializing term-optimized profit optimizer...")
    optimizer = LoanTermProfitOptimizer(
        survival_model=survival_model,
        base_interest_rate=0.12,
        cost_of_funds=0.03,
        recovery_rate=0.40
    )
    
    print("\n3. INDIVIDUAL BORROWER TERM OPTIMIZATION")
    print("-" * 50)
    
    # Example individual optimization
    portfolio_data = test_data.groupby('id').first().reset_index()
    sample_borrower = portfolio_data.iloc[0:1].copy()
    sample_borrower['id'] = 'sample_borrower'
    sample_loan_amount = 30000
    
    # Optimize both pattern and term
    individual_result = optimizer.optimize_payment_pattern_and_term(
        sample_borrower, sample_loan_amount, consider_terms=[24, 36, 48, 60, 72, 84]
    )
    
    print(f"Sample Borrower Analysis (Loan Amount: ${sample_loan_amount:,}):")
    
    if individual_result['best_combination']:
        best_pattern, best_term, best_analysis = individual_result['best_combination']
        print(f"\nOPTIMAL RECOMMENDATION:")
        print(f"  Payment Pattern: {best_pattern.title()}")
        print(f"  Loan Term: {best_term} months")
        print(f"  Expected Profit: ${best_analysis['expected_profit']:,.0f}")
        print(f"  Expected Return: {best_analysis['expected_return']:.2%}")
        print(f"  Annualized Return: {best_analysis['annualized_return']:.2%}")
        print(f"  Profit per Month: ${best_analysis['profit_per_month']:,.0f}")
        print(f"  Default Probability: {best_analysis['total_default_prob']:.1%}")
        print(f"  Interest Rate: {best_analysis['interest_rate']:.2%}")
        print(f"  Monthly Payment: ${best_analysis['monthly_payment']:,.2f}")
        
        if best_analysis.get('balloon_payment', 0) > 0:
            print(f"  Balloon Payment: ${best_analysis['balloon_payment']:,.0f}")
    
    # Show alternatives
    print(f"\nAlternative Optimizations:")
    
    if individual_result['best_profit_combination']:
        pattern, term, analysis = individual_result['best_profit_combination']
        print(f"  Highest Profit: {pattern.title()} {term}m (${analysis['expected_profit']:,.0f})")
    
    if individual_result['best_risk_adjusted_combination']:
        pattern, term, analysis = individual_result['best_risk_adjusted_combination']
        print(f"  Best Risk-Adjusted: {pattern.title()} {term}m ({analysis['risk_adjusted_return']:.2f})")
    
    if individual_result['best_annualized_combination']:
        pattern, term, analysis = individual_result['best_annualized_combination']
        print(f"  Highest Annualized Return: {pattern.title()} {term}m ({analysis['annualized_return']:.2%})")
    
    print(f"\nTotal Eligible Combinations: {individual_result['eligible_combinations']}")
    
    print("\n4. PORTFOLIO OPTIMIZATION WITH TERMS")
    print("-" * 50)
    
    # Portfolio optimization
    portfolio_sample = portfolio_data.head(40)  # Smaller sample for faster processing
    loan_amounts = np.random.uniform(15000, 50000, len(portfolio_sample))
    
    # Run portfolio optimization
    portfolio_results = optimizer.analyze_portfolio_with_term_optimization(
        portfolio_sample, loan_amounts.tolist(), consider_terms=[24, 36, 48, 60, 72, 84]
    )
    
    print(f"\nPortfolio Optimization Results:")
    approved_count = len(portfolio_results.dropna(subset=['recommended_pattern']))
    total_count = len(portfolio_results)
    
    print(f"Total Applications: {total_count}")
    print(f"Approved Loans: {approved_count}")
    print(f"Approval Rate: {approved_count/total_count:.1%}")
    
    if approved_count > 0:
        approved_loans = portfolio_results.dropna(subset=['recommended_pattern'])
        
        print(f"\nOptimization Summary:")
        print(f"  Average Loan Term: {approved_loans['recommended_term'].mean():.1f} months")
        print(f"  Average Expected Profit: ${approved_loans['expected_profit'].mean():,.0f}")
        print(f"  Average Annualized Return: {approved_loans['annualized_return'].mean():.2%}")
        print(f"  Total Portfolio Value: ${approved_loans['expected_profit'].sum():,.0f}")
        
        print(f"\nTerm Distribution:")
        term_dist = approved_loans['recommended_term'].value_counts().sort_index()
        for term, count in term_dist.items():
            percentage = count / len(approved_loans) * 100
            avg_profit = approved_loans[approved_loans['recommended_term'] == term]['expected_profit'].mean()
            print(f"  {term} months: {count} loans ({percentage:.1f}%) - Avg Profit: ${avg_profit:,.0f}")
        
        print(f"\nPattern Distribution:")
        pattern_dist = approved_loans['recommended_pattern'].value_counts()
        for pattern, count in pattern_dist.items():
            percentage = count / len(approved_loans) * 100
            avg_term = approved_loans[approved_loans['recommended_pattern'] == pattern]['recommended_term'].mean()
            avg_profit = approved_loans[approved_loans['recommended_pattern'] == pattern]['expected_profit'].mean()
            print(f"  {pattern.title()}: {count} loans ({percentage:.1f}%) - Avg Term: {avg_term:.1f}m - Avg Profit: ${avg_profit:,.0f}")
    
    # Generate insights
    print("\n5. STRATEGIC INSIGHTS FROM TERM OPTIMIZATION")
    print("-" * 50)
    
    insights = optimizer.generate_term_optimization_insights(portfolio_results)
    
    if 'error' not in insights:
        print("Portfolio Summary:")
        summary = insights['portfolio_summary']
        for key, value in summary.items():
            if isinstance(value, float):
                if 'return' in key or 'risk' in key:
                    print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
                elif 'value' in key or 'profit' in key:
                    print(f"  {key.replace('_', ' ').title()}: ${value:,.0f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value:.1f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nTerm Optimization Insights:")
        term_insights = insights['term_optimization']
        print(f"  Most Profitable Term: {term_insights['optimal_profit_term']} months (${term_insights['optimal_profit_value']:,.0f})")
        print(f"  Best Annual Return Term: {term_insights['optimal_annual_return_term']} months ({term_insights['optimal_annual_return_value']:.2%})")
        print(f"  Lowest Risk Term: {term_insights['lowest_risk_term']} months ({term_insights['lowest_risk_value']:.1%})")
        
        print(f"\nStrategic Recommendations:")
        for i, rec in enumerate(insights['strategic_recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Generate visualizations
    print("\n6. Generating enhanced visualizations...")
    optimizer.plot_term_optimization_analysis(portfolio_results)
    
    print("\n" + "="*80)
    print("ENHANCED TERM OPTIMIZATION ANALYSIS COMPLETE!")
    print("="*80)
    
    return optimizer, portfolio_results, insights


if __name__ == "__main__":
    optimizer, results, insights = main() 