"""
Enhanced Payment Pattern and Term Decision Engine with Joint 4-Variable Optimization

This module integrates payment pattern, loan term, loan amount, and loan pricing 
optimization to make optimal lending decisions that maximize portfolio profit. 
The engine evaluates all viable combinations across 4 dimensions to recommend 
the optimal loan structure for each borrower.

Key Features:
- Joint optimization across 4 variables: pattern, term, amount, price
- Loan amount optimization as percentage of requested amount
- Dynamic pricing strategies with margin optimization
- Sensitivity analysis for approval rates across all variables
- Multi-objective decision criteria with enhanced constraints
- Portfolio-level optimization with concentration limits
- Comprehensive decision reporting and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from credit_survival_model import CreditSurvivalModel
from loan_term_profit_optimizer import LoanTermProfitOptimizer
from typing import Dict, List, Tuple, Optional
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class JointOptimizationEngine:
    """
    Advanced decision engine that jointly optimizes payment patterns, loan terms,
    loan amounts, and pricing strategies for maximum portfolio profitability.
    """
    
    def __init__(self, survival_model: CreditSurvivalModel, 
                 base_interest_rate: float = 0.12,
                 cost_of_funds: float = 0.03,
                 recovery_rate: float = 0.40):
        """
        Initialize the joint optimization engine.
        
        Parameters:
        -----------
        survival_model : CreditSurvivalModel
            Fitted survival model
        base_interest_rate : float
            Base interest rate before adjustments
        cost_of_funds : float
            Cost of funds rate
        recovery_rate : float
            Expected recovery rate on defaults
        """
        self.survival_model = survival_model
        self.base_interest_rate = base_interest_rate
        self.cost_of_funds = cost_of_funds
        self.recovery_rate = recovery_rate
        
        # Initialize the term optimizer
        self.term_optimizer = LoanTermProfitOptimizer(
            survival_model, base_interest_rate, cost_of_funds, recovery_rate
        )
        
        # 4-Variable Decision Space
        
        # 1. Loan Amount Strategies (as multiplier of requested amount)
        self.loan_amount_strategies = {
            'conservative': 0.75,   # 75% of requested amount
            'moderate': 0.85,       # 85% of requested amount
            'standard': 1.0,        # 100% of requested amount
            'generous': 1.15,       # 115% of requested amount
            'aggressive': 1.3       # 130% of requested amount
        }
        
        # 2. Pricing Strategies (adjustment to base rate)
        self.pricing_strategies = {
            'competitive': -0.015,   # -150 bps (competitive pricing)
            'market': -0.005,        # -50 bps (slight discount)
            'standard': 0.0,         # Base rate (standard pricing)
            'premium': 0.01,         # +100 bps (premium pricing)
            'high_margin': 0.02      # +200 bps (high margin)
        }
        
        # 3. Payment Patterns (from term optimizer)
        self.payment_patterns = ['installment', 'balloon', 'interest_only']
        
        # 4. Loan Terms
        self.standard_terms = [24, 36, 48, 60, 72, 84]
        
        # Enhanced decision criteria for 4-variable optimization
        self.decision_criteria = {
            'min_expected_return': 0.04,      # 4% minimum expected return
            'min_annualized_return': 0.06,    # 6% minimum annualized return
            'max_default_probability': 0.30,  # 30% maximum default risk
            'min_profit_per_loan': 300,       # $300 minimum profit per loan
            'min_profit_per_month': 15,       # $15 minimum monthly profit
            'min_interest_margin': 0.03,      # 300 bps minimum spread over cost of funds
            'max_loan_to_income_ratio': 0.40, # 40% max loan-to-income
            'min_debt_service_coverage': 1.25, # 1.25x minimum coverage
            'max_portfolio_concentration_pattern': 0.60,  # 60% max single pattern
            'max_portfolio_concentration_term': 0.50,     # 50% max single term
            'max_balloon_concentration': 0.40,            # 40% max balloon loans
            'max_high_loan_amount_concentration': 0.30,   # 30% max high-amount loans
            'preferred_term_range': (24, 72),
            'preferred_amount_range': (0.8, 1.2),        # 80-120% of requested
            'profit_weight': 0.4,             # Weight for total profit in scoring
            'return_weight': 0.3,             # Weight for annualized return
            'efficiency_weight': 0.3          # Weight for capital efficiency
        }
        
        # Portfolio tracking for concentration limits
        self.portfolio_tracker = {
            'approved_loans': [],
            'pattern_counts': {'installment': 0, 'balloon': 0, 'interest_only': 0},
            'term_counts': {},
            'amount_strategy_counts': {},
            'pricing_strategy_counts': {},
            'total_approved': 0,
            'total_portfolio_amount': 0
        }
    
    def reset_portfolio_tracker(self):
        """Reset the portfolio tracker for new analysis."""
        self.portfolio_tracker = {
            'approved_loans': [],
            'pattern_counts': {'installment': 0, 'balloon': 0, 'interest_only': 0},
            'term_counts': {},
            'amount_strategy_counts': {},
            'pricing_strategy_counts': {},
            'total_approved': 0,
            'total_portfolio_amount': 0
        }
    
    def calculate_loan_specific_metrics(self, borrower_data: pd.DataFrame, 
                                      requested_amount: float,
                                      actual_amount: float,
                                      interest_rate: float) -> Dict:
        """
        Calculate loan-specific underwriting metrics.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        requested_amount : float
            Originally requested loan amount
        actual_amount : float
            Proposed loan amount
        interest_rate : float
            Proposed interest rate
            
        Returns:
        --------
        Dict : Loan-specific metrics
        """
        borrower = borrower_data.iloc[0] if len(borrower_data) > 0 else borrower_data
        
        # Income-based metrics
        income = borrower.get('income', borrower.get('annual_income', 50000))
        loan_to_income = actual_amount / income
        
        # Debt service coverage (simplified)
        monthly_income = income / 12
        estimated_monthly_payment = actual_amount * (interest_rate / 12) * 1.2  # Rough estimate
        debt_service_coverage = monthly_income / estimated_monthly_payment
        
        # Amount adjustment impact
        amount_adjustment_ratio = actual_amount / requested_amount
        
        return {
            'loan_to_income_ratio': loan_to_income,
            'debt_service_coverage': debt_service_coverage,
            'amount_adjustment_ratio': amount_adjustment_ratio,
            'interest_margin': interest_rate - self.cost_of_funds,
            'monthly_income': monthly_income,
            'estimated_monthly_payment': estimated_monthly_payment
        }
    
    def evaluate_4variable_combination(self, borrower_data: pd.DataFrame,
                                     requested_amount: float,
                                     amount_strategy: str,
                                     pricing_strategy: str,
                                     payment_pattern: str,
                                     term_months: int) -> Dict:
        """
        Evaluate a specific combination of all 4 variables.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        requested_amount : float
            Originally requested amount
        amount_strategy : str
            Loan amount strategy key
        pricing_strategy : str
            Pricing strategy key
        payment_pattern : str
            Payment pattern
        term_months : int
            Loan term in months
            
        Returns:
        --------
        Dict : Comprehensive evaluation results
        """
        # Calculate actual loan parameters
        amount_multiplier = self.loan_amount_strategies[amount_strategy]
        actual_amount = requested_amount * amount_multiplier
        
        pricing_adjustment = self.pricing_strategies[pricing_strategy]
        
        # Get pattern-specific parameters
        pattern_params = self.term_optimizer.payment_patterns[payment_pattern]
        
        # Calculate term risk adjustment
        term_risk_adjustment = self.term_optimizer.get_term_risk_adjustment(term_months)
        
        # Final interest rate
        interest_rate = (self.base_interest_rate + 
                        pattern_params['rate_premium'] + 
                        term_risk_adjustment + 
                        pricing_adjustment)
        
        # Calculate loan-specific metrics
        loan_metrics = self.calculate_loan_specific_metrics(
            borrower_data, requested_amount, actual_amount, interest_rate
        )
        
        # Check basic underwriting criteria
        eligibility_checks = {
            'loan_to_income': loan_metrics['loan_to_income_ratio'] <= self.decision_criteria['max_loan_to_income_ratio'],
            'debt_service': loan_metrics['debt_service_coverage'] >= self.decision_criteria['min_debt_service_coverage'],
            'interest_margin': loan_metrics['interest_margin'] >= self.decision_criteria['min_interest_margin'],
            'pattern_underwriting': self.check_pattern_underwriting(borrower_data, payment_pattern),
            'term_range': pattern_params['min_term_months'] <= term_months <= pattern_params['max_term_months']
        }
        
        # If basic checks fail, return rejection
        if not all(eligibility_checks.values()):
            failed_checks = [k for k, v in eligibility_checks.items() if not v]
            return {
                'eligible': False,
                'rejection_reason': f"Failed checks: {', '.join(failed_checks)}",
                'amount_strategy': amount_strategy,
                'pricing_strategy': pricing_strategy,
                'payment_pattern': payment_pattern,
                'term_months': term_months,
                'actual_amount': actual_amount,
                'interest_rate': interest_rate,
                'loan_metrics': loan_metrics
            }
        
        # Temporarily adjust the optimizer's base rate for this calculation
        original_base_rate = self.term_optimizer.base_interest_rate
        self.term_optimizer.base_interest_rate = interest_rate - pattern_params['rate_premium'] - term_risk_adjustment
        
        try:
            # Calculate expected profit using the term optimizer
            profit_analysis = self.term_optimizer.calculate_expected_profit_with_term(
                borrower_data, payment_pattern, actual_amount, term_months
            )
            
            # Calculate enhanced metrics
            composite_score = self.calculate_4variable_composite_score(profit_analysis, loan_metrics)
            
            result = {
                'eligible': True,
                'amount_strategy': amount_strategy,
                'pricing_strategy': pricing_strategy,
                'payment_pattern': payment_pattern,
                'term_months': term_months,
                'actual_amount': actual_amount,
                'requested_amount': requested_amount,
                'interest_rate': interest_rate,
                'loan_metrics': loan_metrics,
                'profit_analysis': profit_analysis,
                'composite_score': composite_score,
                'amount_multiplier': amount_multiplier,
                'pricing_adjustment': pricing_adjustment
            }
            
        except Exception as e:
            result = {
                'eligible': False,
                'rejection_reason': f"Calculation error: {str(e)}",
                'amount_strategy': amount_strategy,
                'pricing_strategy': pricing_strategy,
                'payment_pattern': payment_pattern,
                'term_months': term_months,
                'actual_amount': actual_amount,
                'interest_rate': interest_rate,
                'loan_metrics': loan_metrics
            }
        
        finally:
            # Restore original base rate
            self.term_optimizer.base_interest_rate = original_base_rate
        
        return result
    
    def check_pattern_underwriting(self, borrower_data: pd.DataFrame, payment_pattern: str) -> bool:
        """Check if borrower meets pattern-specific underwriting criteria."""
        borrower = borrower_data.iloc[0] if len(borrower_data) > 0 else borrower_data
        pattern_params = self.term_optimizer.payment_patterns[payment_pattern]
        
        credit_score = borrower.get('credit_score', 650)
        dti_ratio = borrower.get('debt_to_income_ratio', 0.35)
        
        return (credit_score >= pattern_params['min_credit_score'] and 
                dti_ratio <= pattern_params['max_dti'])
    
    def calculate_4variable_composite_score(self, profit_analysis: Dict, loan_metrics: Dict) -> float:
        """
        Calculate composite score for 4-variable optimization.
        
        Parameters:
        -----------
        profit_analysis : Dict
            Results from profit analysis
        loan_metrics : Dict
            Loan-specific metrics
            
        Returns:
        --------
        float : Composite score (higher is better)
        """
        # Normalize profit metrics
        profit_score = min(1.0, profit_analysis['expected_profit'] / 10000)  # Cap at $10k
        return_score = min(1.0, profit_analysis['annualized_return'] / 0.20)  # Cap at 20%
        efficiency_score = min(1.0, profit_analysis['profit_per_month'] / 500)  # Cap at $500/month
        
        # Risk adjustments
        risk_penalty = max(0.1, 1 - profit_analysis['total_default_prob'] / 0.40)
        
        # Loan structure quality bonuses
        amount_bonus = 1.0
        if 0.9 <= loan_metrics['amount_adjustment_ratio'] <= 1.1:
            amount_bonus = 1.1  # Bonus for amounts close to requested
        
        margin_bonus = min(1.2, loan_metrics['interest_margin'] / 0.06)  # Bonus for higher margins
        
        # Weighted composite score with structure bonuses
        base_score = (
            self.decision_criteria['profit_weight'] * profit_score +
            self.decision_criteria['return_weight'] * return_score * risk_penalty +
            self.decision_criteria['efficiency_weight'] * efficiency_score
        )
        
        return base_score * amount_bonus * margin_bonus
    
    def check_4variable_portfolio_constraints(self, amount_strategy: str, 
                                            pricing_strategy: str,
                                            payment_pattern: str, 
                                            term_months: int,
                                            actual_amount: float) -> Tuple[bool, List[str]]:
        """
        Check portfolio concentration constraints for 4-variable optimization.
        
        Returns:
        --------
        Tuple[bool, List[str]] : (allowed, list of violations)
        """
        violations = []
        total_approved = self.portfolio_tracker['total_approved']
        
        if total_approved == 0:
            return True, []
        
        # Pattern concentration
        pattern_count = self.portfolio_tracker['pattern_counts'].get(payment_pattern, 0)
        pattern_concentration = (pattern_count + 1) / (total_approved + 1)
        
        if pattern_concentration > self.decision_criteria['max_portfolio_concentration_pattern']:
            violations.append(f"Pattern concentration ({payment_pattern}: {pattern_concentration:.1%}) exceeds limit")
        
        # Balloon special limit
        if payment_pattern == 'balloon':
            balloon_concentration = (self.portfolio_tracker['pattern_counts']['balloon'] + 1) / (total_approved + 1)
            if balloon_concentration > self.decision_criteria['max_balloon_concentration']:
                violations.append(f"Balloon concentration ({balloon_concentration:.1%}) exceeds limit")
        
        # Term concentration
        term_count = self.portfolio_tracker['term_counts'].get(term_months, 0)
        term_concentration = (term_count + 1) / (total_approved + 1)
        
        if term_concentration > self.decision_criteria['max_portfolio_concentration_term']:
            violations.append(f"Term concentration ({term_months}m: {term_concentration:.1%}) exceeds limit")
        
        # Amount strategy concentration (high loan amounts)
        if amount_strategy in ['generous', 'aggressive']:
            high_amount_count = (self.portfolio_tracker['amount_strategy_counts'].get('generous', 0) + 
                               self.portfolio_tracker['amount_strategy_counts'].get('aggressive', 0))
            high_amount_concentration = (high_amount_count + 1) / (total_approved + 1)
            
            if high_amount_concentration > self.decision_criteria['max_high_loan_amount_concentration']:
                violations.append(f"High loan amount concentration ({high_amount_concentration:.1%}) exceeds limit")
        
        return len(violations) == 0, violations
    
    def update_4variable_portfolio_tracker(self, amount_strategy: str, pricing_strategy: str,
                                         payment_pattern: str, term_months: int,
                                         actual_amount: float, loan_data: Dict):
        """Update portfolio tracker with 4-variable loan approval."""
        self.portfolio_tracker['approved_loans'].append(loan_data)
        self.portfolio_tracker['pattern_counts'][payment_pattern] += 1
        self.portfolio_tracker['term_counts'][term_months] = self.portfolio_tracker['term_counts'].get(term_months, 0) + 1
        self.portfolio_tracker['amount_strategy_counts'][amount_strategy] = self.portfolio_tracker['amount_strategy_counts'].get(amount_strategy, 0) + 1
        self.portfolio_tracker['pricing_strategy_counts'][pricing_strategy] = self.portfolio_tracker['pricing_strategy_counts'].get(pricing_strategy, 0) + 1
        self.portfolio_tracker['total_approved'] += 1
        self.portfolio_tracker['total_portfolio_amount'] += actual_amount
    
    def joint_4variable_optimization(self, borrower_data: pd.DataFrame,
                                   requested_amount: float,
                                   check_portfolio_constraints: bool = True) -> Dict:
        """
        Perform comprehensive 4-variable joint optimization.
        
        Parameters:
        -----------
        borrower_data : pd.DataFrame
            Borrower characteristics
        requested_amount : float
            Originally requested loan amount
        check_portfolio_constraints : bool
            Whether to apply portfolio concentration limits
            
        Returns:
        --------
        Dict : Complete optimization results across all 4 variables
        """
        viable_options = []
        rejected_options = []
        
        # Generate all combinations of the 4 variables
        all_combinations = list(product(
            self.loan_amount_strategies.keys(),  # Amount strategies
            self.pricing_strategies.keys(),      # Pricing strategies  
            self.payment_patterns,               # Payment patterns
            self.standard_terms                  # Terms
        ))
        
        print(f"Evaluating {len(all_combinations)} combinations for 4-variable optimization...")
        
        for amount_strategy, pricing_strategy, payment_pattern, term_months in all_combinations:
            # Evaluate this specific combination
            result = self.evaluate_4variable_combination(
                borrower_data, requested_amount, amount_strategy, 
                pricing_strategy, payment_pattern, term_months
            )
            
            if not result['eligible']:
                rejected_options.append(result)
                continue
            
            # Apply business criteria
            profit_analysis = result['profit_analysis']
            loan_metrics = result['loan_metrics']
            
            business_criteria_passed = True
            rejection_reasons = []
            
            # Enhanced business criteria for 4-variable optimization
            if profit_analysis['expected_return'] < self.decision_criteria['min_expected_return']:
                business_criteria_passed = False
                rejection_reasons.append(f"Expected return {profit_analysis['expected_return']:.2%} below minimum")
            
            if profit_analysis['annualized_return'] < self.decision_criteria['min_annualized_return']:
                business_criteria_passed = False
                rejection_reasons.append(f"Annualized return {profit_analysis['annualized_return']:.2%} below minimum")
            
            if profit_analysis['total_default_prob'] > self.decision_criteria['max_default_probability']:
                business_criteria_passed = False
                rejection_reasons.append(f"Default risk {profit_analysis['total_default_prob']:.1%} exceeds maximum")
            
            if profit_analysis['expected_profit'] < self.decision_criteria['min_profit_per_loan']:
                business_criteria_passed = False
                rejection_reasons.append(f"Expected profit ${profit_analysis['expected_profit']:,.0f} below minimum")
            
            if profit_analysis['profit_per_month'] < self.decision_criteria['min_profit_per_month']:
                business_criteria_passed = False
                rejection_reasons.append(f"Profit per month ${profit_analysis['profit_per_month']:,.0f} below minimum")
            
            # Portfolio constraints
            if check_portfolio_constraints:
                portfolio_allowed, portfolio_violations = self.check_4variable_portfolio_constraints(
                    amount_strategy, pricing_strategy, payment_pattern, term_months, result['actual_amount']
                )
                if not portfolio_allowed:
                    business_criteria_passed = False
                    rejection_reasons.extend(portfolio_violations)
            
            if not business_criteria_passed:
                result['rejection_reason'] = '; '.join(rejection_reasons)
                rejected_options.append(result)
            else:
                viable_options.append(result)
        
        # Find optimal solutions across different objectives
        if viable_options:
            # Sort by composite score for overall best
            viable_options.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Best overall
            best_overall = viable_options[0]
            
            # Alternative rankings
            best_profit = max(viable_options, key=lambda x: x['profit_analysis']['expected_profit'])
            best_return = max(viable_options, key=lambda x: x['profit_analysis']['annualized_return'])
            best_efficiency = max(viable_options, key=lambda x: x['profit_analysis']['profit_per_month'])
            best_margin = max(viable_options, key=lambda x: x['loan_metrics']['interest_margin'])
            
            # Update portfolio tracker if using constraints
            if check_portfolio_constraints:
                loan_data = {
                    'amount_strategy': best_overall['amount_strategy'],
                    'pricing_strategy': best_overall['pricing_strategy'],
                    'payment_pattern': best_overall['payment_pattern'],
                    'term_months': best_overall['term_months'],
                    'actual_amount': best_overall['actual_amount'],
                    'expected_profit': best_overall['profit_analysis']['expected_profit'],
                    'composite_score': best_overall['composite_score']
                }
                
                self.update_4variable_portfolio_tracker(
                    best_overall['amount_strategy'], best_overall['pricing_strategy'],
                    best_overall['payment_pattern'], best_overall['term_months'],
                    best_overall['actual_amount'], loan_data
                )
            
            decision = 'APPROVE'
        else:
            decision = 'DECLINE'
            best_overall = None
            best_profit = None
            best_return = None
            best_efficiency = None
            best_margin = None
        
        return {
            'decision': decision,
            'requested_amount': requested_amount,
            'total_combinations_evaluated': len(all_combinations),
            'viable_options_count': len(viable_options),
            'rejected_options_count': len(rejected_options),
            'best_overall': best_overall,
            'alternative_bests': {
                'highest_profit': best_profit,
                'highest_return': best_return,
                'most_efficient': best_efficiency,
                'highest_margin': best_margin
            },
            'viable_options': viable_options,
            'rejected_options': rejected_options,
            'portfolio_constraints_applied': check_portfolio_constraints
        }
    
    def analyze_approval_rate_sensitivity(self, borrower_sample: pd.DataFrame,
                                        requested_amounts: List[float],
                                        sensitivity_variable: str = 'all') -> Dict:
        """
        Analyze sensitivity of approval rates to each of the 4 variables.
        
        Parameters:
        -----------
        borrower_sample : pd.DataFrame
            Sample of borrowers for sensitivity analysis
        requested_amounts : List[float]
            Corresponding requested amounts
        sensitivity_variable : str
            Which variable to analyze ('amount', 'pricing', 'pattern', 'term', 'all')
            
        Returns:
        --------
        Dict : Sensitivity analysis results
        """
        print(f"Starting approval rate sensitivity analysis for: {sensitivity_variable}")
        
        sensitivity_results = {}
        
        if sensitivity_variable in ['amount', 'all']:
            # Amount strategy sensitivity
            amount_results = {}
            
            for strategy in self.loan_amount_strategies.keys():
                self.reset_portfolio_tracker()
                approvals = 0
                total = 0
                
                for idx, (_, borrower) in enumerate(borrower_sample.iterrows()):
                    if idx >= len(requested_amounts):
                        break
                        
                    borrower_df = borrower.to_frame().T
                    borrower_df['id'] = f'sensitivity_borrower_{idx}'
                    
                    requested_amount = requested_amounts[idx]
                    
                    # Test only this amount strategy with default other variables
                    result = self.evaluate_4variable_combination(
                        borrower_df, requested_amount, strategy, 'standard', 
                        'installment', 48
                    )
                    
                    total += 1
                    if result['eligible'] and result.get('profit_analysis', {}).get('expected_return', 0) >= self.decision_criteria['min_expected_return']:
                        approvals += 1
                
                amount_results[strategy] = {
                    'approval_rate': approvals / total if total > 0 else 0,
                    'approvals': approvals,
                    'total': total,
                    'multiplier': self.loan_amount_strategies[strategy]
                }
            
            sensitivity_results['amount_strategy'] = amount_results
        
        if sensitivity_variable in ['pricing', 'all']:
            # Pricing strategy sensitivity
            pricing_results = {}
            
            for strategy in self.pricing_strategies.keys():
                self.reset_portfolio_tracker()
                approvals = 0
                total = 0
                
                for idx, (_, borrower) in enumerate(borrower_sample.iterrows()):
                    if idx >= len(requested_amounts):
                        break
                        
                    borrower_df = borrower.to_frame().T
                    borrower_df['id'] = f'sensitivity_borrower_{idx}'
                    
                    requested_amount = requested_amounts[idx]
                    
                    # Test only this pricing strategy with default other variables
                    result = self.evaluate_4variable_combination(
                        borrower_df, requested_amount, 'standard', strategy,
                        'installment', 48
                    )
                    
                    total += 1
                    if result['eligible'] and result.get('profit_analysis', {}).get('expected_return', 0) >= self.decision_criteria['min_expected_return']:
                        approvals += 1
                
                pricing_results[strategy] = {
                    'approval_rate': approvals / total if total > 0 else 0,
                    'approvals': approvals,
                    'total': total,
                    'rate_adjustment': self.pricing_strategies[strategy]
                }
            
            sensitivity_results['pricing_strategy'] = pricing_results
        
        if sensitivity_variable in ['pattern', 'all']:
            # Payment pattern sensitivity
            pattern_results = {}
            
            for pattern in self.payment_patterns:
                self.reset_portfolio_tracker()
                approvals = 0
                total = 0
                
                for idx, (_, borrower) in enumerate(borrower_sample.iterrows()):
                    if idx >= len(requested_amounts):
                        break
                        
                    borrower_df = borrower.to_frame().T
                    borrower_df['id'] = f'sensitivity_borrower_{idx}'
                    
                    requested_amount = requested_amounts[idx]
                    
                    # Test only this pattern with default other variables
                    result = self.evaluate_4variable_combination(
                        borrower_df, requested_amount, 'standard', 'standard',
                        pattern, 48
                    )
                    
                    total += 1
                    if result['eligible'] and result.get('profit_analysis', {}).get('expected_return', 0) >= self.decision_criteria['min_expected_return']:
                        approvals += 1
                
                pattern_results[pattern] = {
                    'approval_rate': approvals / total if total > 0 else 0,
                    'approvals': approvals,
                    'total': total
                }
            
            sensitivity_results['payment_pattern'] = pattern_results
        
        if sensitivity_variable in ['term', 'all']:
            # Term sensitivity
            term_results = {}
            
            for term in self.standard_terms:
                self.reset_portfolio_tracker()
                approvals = 0
                total = 0
                
                for idx, (_, borrower) in enumerate(borrower_sample.iterrows()):
                    if idx >= len(requested_amounts):
                        break
                        
                    borrower_df = borrower.to_frame().T
                    borrower_df['id'] = f'sensitivity_borrower_{idx}'
                    
                    requested_amount = requested_amounts[idx]
                    
                    # Test only this term with default other variables
                    result = self.evaluate_4variable_combination(
                        borrower_df, requested_amount, 'standard', 'standard',
                        'installment', term
                    )
                    
                    total += 1
                    if result['eligible'] and result.get('profit_analysis', {}).get('expected_return', 0) >= self.decision_criteria['min_expected_return']:
                        approvals += 1
                
                term_results[term] = {
                    'approval_rate': approvals / total if total > 0 else 0,
                    'approvals': approvals,
                    'total': total
                }
            
            sensitivity_results['loan_term'] = term_results
        
        return sensitivity_results

    def batch_4variable_optimization(self, applications: pd.DataFrame,
                                   requested_amounts: List[float],
                                   apply_portfolio_constraints: bool = True) -> pd.DataFrame:
        """
        Perform 4-variable optimization for a batch of applications.
        
        Parameters:
        -----------
        applications : pd.DataFrame
            Multiple borrower applications
        requested_amounts : List[float]
            Corresponding requested amounts
        apply_portfolio_constraints : bool
            Whether to apply portfolio concentration limits
            
        Returns:
        --------
        pd.DataFrame : Batch optimization results
        """
        # Reset portfolio tracker
        if apply_portfolio_constraints:
            self.reset_portfolio_tracker()
        
        results = []
        
        print(f"4-Variable Joint Optimization of {len(applications)} loan applications...")
        print(f"Constraints: {'Enabled' if apply_portfolio_constraints else 'Disabled'}")
        
        for idx, (_, borrower) in enumerate(applications.iterrows()):
            borrower_df = borrower.to_frame().T
            borrower_df['id'] = f'applicant_{idx}'
            
            requested_amount = requested_amounts[idx] if idx < len(requested_amounts) else np.mean(requested_amounts)
            
            # Joint 4-variable optimization
            optimization_result = self.joint_4variable_optimization(
                borrower_df, requested_amount, apply_portfolio_constraints
            )
            
            # Extract results
            result = {
                'applicant_id': idx,
                'requested_amount': requested_amount,
                'decision': optimization_result['decision'],
                'total_combinations_evaluated': optimization_result['total_combinations_evaluated'],
                'viable_options_count': optimization_result['viable_options_count'],
                'rejected_options_count': optimization_result['rejected_options_count']
            }
            
            # Add optimal solution details if approved
            if optimization_result['decision'] == 'APPROVE':
                best = optimization_result['best_overall']
                profit_analysis = best['profit_analysis']
                loan_metrics = best['loan_metrics']
                
                result.update({
                    'amount_strategy': best['amount_strategy'],
                    'pricing_strategy': best['pricing_strategy'],
                    'payment_pattern': best['payment_pattern'],
                    'term_months': best['term_months'],
                    'actual_amount': best['actual_amount'],
                    'interest_rate': best['interest_rate'],
                    'amount_multiplier': best['amount_multiplier'],
                    'pricing_adjustment': best['pricing_adjustment'],
                    'expected_profit': profit_analysis['expected_profit'],
                    'expected_return': profit_analysis['expected_return'],
                    'annualized_return': profit_analysis['annualized_return'],
                    'profit_per_month': profit_analysis['profit_per_month'],
                    'default_probability': profit_analysis['total_default_prob'],
                    'composite_score': best['composite_score'],
                    'loan_to_income': loan_metrics['loan_to_income_ratio'],
                    'debt_service_coverage': loan_metrics['debt_service_coverage'],
                    'interest_margin': loan_metrics['interest_margin']
                })
            else:
                result.update({
                    'amount_strategy': None,
                    'pricing_strategy': None,
                    'payment_pattern': None,
                    'term_months': None,
                    'actual_amount': None,
                    'interest_rate': None,
                    'amount_multiplier': None,
                    'pricing_adjustment': None,
                    'expected_profit': np.nan,
                    'expected_return': np.nan,
                    'annualized_return': np.nan,
                    'profit_per_month': np.nan,
                    'default_probability': np.nan,
                    'composite_score': np.nan,
                    'loan_to_income': np.nan,
                    'debt_service_coverage': np.nan,
                    'interest_margin': np.nan
                })
            
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} applications...")
        
        return pd.DataFrame(results)
    
    def plot_4variable_sensitivity_analysis(self, sensitivity_results: Dict):
        """
        Create visualizations for 4-variable sensitivity analysis.
        
        Parameters:
        -----------
        sensitivity_results : Dict
            Results from analyze_approval_rate_sensitivity
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Amount Strategy Sensitivity
        if 'amount_strategy' in sensitivity_results:
            ax1 = axes[0, 0]
            amount_data = sensitivity_results['amount_strategy']
            
            strategies = list(amount_data.keys())
            approval_rates = [amount_data[s]['approval_rate'] for s in strategies]
            multipliers = [amount_data[s]['multiplier'] for s in strategies]
            
            colors = ['red' if rate < 0.05 else 'orange' if rate < 0.15 else 'green' for rate in approval_rates]
            bars = ax1.bar(strategies, approval_rates, color=colors, alpha=0.7)
            
            ax1.set_ylabel('Approval Rate')
            ax1.set_title('Approval Rate Sensitivity to Loan Amount Strategy')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add multiplier labels
            for bar, mult, rate in zip(bars, multipliers, approval_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mult:.0%}\n({rate:.1%})', ha='center', va='bottom')
        
        # 2. Pricing Strategy Sensitivity
        if 'pricing_strategy' in sensitivity_results:
            ax2 = axes[0, 1]
            pricing_data = sensitivity_results['pricing_strategy']
            
            strategies = list(pricing_data.keys())
            approval_rates = [pricing_data[s]['approval_rate'] for s in strategies]
            rate_adjustments = [pricing_data[s]['rate_adjustment'] for s in strategies]
            
            colors = ['green' if adj < 0 else 'orange' if adj == 0 else 'red' for adj in rate_adjustments]
            bars = ax2.bar(strategies, approval_rates, color=colors, alpha=0.7)
            
            ax2.set_ylabel('Approval Rate')
            ax2.set_title('Approval Rate Sensitivity to Pricing Strategy')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add rate adjustment labels
            for bar, adj, rate in zip(bars, rate_adjustments, approval_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{adj:+.1%}\n({rate:.1%})', ha='center', va='bottom')
        
        # 3. Payment Pattern Sensitivity
        if 'payment_pattern' in sensitivity_results:
            ax3 = axes[1, 0]
            pattern_data = sensitivity_results['payment_pattern']
            
            patterns = list(pattern_data.keys())
            approval_rates = [pattern_data[p]['approval_rate'] for p in patterns]
            
            colors = ['green', 'red', 'orange']
            bars = ax3.bar(patterns, approval_rates, color=colors[:len(patterns)], alpha=0.7)
            
            ax3.set_ylabel('Approval Rate')
            ax3.set_title('Approval Rate Sensitivity to Payment Pattern')
            
            # Add approval rate labels
            for bar, rate in zip(bars, approval_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
        
        # 4. Term Sensitivity
        if 'loan_term' in sensitivity_results:
            ax4 = axes[1, 1]
            term_data = sensitivity_results['loan_term']
            
            terms = sorted(term_data.keys())
            approval_rates = [term_data[t]['approval_rate'] for t in terms]
            
            ax4.plot(terms, approval_rates, marker='o', linewidth=2, markersize=8, color='blue')
            ax4.fill_between(terms, approval_rates, alpha=0.3, color='blue')
            
            ax4.set_xlabel('Loan Term (months)')
            ax4.set_ylabel('Approval Rate')
            ax4.set_title('Approval Rate Sensitivity to Loan Term')
            ax4.grid(True, alpha=0.3)
            
            # Add approval rate labels
            for term, rate in zip(terms, approval_rates):
                ax4.text(term, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_4variable_optimization_results(self, batch_results: pd.DataFrame):
        """
        Create comprehensive visualizations for 4-variable optimization results.
        
        Parameters:
        -----------
        batch_results : pd.DataFrame
            Results from batch_4variable_optimization
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        approved_loans = batch_results[batch_results['decision'] == 'APPROVE']
        total_applications = len(batch_results)
        
        # 1. Overall approval results
        ax1 = axes[0, 0]
        approval_counts = batch_results['decision'].value_counts()
        colors = ['green' if decision == 'APPROVE' else 'red' for decision in approval_counts.index]
        bars = ax1.bar(approval_counts.index, approval_counts.values, color=colors, alpha=0.7)
        ax1.set_title('4-Variable Optimization Results')
        ax1.set_ylabel('Number of Applications')
        
        for bar, count in zip(bars, approval_counts.values):
            percentage = count / total_applications * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # 2. Amount strategy distribution
        ax2 = axes[0, 1]
        if len(approved_loans) > 0:
            amount_counts = approved_loans['amount_strategy'].value_counts()
            colors_amount = ['lightblue', 'blue', 'darkblue', 'navy', 'black']
            ax2.pie(amount_counts.values, labels=amount_counts.index, autopct='%1.1f%%',
                   colors=colors_amount[:len(amount_counts)])
            ax2.set_title('Optimal Amount Strategy Distribution')
        
        # 3. Pricing strategy distribution
        ax3 = axes[0, 2]
        if len(approved_loans) > 0:
            pricing_counts = approved_loans['pricing_strategy'].value_counts()
            colors_pricing = ['lightgreen', 'green', 'yellow', 'orange', 'red']
            ax3.pie(pricing_counts.values, labels=pricing_counts.index, autopct='%1.1f%%',
                   colors=colors_pricing[:len(pricing_counts)])
            ax3.set_title('Optimal Pricing Strategy Distribution')
        
        # 4. 4D Optimization effectiveness
        ax4 = axes[1, 0]
        if len(batch_results) > 0:
            avg_combinations = batch_results['total_combinations_evaluated'].mean()
            avg_viable = batch_results['viable_options_count'].mean()
            avg_rejected = batch_results['rejected_options_count'].mean()
            
            categories = ['Total\nCombinations', 'Viable\nOptions', 'Rejected\nOptions']
            values = [avg_combinations, avg_viable, avg_rejected]
            colors = ['blue', 'green', 'red']
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_ylabel('Average Count per Application')
            ax4.set_title('4-Variable Optimization Effectiveness')
            
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.0f}', ha='center', va='bottom')
        
        # 5. Amount adjustment impact
        ax5 = axes[1, 1]
        if len(approved_loans) > 0:
            scatter = ax5.scatter(approved_loans['amount_multiplier'], 
                                approved_loans['expected_profit'],
                                c=approved_loans['composite_score'], 
                                cmap='viridis', alpha=0.7, s=60)
            ax5.set_xlabel('Amount Multiplier (% of Requested)')
            ax5.set_ylabel('Expected Profit ($)')
            ax5.set_title('Amount Strategy vs Profit (colored by score)')
            plt.colorbar(scatter, ax=ax5, label='Composite Score')
            ax5.grid(True, alpha=0.3)
        
        # 6. Pricing impact on margins
        ax6 = axes[1, 2]
        if len(approved_loans) > 0:
            pricing_profit = approved_loans.groupby('pricing_strategy')['interest_margin'].agg(['mean', 'std'])
            
            if not pricing_profit.empty:
                ax6.bar(pricing_profit.index, pricing_profit['mean'], 
                       yerr=pricing_profit['std'], capsize=5, alpha=0.7, color='gold')
                ax6.set_ylabel('Interest Margin (decimal)')
                ax6.set_title('Interest Margin by Pricing Strategy')
                ax6.tick_params(axis='x', rotation=45)
        
        # 7. 4-Variable combination heatmap
        ax7 = axes[2, 0]
        if len(approved_loans) > 0:
            # Create pattern-term combination counts
            combo_pivot = approved_loans.groupby(['payment_pattern', 'term_months']).size().unstack(fill_value=0)
            
            if not combo_pivot.empty:
                sns.heatmap(combo_pivot, annot=True, fmt='d', cmap='Blues', ax=ax7)
                ax7.set_title('Pattern-Term Combination Frequency')
                ax7.set_xlabel('Term (months)')
                ax7.set_ylabel('Payment Pattern')
        
        # 8. Risk-return with 4 variables
        ax8 = axes[2, 1]
        if len(approved_loans) > 0:
            # Size by amount multiplier, color by pricing strategy
            pricing_colors = {'competitive': 'green', 'market': 'blue', 'standard': 'yellow', 
                            'premium': 'orange', 'high_margin': 'red'}
            
            for strategy in approved_loans['pricing_strategy'].unique():
                strategy_data = approved_loans[approved_loans['pricing_strategy'] == strategy]
                if len(strategy_data) > 0:
                    ax8.scatter(strategy_data['default_probability'], 
                              strategy_data['annualized_return'],
                              label=strategy, alpha=0.7,
                              s=strategy_data['amount_multiplier'] * 100,
                              color=pricing_colors.get(strategy, 'gray'))
            
            ax8.set_xlabel('Default Probability')
            ax8.set_ylabel('Annualized Return')
            ax8.set_title('Risk-Return by Pricing (size=amount)')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Portfolio value distribution
        ax9 = axes[2, 2]
        if len(approved_loans) > 0:
            # Show distribution of expected profits
            profits = approved_loans['expected_profit']
            ax9.hist(profits, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax9.axvline(x=profits.mean(), color='red', linestyle='--', 
                       label=f'Mean: ${profits.mean():,.0f}')
            ax9.set_xlabel('Expected Profit ($)')
            ax9.set_ylabel('Number of Loans')
            ax9.set_title('Expected Profit Distribution')
            ax9.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_4variable_insights(self, batch_results: pd.DataFrame, 
                                  sensitivity_results: Dict = None) -> Dict:
        """
        Generate comprehensive insights from 4-variable optimization.
        
        Parameters:
        -----------
        batch_results : pd.DataFrame
            Results from batch optimization
        sensitivity_results : Dict
            Results from sensitivity analysis
            
        Returns:
        --------
        Dict : Comprehensive business insights
        """
        approved_loans = batch_results[batch_results['decision'] == 'APPROVE']
        total_applications = len(batch_results)
        
        if len(approved_loans) == 0:
            return {"error": "No approved loans to analyze"}
        
        insights = {
            'optimization_summary': {
                'total_applications': total_applications,
                'approved_loans': len(approved_loans),
                'approval_rate': len(approved_loans) / total_applications,
                'avg_combinations_evaluated': batch_results['total_combinations_evaluated'].mean(),
                'avg_viable_options': batch_results['viable_options_count'].mean(),
                'optimization_efficiency': batch_results['viable_options_count'].sum() / batch_results['total_combinations_evaluated'].sum()
            },
            
            'portfolio_metrics': {
                'total_portfolio_value': approved_loans['expected_profit'].sum(),
                'average_expected_profit': approved_loans['expected_profit'].mean(),
                'average_annualized_return': approved_loans['annualized_return'].mean(),
                'average_composite_score': approved_loans['composite_score'].mean(),
                'total_loan_volume': approved_loans['actual_amount'].sum(),
                'average_loan_amount': approved_loans['actual_amount'].mean(),
                'average_amount_multiplier': approved_loans['amount_multiplier'].mean(),
                'average_interest_margin': approved_loans['interest_margin'].mean()
            },
            
            'strategy_distribution': {
                'amount_strategies': approved_loans['amount_strategy'].value_counts(normalize=True).to_dict(),
                'pricing_strategies': approved_loans['pricing_strategy'].value_counts(normalize=True).to_dict(),
                'payment_patterns': approved_loans['payment_pattern'].value_counts(normalize=True).to_dict(),
                'term_distribution': approved_loans['term_months'].value_counts(normalize=True).to_dict()
            },
            
            'risk_profile': {
                'average_default_probability': approved_loans['default_probability'].mean(),
                'default_risk_range': (approved_loans['default_probability'].min(), 
                                     approved_loans['default_probability'].max()),
                'average_loan_to_income': approved_loans['loan_to_income'].mean(),
                'average_debt_service_coverage': approved_loans['debt_service_coverage'].mean()
            }
        }
        
        # Add sensitivity insights if available
        if sensitivity_results:
            insights['sensitivity_analysis'] = {}
            
            for variable, results in sensitivity_results.items():
                if results:
                    approval_rates = [r['approval_rate'] for r in results.values()]
                    insights['sensitivity_analysis'][variable] = {
                        'highest_approval_strategy': max(results.keys(), key=lambda k: results[k]['approval_rate']),
                        'lowest_approval_strategy': min(results.keys(), key=lambda k: results[k]['approval_rate']),
                        'approval_rate_range': (min(approval_rates), max(approval_rates)),
                        'approval_rate_volatility': np.std(approval_rates)
                    }
        
        # Strategic recommendations
        insights['strategic_recommendations'] = []
        
        # Amount strategy recommendations
        amount_dist = insights['strategy_distribution']['amount_strategies']
        dominant_amount = max(amount_dist.keys(), key=lambda k: amount_dist[k])
        if amount_dist[dominant_amount] > 0.6:
            insights['strategic_recommendations'].append(
                f"High concentration in {dominant_amount} amount strategy ({amount_dist[dominant_amount]:.1%}) - consider diversification"
            )
        
        # Pricing strategy recommendations
        pricing_dist = insights['strategy_distribution']['pricing_strategies']
        if 'high_margin' in pricing_dist and pricing_dist['high_margin'] > 0.4:
            insights['strategic_recommendations'].append(
                "High concentration in high-margin pricing - monitor competitive position"
            )
        elif 'competitive' in pricing_dist and pricing_dist['competitive'] > 0.5:
            insights['strategic_recommendations'].append(
                "Heavy use of competitive pricing - ensure adequate profitability"
            )
        
        # Portfolio quality recommendations
        avg_score = insights['portfolio_metrics']['average_composite_score']
        if avg_score > 0.8:
            insights['strategic_recommendations'].append(
                f"Excellent portfolio quality (avg score: {avg_score:.2f}) - consider expanding criteria"
            )
        elif avg_score < 0.5:
            insights['strategic_recommendations'].append(
                f"Low portfolio quality (avg score: {avg_score:.2f}) - tighten criteria"
            )
        
        # Efficiency recommendations
        efficiency = insights['optimization_summary']['optimization_efficiency']
        if efficiency < 0.1:
            insights['strategic_recommendations'].append(
                f"Low optimization efficiency ({efficiency:.1%}) - review decision criteria"
            )
        
        return insights


def main():
    """Demonstrate the comprehensive 4-variable joint optimization engine."""
    print("="*80)
    print("4-VARIABLE JOINT OPTIMIZATION ENGINE")
    print("PATTERN × TERM × AMOUNT × PRICING")
    print("="*80)
    
    # Initialize survival model
    print("\n1. Setting up survival model and joint optimization engine...")
    
    survival_model = CreditSurvivalModel(random_state=42)
    train_data, test_data, val_data = survival_model.generate_sample_data(
        n_subjects=200, max_time=72, test_size=0.3, val_size=0.1
    )
    survival_model.fit_cox_model(penalizer=0.01)
    
    # Initialize 4-variable optimization engine
    joint_engine = JointOptimizationEngine(
        survival_model=survival_model,
        base_interest_rate=0.12,
        cost_of_funds=0.03,
        recovery_rate=0.40
    )
    
    print(f"Decision Variables:")
    print(f"  Amount Strategies: {list(joint_engine.loan_amount_strategies.keys())}")
    print(f"  Pricing Strategies: {list(joint_engine.pricing_strategies.keys())}")
    print(f"  Payment Patterns: {joint_engine.payment_patterns}")
    print(f"  Loan Terms: {joint_engine.standard_terms}")
    print(f"  Total Combinations: {len(joint_engine.loan_amount_strategies) * len(joint_engine.pricing_strategies) * len(joint_engine.payment_patterns) * len(joint_engine.standard_terms)}")
    
    print("\n2. INDIVIDUAL 4-VARIABLE OPTIMIZATION EXAMPLE")
    print("-" * 60)
    
    # Example individual optimization
    sample_applications = test_data.groupby('id').first().reset_index()
    sample_borrower = sample_applications.iloc[0:1].copy()
    sample_borrower['id'] = 'joint_opt_borrower'
    
    requested_amount = 40000
    
    # 4-variable optimization
    joint_result = joint_engine.joint_4variable_optimization(
        sample_borrower, requested_amount, check_portfolio_constraints=False
    )
    
    print(f"4-Variable Optimization Results for ${requested_amount:,} request:")
    print(f"  Decision: {joint_result['decision']}")
    print(f"  Combinations Evaluated: {joint_result['total_combinations_evaluated']}")
    print(f"  Viable Options: {joint_result['viable_options_count']}")
    print(f"  Rejected Options: {joint_result['rejected_options_count']}")
    
    if joint_result['decision'] == 'APPROVE':
        best = joint_result['best_overall']
        profit_analysis = best['profit_analysis']
        loan_metrics = best['loan_metrics']
        
        print(f"\nOPTIMAL LOAN STRUCTURE:")
        print(f"  Amount Strategy: {best['amount_strategy']} ({best['amount_multiplier']:.0%} of requested)")
        print(f"  Actual Amount: ${best['actual_amount']:,.0f}")
        print(f"  Pricing Strategy: {best['pricing_strategy']} ({best['pricing_adjustment']:+.1%} adjustment)")
        print(f"  Interest Rate: {best['interest_rate']:.2%}")
        print(f"  Payment Pattern: {best['payment_pattern'].title()}")
        print(f"  Loan Term: {best['term_months']} months")
        print(f"  Composite Score: {best['composite_score']:.3f}")
        
        print(f"\nFINANCIAL PROJECTIONS:")
        print(f"  Expected Profit: ${profit_analysis['expected_profit']:,.0f}")
        print(f"  Expected Return: {profit_analysis['expected_return']:.2%}")
        print(f"  Annualized Return: {profit_analysis['annualized_return']:.2%}")
        print(f"  Profit per Month: ${profit_analysis['profit_per_month']:,.0f}")
        print(f"  Default Probability: {profit_analysis['total_default_prob']:.1%}")
        
        print(f"\nLOAN METRICS:")
        print(f"  Loan-to-Income Ratio: {loan_metrics['loan_to_income_ratio']:.1%}")
        print(f"  Debt Service Coverage: {loan_metrics['debt_service_coverage']:.2f}x")
        print(f"  Interest Margin: {loan_metrics['interest_margin']:.2%}")
        
        # Show top alternatives
        alternatives = joint_result['alternative_bests']
        print(f"\nALTERNATIVE OPTIMIZATIONS:")
        
        if alternatives['highest_profit'] != best:
            alt = alternatives['highest_profit']
            print(f"  Highest Profit: {alt['amount_strategy']}/{alt['pricing_strategy']}/{alt['payment_pattern']}/{alt['term_months']}m")
            print(f"    Profit: ${alt['profit_analysis']['expected_profit']:,.0f}")
        
        if alternatives['highest_return'] != best:
            alt = alternatives['highest_return']
            print(f"  Highest Return: {alt['amount_strategy']}/{alt['pricing_strategy']}/{alt['payment_pattern']}/{alt['term_months']}m")
            print(f"    Annual Return: {alt['profit_analysis']['annualized_return']:.2%}")
        
        if alternatives['highest_margin'] != best:
            alt = alternatives['highest_margin']
            print(f"  Highest Margin: {alt['amount_strategy']}/{alt['pricing_strategy']}/{alt['payment_pattern']}/{alt['term_months']}m")
            print(f"    Interest Margin: {alt['loan_metrics']['interest_margin']:.2%}")
    
    print("\n3. APPROVAL RATE SENSITIVITY ANALYSIS")
    print("-" * 60)
    
    # Sensitivity analysis
    sensitivity_sample = sample_applications.head(20)  # Smaller sample for faster processing
    sensitivity_amounts = np.random.uniform(25000, 45000, len(sensitivity_sample))
    
    print("Analyzing approval rate sensitivity across all 4 variables...")
    sensitivity_results = joint_engine.analyze_approval_rate_sensitivity(
        sensitivity_sample, sensitivity_amounts.tolist(), sensitivity_variable='all'
    )
    
    print(f"\nSENSITIVITY ANALYSIS RESULTS:")
    
    for variable, results in sensitivity_results.items():
        print(f"\n{variable.replace('_', ' ').title()} Sensitivity:")
        
        for strategy, metrics in results.items():
            rate = metrics['approval_rate']
            total = metrics['total']
            
            if variable == 'amount_strategy':
                multiplier = metrics['multiplier']
                print(f"  {strategy}: {rate:.1%} approval rate ({multiplier:.0%} of requested)")
            elif variable == 'pricing_strategy':
                adjustment = metrics['rate_adjustment']
                print(f"  {strategy}: {rate:.1%} approval rate ({adjustment:+.1%} rate)")
            else:
                print(f"  {strategy}: {rate:.1%} approval rate")
    
    # Find most sensitive variable
    sensitivity_ranges = {}
    for variable, results in sensitivity_results.items():
        rates = [r['approval_rate'] for r in results.values()]
        sensitivity_ranges[variable] = max(rates) - min(rates)
    
    most_sensitive = max(sensitivity_ranges.keys(), key=lambda k: sensitivity_ranges[k])
    print(f"\nMost Sensitive Variable: {most_sensitive.replace('_', ' ').title()}")
    print(f"  Range: {sensitivity_ranges[most_sensitive]:.1%}")
    
    print("\n4. PORTFOLIO BATCH PROCESSING")
    print("-" * 60)
    
    # Portfolio optimization
    portfolio_sample = sample_applications.head(25)  # Manageable sample
    portfolio_amounts = np.random.uniform(20000, 50000, len(portfolio_sample))
    
    # Batch 4-variable optimization
    batch_results = joint_engine.batch_4variable_optimization(
        portfolio_sample, portfolio_amounts.tolist(), apply_portfolio_constraints=True
    )
    
    print(f"\n4-Variable Portfolio Optimization Results:")
    approved_count = (batch_results['decision'] == 'APPROVE').sum()
    total_count = len(batch_results)
    
    print(f"  Total Applications: {total_count}")
    print(f"  Approved Loans: {approved_count}")
    print(f"  Approval Rate: {approved_count/total_count:.1%}")
    
    if approved_count > 0:
        approved_loans = batch_results[batch_results['decision'] == 'APPROVE']
        
        print(f"\nPortfolio Summary:")
        print(f"  Total Portfolio Value: ${approved_loans['expected_profit'].sum():,.0f}")
        print(f"  Average Expected Profit: ${approved_loans['expected_profit'].mean():,.0f}")
        print(f"  Average Annualized Return: {approved_loans['annualized_return'].mean():.2%}")
        print(f"  Average Composite Score: {approved_loans['composite_score'].mean():.3f}")
        print(f"  Total Loan Volume: ${approved_loans['actual_amount'].sum():,.0f}")
        print(f"  Average Amount Multiplier: {approved_loans['amount_multiplier'].mean():.1%}")
        
        print(f"\nStrategy Distribution:")
        
        # Amount strategies
        amount_dist = approved_loans['amount_strategy'].value_counts()
        print(f"  Amount Strategies:")
        for strategy, count in amount_dist.items():
            percentage = count / len(approved_loans) * 100
            avg_multiplier = approved_loans[approved_loans['amount_strategy'] == strategy]['amount_multiplier'].mean()
            print(f"    {strategy}: {count} loans ({percentage:.1f}%) - Avg: {avg_multiplier:.0%}")
        
        # Pricing strategies
        pricing_dist = approved_loans['pricing_strategy'].value_counts()
        print(f"  Pricing Strategies:")
        for strategy, count in pricing_dist.items():
            percentage = count / len(approved_loans) * 100
            avg_margin = approved_loans[approved_loans['pricing_strategy'] == strategy]['interest_margin'].mean()
            print(f"    {strategy}: {count} loans ({percentage:.1f}%) - Avg Margin: {avg_margin:.2%}")
        
        # Pattern-term combinations
        combo_dist = approved_loans.groupby(['payment_pattern', 'term_months']).size()
        print(f"  Top Pattern-Term Combinations:")
        for (pattern, term), count in combo_dist.head(5).items():
            percentage = count / len(approved_loans) * 100
            print(f"    {pattern.title()} {term}m: {count} loans ({percentage:.1f}%)")
    
    print("\n5. COMPREHENSIVE BUSINESS INSIGHTS")
    print("-" * 60)
    
    # Generate insights
    insights = joint_engine.generate_4variable_insights(batch_results, sensitivity_results)
    
    if 'error' not in insights:
        print("Optimization Performance:")
        opt_summary = insights['optimization_summary']
        print(f"  Approval Rate: {opt_summary['approval_rate']:.1%}")
        print(f"  Avg Combinations per Application: {opt_summary['avg_combinations_evaluated']:.0f}")
        print(f"  Optimization Efficiency: {opt_summary['optimization_efficiency']:.1%}")
        
        print(f"\nPortfolio Quality:")
        portfolio_metrics = insights['portfolio_metrics']
        print(f"  Average Composite Score: {portfolio_metrics['average_composite_score']:.3f}")
        print(f"  Average Annualized Return: {portfolio_metrics['average_annualized_return']:.2%}")
        print(f"  Average Interest Margin: {portfolio_metrics['average_interest_margin']:.2%}")
        
        print(f"\nRisk Profile:")
        risk_profile = insights['risk_profile']
        print(f"  Average Default Probability: {risk_profile['average_default_probability']:.1%}")
        print(f"  Average Debt Service Coverage: {risk_profile['average_debt_service_coverage']:.2f}x")
        
        print(f"\nStrategic Recommendations:")
        for i, rec in enumerate(insights['strategic_recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\n6. Generating comprehensive visualizations...")
    
    # Generate visualizations
    if approved_count > 0:
        joint_engine.plot_4variable_sensitivity_analysis(sensitivity_results)
        joint_engine.plot_4variable_optimization_results(batch_results)
    
    print("\n" + "="*80)
    print("4-VARIABLE JOINT OPTIMIZATION ANALYSIS COMPLETE!")
    print("="*80)
    
    return joint_engine, batch_results, sensitivity_results, insights


if __name__ == "__main__":
    engine, results, sensitivity, insights = main() 