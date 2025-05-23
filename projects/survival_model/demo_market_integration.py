"""
Market Factors Integration Demonstration

This script demonstrates how to integrate the Market Factors Risk Model
with SME lending decisions for enhanced risk assessment.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from market_factors import MarketFactorsContainer

class SimplifiedLendingEngine:
    """
    Simplified lending engine for demonstration of market factors integration.
    """
    
    def __init__(self, market_factors_container=None):
        self.market_factors = market_factors_container or MarketFactorsContainer()
        
    def evaluate_loan_with_market_factors(self, borrower_data, loan_terms):
        """
        Evaluate loan considering both borrower-specific and market factors.
        """
        # Get base evaluation using simplified approach
        base_evaluation = self._simplified_loan_evaluation(borrower_data, loan_terms)
        
        if not base_evaluation['approved']:
            return base_evaluation
            
        # Apply market factor adjustments
        industry = borrower_data.get('industry', 'services')
        region = borrower_data.get('region', None)
        
        market_multiplier = self.market_factors.get_portfolio_risk_adjustment(
            industry=industry, 
            region=region
        )
        
        # Adjust default probability
        adjusted_default_prob = min(1.0, 
            base_evaluation['default_probability'] * market_multiplier
        )
        
        # Recalculate metrics with adjusted default probability
        adjusted_evaluation = self._recalculate_with_market_adjustment(
            base_evaluation, 
            adjusted_default_prob,
            market_multiplier
        )
        
        return adjusted_evaluation
    
    def _simplified_loan_evaluation(self, borrower_data, loan_terms):
        """Simplified loan evaluation for demonstration purposes."""
        # Basic risk assessment
        credit_score = borrower_data.get('credit_score', 700)
        debt_to_income = borrower_data.get('debt_to_income', 0.35)
        years_in_business = borrower_data.get('years_in_business', 5)
        
        # Simple default probability calculation
        base_default_prob = 0.05  # Start with 5% base
        
        # Adjust based on credit score
        if credit_score < 650:
            base_default_prob += 0.10
        elif credit_score < 700:
            base_default_prob += 0.05
        elif credit_score > 750:
            base_default_prob -= 0.02
            
        # Adjust based on debt-to-income
        if debt_to_income > 0.4:
            base_default_prob += 0.05
        elif debt_to_income < 0.3:
            base_default_prob -= 0.01
            
        # Adjust based on business experience
        if years_in_business < 3:
            base_default_prob += 0.03
        elif years_in_business > 7:
            base_default_prob -= 0.01
            
        # Cap at reasonable bounds
        base_default_prob = max(0.01, min(0.50, base_default_prob))
        
        # Calculate basic metrics
        loan_amount = borrower_data.get('loan_amount', 50000)
        expected_return = 0.08  # 8% base expected return
        expected_profit = loan_amount * 0.05  # 5% profit margin
        
        # Basic approval criteria
        approved = (
            base_default_prob <= 0.30 and
            credit_score >= 650 and
            debt_to_income <= 0.45 and
            expected_return >= 0.04
        )
        
        return {
            'approved': approved,
            'default_probability': base_default_prob,
            'expected_return': expected_return,
            'expected_profit': expected_profit,
            'loan_amount': loan_amount
        }
        
    def _recalculate_with_market_adjustment(self, base_eval, adj_default_prob, multiplier):
        """Recalculate loan metrics with market-adjusted default probability."""
        
        # Adjust expected return (simplified calculation)
        survival_prob = 1 - adj_default_prob
        adj_expected_return = base_eval.get('expected_return', 0.05) * survival_prob
        
        # Adjust profit calculations
        adj_expected_profit = base_eval.get('expected_profit', 1000) * survival_prob
        
        # Create adjusted evaluation
        adjusted_eval = base_eval.copy()
        adjusted_eval.update({
            'default_probability': adj_default_prob,
            'expected_return': adj_expected_return,
            'expected_profit': adj_expected_profit,
            'market_risk_multiplier': multiplier,
            'market_adjusted': True
        })
        
        # Re-check approval criteria with adjusted metrics
        adjusted_eval['approved'] = self._check_adjusted_approval_criteria(adjusted_eval)
        
        return adjusted_eval
        
    def _check_adjusted_approval_criteria(self, evaluation):
        """Check approval criteria with market-adjusted metrics."""
        criteria = {
            'min_expected_return': evaluation['expected_return'] >= 0.04,
            'max_default_probability': evaluation['default_probability'] <= 0.30,
            'min_expected_profit': evaluation['expected_profit'] >= 300,
            'market_risk_acceptable': evaluation['market_risk_multiplier'] <= 1.5
        }
        
        return all(criteria.values())

def demonstrate_market_integration():
    """Demonstrate market factors integration with lending decisions."""
    
    print("="*80)
    print("MARKET FACTORS INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Initialize market-aware lending engine
    print("\n1. Initializing Market-Aware Lending Engine...")
    lending_engine = SimplifiedLendingEngine()
    
    # Update current market conditions
    print("\n2. Setting Current Market Conditions...")
    market_conditions = {
        'unemployment_rate': 3.7,      # Low unemployment (good)
        'fed_funds_rate': 5.25,        # High rates (challenging for SMEs)
        'inflation_rate': 3.2,         # Moderate inflation
        'vix_volatility': 18.5,        # Low volatility (stable)
        'gdp_growth_rate': 2.1,        # Moderate growth
        'consumer_confidence': 102.0,   # High confidence (good)
        'business_confidence': 98.5,    # High confidence (good)
        'credit_spreads': 450,         # Moderate spreads
        'bank_lending_standards': 15.0  # Some tightening
    }
    
    for factor, value in market_conditions.items():
        lending_engine.market_factors.update_factor_manual(factor, value)
        print(f"  Updated {factor}: {value}")
    
    # Generate market risk report
    print("\n3. Current Market Risk Assessment...")
    risk_report = lending_engine.market_factors.generate_risk_report()
    print(risk_report)
    
    # Test loan applications across different industries
    print("\n4. Testing Loan Applications Across Industries...")
    
    # Sample borrower profiles
    borrowers = [
        {
            'name': 'Tech Startup',
            'loan_amount': 75000,
            'annual_income': 120000,
            'credit_score': 750,
            'debt_to_income': 0.25,
            'years_in_business': 3,
            'industry': 'technology',
            'collateral_value': 80000,
            'payment_history_score': 0.90
        },
        {
            'name': 'Retail Store',
            'loan_amount': 50000,
            'annual_income': 85000,
            'credit_score': 720,
            'debt_to_income': 0.35,
            'years_in_business': 7,
            'industry': 'retail',
            'collateral_value': 60000,
            'payment_history_score': 0.85
        },
        {
            'name': 'Construction Company',
            'loan_amount': 100000,
            'annual_income': 150000,
            'credit_score': 700,
            'debt_to_income': 0.40,
            'years_in_business': 10,
            'industry': 'construction',
            'collateral_value': 120000,
            'payment_history_score': 0.80
        },
        {
            'name': 'Manufacturing Firm',
            'loan_amount': 80000,
            'annual_income': 110000,
            'credit_score': 730,
            'debt_to_income': 0.30,
            'years_in_business': 5,
            'industry': 'manufacturing',
            'collateral_value': 90000,
            'payment_history_score': 0.88
        }
    ]
    
    # Standard loan terms for comparison
    loan_terms = {
        'amount_strategy': 'standard',
        'pricing_strategy': 'premium',
        'payment_pattern': 'installment',
        'term_months': 36
    }
    
    print(f"\nEvaluating {len(borrowers)} loan applications with market factor adjustments:")
    print("-" * 80)
    
    results = []
    for borrower in borrowers:
        # Evaluate with market factors
        evaluation = lending_engine.evaluate_loan_with_market_factors(
            borrower, 
            loan_terms
        )
        
        results.append({
            'borrower': borrower['name'],
            'industry': borrower['industry'],
            'approved': evaluation.get('approved', False),
            'default_probability': evaluation.get('default_probability', 0),
            'expected_return': evaluation.get('expected_return', 0),
            'expected_profit': evaluation.get('expected_profit', 0),
            'market_multiplier': evaluation.get('market_risk_multiplier', 1.0)
        })
        
        print(f"\n{borrower['name']} ({borrower['industry'].title()}):")
        print(f"  Approved: {'✓' if evaluation.get('approved', False) else '✗'}")
        print(f"  Default Probability: {evaluation.get('default_probability', 0):.3f}")
        print(f"  Expected Return: {evaluation.get('expected_return', 0):.3f}")
        print(f"  Expected Profit: ${evaluation.get('expected_profit', 0):,.0f}")
        print(f"  Market Risk Multiplier: {evaluation.get('market_risk_multiplier', 1.0):.3f}")
    
    # Summary analysis
    print("\n5. Portfolio Analysis Summary...")
    print("-" * 80)
    
    approved_loans = [r for r in results if r['approved']]
    total_applications = len(results)
    approval_rate = len(approved_loans) / total_applications if total_applications > 0 else 0
    
    print(f"Total Applications: {total_applications}")
    print(f"Approved Loans: {len(approved_loans)}")
    print(f"Approval Rate: {approval_rate:.1%}")
    
    if approved_loans:
        avg_default_prob = np.mean([r['default_probability'] for r in approved_loans])
        avg_expected_return = np.mean([r['expected_return'] for r in approved_loans])
        total_expected_profit = sum([r['expected_profit'] for r in approved_loans])
        avg_market_multiplier = np.mean([r['market_multiplier'] for r in approved_loans])
        
        print(f"\nApproved Loans Portfolio Metrics:")
        print(f"  Average Default Probability: {avg_default_prob:.3f}")
        print(f"  Average Expected Return: {avg_expected_return:.3f}")
        print(f"  Total Expected Profit: ${total_expected_profit:,.0f}")
        print(f"  Average Market Risk Multiplier: {avg_market_multiplier:.3f}")
    
    # Industry risk comparison
    print("\n6. Industry Risk Comparison...")
    print("-" * 80)
    
    industries = ['technology', 'retail', 'construction', 'manufacturing', 'services']
    industry_risks = {}
    
    for industry in industries:
        risk_analysis = lending_engine.market_factors.calculate_risk_score(industry=industry)
        multiplier = lending_engine.market_factors.get_portfolio_risk_adjustment(industry=industry)
        
        industry_risks[industry] = {
            'risk_score': risk_analysis['risk_score'],
            'multiplier': multiplier
        }
        
        if multiplier > 1.1:
            recommendation = "RESTRICT (High Risk)"
        elif multiplier > 1.05:
            recommendation = "CAUTION (Elevated Risk)"
        elif multiplier < 0.95:
            recommendation = "EXPAND (Low Risk)"
        else:
            recommendation = "MAINTAIN (Normal Risk)"
        
        print(f"{industry.title():15} | Risk Score: {risk_analysis['risk_score']:.3f} | "
              f"Multiplier: {multiplier:.3f} | {recommendation}")
    
    # Market scenario analysis
    print("\n7. Market Scenario Analysis...")
    print("-" * 80)
    
    scenarios = {
        'Economic Downturn': {
            'unemployment_rate': 6.5,
            'gdp_growth_rate': -1.0,
            'consumer_confidence': 85.0,
            'credit_spreads': 800
        },
        'Economic Boom': {
            'unemployment_rate': 3.0,
            'gdp_growth_rate': 4.5,
            'consumer_confidence': 120.0,
            'credit_spreads': 300
        },
        'High Inflation': {
            'inflation_rate': 6.0,
            'fed_funds_rate': 7.0,
            'business_confidence': 85.0
        }
    }
    
    print("Scenario Impact Analysis:")
    for scenario_name, changes in scenarios.items():
        # Temporarily update factors
        original_values = {}
        for factor, new_value in changes.items():
            original_values[factor] = lending_engine.market_factors.factors[factor].current_value
            lending_engine.market_factors.update_factor_manual(factor, new_value)
        
        # Calculate new risk score
        scenario_risk = lending_engine.market_factors.calculate_risk_score()
        scenario_multiplier = lending_engine.market_factors.get_portfolio_risk_adjustment()
        
        print(f"\n{scenario_name}:")
        print(f"  Risk Score: {scenario_risk['risk_score']:.3f}")
        print(f"  Risk Multiplier: {scenario_multiplier:.3f}")
        print(f"  Impact: {((scenario_multiplier - 1.0) * 100):+.1f}% adjustment to default probabilities")
        
        # Restore original values
        for factor, original_value in original_values.items():
            if original_value is not None:
                lending_engine.market_factors.update_factor_manual(factor, original_value)
    
    print("\n" + "="*80)
    print("MARKET FACTORS INTEGRATION DEMONSTRATION COMPLETE")
    print("="*80)
    
    return lending_engine, results

def main():
    """Run the market factors integration demonstration."""
    try:
        lending_engine, results = demonstrate_market_integration()
        
        print(f"\n✓ Successfully demonstrated market factors integration")
        print(f"✓ Evaluated {len(results)} loan applications with market adjustments")
        print(f"✓ Generated comprehensive risk analysis across industries")
        print(f"✓ Performed scenario analysis for different market conditions")
        
        return lending_engine
        
    except Exception as e:
        print(f"\n✗ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    lending_engine = main() 