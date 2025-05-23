"""
Payment Pattern Decision Engine

This module demonstrates how to use survival model predictions to make optimal 
payment pattern approval decisions for individual borrowers to maximize loan portfolio profit.

The decision engine evaluates each borrower against all available payment patterns
and recommends the one that maximizes expected profit while meeting risk criteria.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from credit_survival_model import CreditSurvivalModel
from loan_profit_optimizer import LoanProfitOptimizer

class PaymentPatternDecisionEngine:
    """
    Decision engine for optimal payment pattern approval based on survival predictions.
    """
    
    def __init__(self, survival_model: CreditSurvivalModel, profit_optimizer: LoanProfitOptimizer):
        """
        Initialize the decision engine.
        
        Parameters:
        -----------
        survival_model : CreditSurvivalModel
            Fitted survival model
        profit_optimizer : LoanProfitOptimizer
            Configured profit optimizer
        """
        self.survival_model = survival_model
        self.profit_optimizer = profit_optimizer
        
        # Decision criteria
        self.min_expected_return = 0.05  # Minimum 5% expected return
        self.max_default_probability = 0.40  # Maximum 40% default probability
        self.min_profit_per_loan = 500  # Minimum $500 profit per loan
        
    def evaluate_borrower(self, borrower_data: pd.DataFrame, loan_amount: float, 
                         term_months: int = 48) -> dict:
        """
        Evaluate a borrower and recommend optimal payment pattern.
        
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
        dict : Decision recommendation with detailed analysis
        """
        # Run optimization
        optimization_result = self.profit_optimizer.optimize_payment_pattern(
            borrower_data, loan_amount, term_months
        )
        
        # Apply business rules and constraints
        approved_patterns = []
        rejected_patterns = []
        
        for pattern, analysis in optimization_result['borrower_analysis'].items():
            if not analysis.get('eligible', False):
                rejected_patterns.append({
                    'pattern': pattern,
                    'reason': analysis.get('reason', 'Not eligible')
                })
                continue
            
            # Apply business constraints
            rejection_reasons = []
            
            if analysis['expected_return'] < self.min_expected_return:
                rejection_reasons.append(f"Return {analysis['expected_return']:.1%} below minimum {self.min_expected_return:.1%}")
            
            if analysis['total_default_prob'] > self.max_default_probability:
                rejection_reasons.append(f"Default risk {analysis['total_default_prob']:.1%} exceeds maximum {self.max_default_probability:.1%}")
            
            if analysis['expected_profit'] < self.min_profit_per_loan:
                rejection_reasons.append(f"Profit ${analysis['expected_profit']:,.0f} below minimum ${self.min_profit_per_loan:,.0f}")
            
            if rejection_reasons:
                rejected_patterns.append({
                    'pattern': pattern,
                    'reason': '; '.join(rejection_reasons),
                    'analysis': analysis
                })
            else:
                approved_patterns.append({
                    'pattern': pattern,
                    'analysis': analysis
                })
        
        # Determine final recommendation
        if approved_patterns:
            # Choose pattern with highest expected profit
            best_pattern = max(approved_patterns, key=lambda x: x['analysis']['expected_profit'])
            
            # Alternative: highest risk-adjusted return
            best_risk_adjusted = max(approved_patterns, key=lambda x: x['analysis']['risk_adjusted_return'])
            
            decision = 'APPROVE'
            recommended_pattern = best_pattern['pattern']
        else:
            decision = 'DECLINE'
            recommended_pattern = None
            best_pattern = None
            best_risk_adjusted = None
        
        return {
            'decision': decision,
            'recommended_pattern': recommended_pattern,
            'best_profit_pattern': best_pattern,
            'best_risk_adjusted_pattern': best_risk_adjusted,
            'approved_patterns': approved_patterns,
            'rejected_patterns': rejected_patterns,
            'loan_amount': loan_amount,
            'term_months': term_months,
            'decision_criteria': {
                'min_expected_return': self.min_expected_return,
                'max_default_probability': self.max_default_probability,
                'min_profit_per_loan': self.min_profit_per_loan
            }
        }
    
    def generate_decision_report(self, decision_result: dict) -> str:
        """
        Generate a detailed decision report.
        
        Parameters:
        -----------
        decision_result : dict
            Result from evaluate_borrower
            
        Returns:
        --------
        str : Formatted decision report
        """
        report = []
        report.append("="*60)
        report.append("LOAN APPROVAL DECISION REPORT")
        report.append("="*60)
        
        report.append(f"\nLoan Details:")
        report.append(f"  Amount: ${decision_result['loan_amount']:,}")
        report.append(f"  Term: {decision_result['term_months']} months")
        
        report.append(f"\nDECISION: {decision_result['decision']}")
        
        if decision_result['decision'] == 'APPROVE':
            recommended = decision_result['best_profit_pattern']
            analysis = recommended['analysis']
            
            report.append(f"RECOMMENDED PAYMENT PATTERN: {recommended['pattern'].upper()}")
            report.append(f"\nExpected Performance:")
            report.append(f"  Expected Profit: ${analysis['expected_profit']:,.0f}")
            report.append(f"  Expected Return: {analysis['expected_return']:.2%}")
            report.append(f"  Default Probability: {analysis['total_default_prob']:.1%}")
            report.append(f"  Interest Rate: {analysis['interest_rate']:.2%}")
            report.append(f"  Monthly Payment: ${analysis['monthly_payment']:,.2f}")
            
            if analysis.get('balloon_payment', 0) > 0:
                report.append(f"  Balloon Payment: ${analysis['balloon_payment']:,.0f}")
            
            # Show alternatives if available
            if len(decision_result['approved_patterns']) > 1:
                report.append(f"\nAlternative Approved Patterns:")
                for pattern_info in decision_result['approved_patterns']:
                    if pattern_info['pattern'] != recommended['pattern']:
                        alt_analysis = pattern_info['analysis']
                        report.append(f"  {pattern_info['pattern'].title()}:")
                        report.append(f"    Profit: ${alt_analysis['expected_profit']:,.0f}")
                        report.append(f"    Return: {alt_analysis['expected_return']:.2%}")
                        report.append(f"    Default Risk: {alt_analysis['total_default_prob']:.1%}")
        
        else:  # DECLINE
            report.append(f"\nREASON FOR DECLINE:")
            report.append(f"  No payment pattern meets minimum business criteria")
        
        # Show rejected patterns
        if decision_result['rejected_patterns']:
            report.append(f"\nRejected Payment Patterns:")
            for rejected in decision_result['rejected_patterns']:
                report.append(f"  {rejected['pattern'].title()}: {rejected['reason']}")
        
        # Decision criteria
        criteria = decision_result['decision_criteria']
        report.append(f"\nDecision Criteria Applied:")
        report.append(f"  Minimum Expected Return: {criteria['min_expected_return']:.1%}")
        report.append(f"  Maximum Default Probability: {criteria['max_default_probability']:.1%}")
        report.append(f"  Minimum Profit per Loan: ${criteria['min_profit_per_loan']:,}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def batch_evaluate_applications(self, applications: pd.DataFrame, 
                                  loan_amounts: list) -> pd.DataFrame:
        """
        Evaluate multiple loan applications.
        
        Parameters:
        -----------
        applications : pd.DataFrame
            Multiple borrower applications
        loan_amounts : list
            Corresponding loan amounts
            
        Returns:
        --------
        pd.DataFrame : Batch evaluation results
        """
        results = []
        
        print(f"Evaluating {len(applications)} loan applications...")
        
        for idx, (_, borrower) in enumerate(applications.iterrows()):
            # Convert to DataFrame
            borrower_df = borrower.to_frame().T
            borrower_df['id'] = f'applicant_{idx}'
            
            loan_amount = loan_amounts[idx] if idx < len(loan_amounts) else np.mean(loan_amounts)
            
            # Evaluate borrower
            decision_result = self.evaluate_borrower(borrower_df, loan_amount)
            
            # Extract key results
            result = {
                'applicant_id': idx,
                'loan_amount': loan_amount,
                'decision': decision_result['decision'],
                'recommended_pattern': decision_result['recommended_pattern'],
                'num_approved_patterns': len(decision_result['approved_patterns']),
                'num_rejected_patterns': len(decision_result['rejected_patterns'])
            }
            
            # Add financial metrics for approved pattern
            if decision_result['decision'] == 'APPROVE':
                best_analysis = decision_result['best_profit_pattern']['analysis']
                result.update({
                    'expected_profit': best_analysis['expected_profit'],
                    'expected_return': best_analysis['expected_return'],
                    'default_probability': best_analysis['total_default_prob'],
                    'interest_rate': best_analysis['interest_rate'],
                    'monthly_payment': best_analysis['monthly_payment']
                })
            else:
                result.update({
                    'expected_profit': np.nan,
                    'expected_return': np.nan,
                    'default_probability': np.nan,
                    'interest_rate': np.nan,
                    'monthly_payment': np.nan
                })
            
            results.append(result)
            
            if (idx + 1) % 25 == 0:
                print(f"Processed {idx + 1} applications...")
        
        return pd.DataFrame(results)
    
    def plot_decision_analysis(self, batch_results: pd.DataFrame):
        """
        Visualize batch decision analysis results.
        
        Parameters:
        -----------
        batch_results : pd.DataFrame
            Results from batch_evaluate_applications
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Approval rate
        ax1 = axes[0, 0]
        approval_counts = batch_results['decision'].value_counts()
        colors = ['green' if decision == 'APPROVE' else 'red' for decision in approval_counts.index]
        bars = ax1.bar(approval_counts.index, approval_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Loan Application Decisions')
        ax1.set_ylabel('Number of Applications')
        
        # Add percentage labels
        total = len(batch_results)
        for bar, count in zip(bars, approval_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count/total:.1%}', ha='center', va='bottom')
        
        # 2. Approved pattern distribution
        ax2 = axes[0, 1]
        approved_data = batch_results[batch_results['decision'] == 'APPROVE']
        if len(approved_data) > 0:
            pattern_counts = approved_data['recommended_pattern'].value_counts()
            colors = ['green', 'red', 'orange']
            ax2.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(pattern_counts)])
            ax2.set_title('Approved Payment Pattern Distribution')
        else:
            ax2.text(0.5, 0.5, 'No Approvals', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Approved Payment Pattern Distribution')
        
        # 3. Expected profit distribution
        ax3 = axes[1, 0]
        if len(approved_data) > 0:
            profits = approved_data['expected_profit'].dropna()
            if len(profits) > 0:
                ax3.hist(profits, bins=15, alpha=0.7, color='green')
                ax3.set_xlabel('Expected Profit ($)')
                ax3.set_ylabel('Number of Loans')
                ax3.set_title('Expected Profit Distribution (Approved Loans)')
                ax3.axvline(x=profits.mean(), color='red', linestyle='--', 
                           label=f'Mean: ${profits.mean():,.0f}')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Approved Loans', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Expected Profit Distribution')
        
        # 4. Risk vs Return scatter
        ax4 = axes[1, 1]
        if len(approved_data) > 0:
            valid_data = approved_data.dropna(subset=['expected_return', 'default_probability'])
            if len(valid_data) > 0:
                patterns = valid_data['recommended_pattern'].unique()
                colors = ['green', 'red', 'orange']
                
                for i, pattern in enumerate(patterns):
                    pattern_data = valid_data[valid_data['recommended_pattern'] == pattern]
                    ax4.scatter(pattern_data['default_probability'], 
                              pattern_data['expected_return'],
                              label=pattern.title(), alpha=0.7, 
                              color=colors[i % len(colors)])
                
                ax4.set_xlabel('Default Probability')
                ax4.set_ylabel('Expected Return')
                ax4.set_title('Risk vs Return (Approved Loans)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Approved Loans', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Risk vs Return Analysis')
        
        plt.tight_layout()
        plt.show()


def main():
    """Demonstrate the payment pattern decision engine."""
    print("="*80)
    print("PAYMENT PATTERN DECISION ENGINE DEMONSTRATION")
    print("="*80)
    
    # Initialize components
    print("\n1. Setting up survival model and profit optimizer...")
    
    # Load existing model
    survival_model = CreditSurvivalModel(random_state=42)
    train_data, test_data, val_data = survival_model.generate_sample_data(
        n_subjects=300, max_time=48, test_size=0.3, val_size=0.1
    )
    survival_model.fit_cox_model(penalizer=0.01)
    
    # Initialize profit optimizer
    profit_optimizer = LoanProfitOptimizer(
        survival_model=survival_model,
        base_interest_rate=0.12,
        cost_of_funds=0.03,
        recovery_rate=0.40
    )
    
    # Initialize decision engine
    decision_engine = PaymentPatternDecisionEngine(survival_model, profit_optimizer)
    
    print("\n2. INDIVIDUAL LOAN APPLICATION EXAMPLE")
    print("-" * 50)
    
    # Example individual borrower
    sample_applications = test_data.groupby('id').first().reset_index()
    sample_borrower = sample_applications.iloc[0:1].copy()
    sample_borrower['id'] = 'individual_applicant'
    
    # Evaluate individual application
    decision_result = decision_engine.evaluate_borrower(
        sample_borrower, loan_amount=30000, term_months=48
    )
    
    # Generate and print report
    report = decision_engine.generate_decision_report(decision_result)
    print(report)
    
    print("\n3. BATCH APPLICATION PROCESSING")
    print("-" * 50)
    
    # Process multiple applications
    batch_applications = sample_applications.head(50)  # Process 50 applications
    loan_amounts = np.random.uniform(15000, 45000, len(batch_applications))
    
    # Batch evaluation
    batch_results = decision_engine.batch_evaluate_applications(
        batch_applications, loan_amounts.tolist()
    )
    
    # Print batch summary
    print(f"\nBatch Processing Summary:")
    print(f"Total Applications: {len(batch_results)}")
    
    approval_rate = (batch_results['decision'] == 'APPROVE').mean()
    print(f"Approval Rate: {approval_rate:.1%}")
    
    if approval_rate > 0:
        approved_loans = batch_results[batch_results['decision'] == 'APPROVE']
        
        print(f"\nApproved Loans Analysis:")
        print(f"  Number Approved: {len(approved_loans)}")
        print(f"  Average Loan Amount: ${approved_loans['loan_amount'].mean():,.0f}")
        print(f"  Average Expected Profit: ${approved_loans['expected_profit'].mean():,.0f}")
        print(f"  Average Expected Return: {approved_loans['expected_return'].mean():.2%}")
        print(f"  Average Default Risk: {approved_loans['default_probability'].mean():.1%}")
        
        print(f"\nPayment Pattern Distribution:")
        pattern_dist = approved_loans['recommended_pattern'].value_counts()
        for pattern, count in pattern_dist.items():
            percentage = count / len(approved_loans) * 100
            avg_profit = approved_loans[approved_loans['recommended_pattern'] == pattern]['expected_profit'].mean()
            print(f"  {pattern.title()}: {count} loans ({percentage:.1f}%) - Avg Profit: ${avg_profit:,.0f}")
        
        # Calculate total portfolio value
        total_portfolio_value = approved_loans['expected_profit'].sum()
        print(f"\nTotal Portfolio Value: ${total_portfolio_value:,.0f}")
    
    print(f"\nDeclined Loans:")
    declined_loans = batch_results[batch_results['decision'] == 'DECLINE']
    print(f"  Number Declined: {len(declined_loans)}")
    if len(declined_loans) > 0:
        print(f"  Average Loan Amount: ${declined_loans['loan_amount'].mean():,.0f}")
    
    # Generate visualizations
    print("\n4. Generating decision analysis visualizations...")
    decision_engine.plot_decision_analysis(batch_results)
    
    print("\n5. BUSINESS INSIGHTS FROM DECISION ENGINE")
    print("-" * 50)
    
    if approval_rate > 0:
        # Risk-based insights
        approved_loans = batch_results[batch_results['decision'] == 'APPROVE']
        
        # Find most profitable pattern
        pattern_profits = approved_loans.groupby('recommended_pattern')['expected_profit'].agg(['mean', 'sum', 'count'])
        best_pattern = pattern_profits['sum'].idxmax()
        
        print(f"Key Insights:")
        print(f"1. Most Profitable Pattern: {best_pattern.title()}")
        print(f"   - Total Portfolio Value: ${pattern_profits.loc[best_pattern, 'sum']:,.0f}")
        print(f"   - Average Profit per Loan: ${pattern_profits.loc[best_pattern, 'mean']:,.0f}")
        print(f"   - Number of Loans: {pattern_profits.loc[best_pattern, 'count']}")
        
        # Risk analysis
        avg_default_risk = approved_loans['default_probability'].mean()
        max_default_risk = approved_loans['default_probability'].max()
        
        print(f"\n2. Portfolio Risk Profile:")
        print(f"   - Average Default Risk: {avg_default_risk:.1%}")
        print(f"   - Maximum Default Risk: {max_default_risk:.1%}")
        print(f"   - Risk Threshold: {decision_engine.max_default_probability:.1%}")
        
        # ROI analysis
        avg_return = approved_loans['expected_return'].mean()
        print(f"\n3. Return Analysis:")
        print(f"   - Average Expected Return: {avg_return:.2%}")
        print(f"   - Minimum Required Return: {decision_engine.min_expected_return:.1%}")
        
        # Decision criteria effectiveness
        print(f"\n4. Decision Criteria Effectiveness:")
        print(f"   - Applications meeting all criteria: {approval_rate:.1%}")
        print(f"   - Average profit above minimum: ${(approved_loans['expected_profit'] - decision_engine.min_profit_per_loan).mean():,.0f}")
    
    print(f"\n6. ACTIONABLE RECOMMENDATIONS")
    print("-" * 50)
    
    recommendations = []
    
    if approval_rate < 0.3:
        recommendations.append("Consider relaxing decision criteria - low approval rate may indicate overly conservative thresholds")
    elif approval_rate > 0.8:
        recommendations.append("Consider tightening decision criteria - high approval rate may indicate insufficient risk management")
    
    if approval_rate > 0:
        # Pattern-specific recommendations
        approved_loans = batch_results[batch_results['decision'] == 'APPROVE']
        balloon_share = (approved_loans['recommended_pattern'] == 'balloon').mean()
        
        if balloon_share > 0.5:
            recommendations.append("HIGH CONCENTRATION: Balloon payments exceed 50% - implement portfolio diversification limits")
        
        if balloon_share > 0.3:
            recommendations.append("Develop specialized balloon payment monitoring and refinancing programs")
        
        # Profitability recommendations
        low_profit_loans = (approved_loans['expected_profit'] < decision_engine.min_profit_per_loan * 1.5).sum()
        if low_profit_loans > len(approved_loans) * 0.3:
            recommendations.append("Consider increasing minimum profit thresholds - many loans barely meet criteria")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("Current decision criteria appear well-calibrated for this portfolio.")
    
    print("\n" + "="*80)
    print("PAYMENT PATTERN DECISION ENGINE COMPLETE!")
    print("="*80)
    
    return decision_engine, batch_results


if __name__ == "__main__":
    engine, results = main() 