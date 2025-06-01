"""
Portfolio Simulator for PORTICO Model
Simulates a portfolio of credit card accounts and applies the PORTICO model
for dynamic credit line and pricing decisions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

from portico_model import PorticoModel, State, Action, ActionType


@dataclass
class Customer:
    """Represents a credit card customer"""
    customer_id: str
    current_state: State
    account_history: List[Dict] = field(default_factory=list)
    balance: float = 0.0
    last_payment: float = 0.0
    months_on_book: int = 0
    risk_score: float = 0.5
    
    def update_behavior(self):
        """Update customer behavior variables based on stochastic process"""
        # Simulate behavior evolution with some persistence
        for i in range(len(self.current_state.behavior_vars)):
            # Add random walk with mean reversion
            current_val = self.current_state.behavior_vars[i]
            change = np.random.normal(0, 0.3)  # Random change
            mean_reversion = 0.1 * (2.5 - current_val)  # Pull toward mean of 2.5
            
            new_val = current_val + change + mean_reversion
            new_val = max(1, min(4, new_val))  # Bound between 1 and 4
            self.current_state.behavior_vars[i] = new_val
    
    def calculate_utilization(self) -> float:
        """Calculate credit utilization rate"""
        if self.current_state.credit_line == 0:
            return 0.0
        return min(0.9, self.balance / (self.current_state.credit_line * 1000))
    
    def make_purchase(self, month: int):
        """Simulate customer purchases"""
        utilization_tendency = np.mean(self.current_state.behavior_vars) / 4
        base_purchase = self.current_state.credit_line * 100 * utilization_tendency
        
        # Add randomness
        purchase_amount = np.random.gamma(2, base_purchase / 2)
        credit_limit = self.current_state.credit_line * 1000
        
        # Don't exceed credit limit
        available_credit = credit_limit - self.balance
        purchase_amount = min(purchase_amount, available_credit * 0.8)
        
        if purchase_amount > 0:
            self.balance += purchase_amount
            
        return purchase_amount
    
    def make_payment(self, month: int):
        """Simulate customer payments"""
        if self.balance <= 0:
            return 0
        
        # Payment behavior based on behavior variables
        payment_tendency = (self.current_state.behavior_vars[1] + 
                          self.current_state.behavior_vars[2]) / 8
        
        # Minimum payment (2% of balance)
        min_payment = self.balance * 0.02
        
        # Full payment probability increases with good behavior
        if np.random.random() < payment_tendency:
            payment = self.balance  # Pay in full
        else:
            # Partial payment
            payment_ratio = np.random.uniform(0.02, 0.3)
            payment = self.balance * payment_ratio
        
        payment = min(payment, self.balance)
        self.balance -= payment
        self.last_payment = payment
        
        return payment


class PortfolioSimulator:
    """
    Simulates a portfolio of credit card accounts using the PORTICO model
    for dynamic credit line and pricing decisions.
    """
    
    def __init__(self, 
                 portico_model: PorticoModel,
                 n_customers: int = 1000,
                 simulation_months: int = 36):
        """
        Initialize the portfolio simulator
        
        Args:
            portico_model: Trained PORTICO model
            n_customers: Number of customers in portfolio
            simulation_months: Number of months to simulate
        """
        self.portico_model = portico_model
        self.n_customers = n_customers
        self.simulation_months = simulation_months
        
        self.customers: List[Customer] = []
        self.monthly_results: List[Dict] = []
        self.portfolio_metrics: Dict = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def generate_initial_portfolio(self):
        """Generate initial portfolio of customers"""
        self.customers = []
        
        for i in range(self.n_customers):
            # Generate initial state
            credit_line = np.random.randint(1, 11)  # Credit line levels 1-10
            apr = np.random.uniform(1.0, 5.0)
            behavior_vars = np.random.uniform(1, 4, 6)  # 6 behavior variables
            
            initial_state = State(
                credit_line=credit_line,
                apr=round(apr, 1),
                behavior_vars=behavior_vars
            )
            
            customer = Customer(
                customer_id=f"CUST_{i:06d}",
                current_state=initial_state,
                balance=np.random.uniform(0, credit_line * 500),  # Initial balance
                risk_score=np.mean(behavior_vars) / 4
            )
            
            self.customers.append(customer)
        
        self.logger.info(f"Generated portfolio of {len(self.customers)} customers")
    
    def simulate_month(self, month: int) -> Dict:
        """
        Simulate one month of portfolio activity
        
        Args:
            month: Current month number
            
        Returns:
            Dictionary with monthly results
        """
        monthly_data = {
            'month': month,
            'total_revenue': 0.0,
            'total_charge_offs': 0.0,
            'total_balances': 0.0,
            'avg_utilization': 0.0,
            'actions_taken': {'do_nothing': 0, 'increase_line': 0, 'decrease_line': 0,
                            'increase_apr': 0, 'decrease_apr': 0},
            'total_credit_extended': 0.0,
            'customer_count': len(self.customers)
        }
        
        for customer in self.customers:
            # Update customer behavior
            customer.update_behavior()
            customer.months_on_book += 1
            
            # Customer activities
            purchase_amount = customer.make_purchase(month)
            payment_amount = customer.make_payment(month)
            
            # Calculate monthly revenue
            if customer.balance > 0:
                monthly_interest = customer.balance * (customer.current_state.apr / 100) / 12
                monthly_data['total_revenue'] += monthly_interest
            
            # Apply PORTICO model decisions (every 6 months)
            if month % self.portico_model.update_frequency == 0:
                optimal_action = self.portico_model.get_optimal_action(customer.current_state)
                
                # Check business rules
                if self.portico_model.apply_business_rules(customer.current_state, optimal_action):
                    # Apply action
                    new_state = self.portico_model.apply_action(customer.current_state, optimal_action)
                    customer.current_state = new_state
                    
                    # Track actions
                    action_name = optimal_action.action_type.name.lower()
                    if 'line' in action_name:
                        if 'increase' in action_name:
                            monthly_data['actions_taken']['increase_line'] += 1
                        else:
                            monthly_data['actions_taken']['decrease_line'] += 1
                    elif 'apr' in action_name:
                        if 'increase' in action_name:
                            monthly_data['actions_taken']['increase_apr'] += 1
                        else:
                            monthly_data['actions_taken']['decrease_apr'] += 1
                    else:
                        monthly_data['actions_taken']['do_nothing'] += 1
                else:
                    monthly_data['actions_taken']['do_nothing'] += 1
            
            # Calculate charge-offs (simplified)
            if customer.risk_score > 0.8 and customer.balance > 0:
                if np.random.random() < 0.01:  # 1% monthly charge-off probability for high-risk
                    charge_off_amount = customer.balance * 0.5  # Partial charge-off
                    monthly_data['total_charge_offs'] += charge_off_amount
                    customer.balance *= 0.5  # Reduce balance
            
            # Update aggregates
            monthly_data['total_balances'] += customer.balance
            monthly_data['total_credit_extended'] += customer.current_state.credit_line * 1000
            
            # Store customer history
            customer.account_history.append({
                'month': month,
                'balance': customer.balance,
                'credit_line': customer.current_state.credit_line,
                'apr': customer.current_state.apr,
                'utilization': customer.calculate_utilization(),
                'purchase_amount': purchase_amount,
                'payment_amount': payment_amount,
                'risk_score': customer.risk_score
            })
        
        # Calculate portfolio averages
        if len(self.customers) > 0:
            total_credit = sum(c.current_state.credit_line * 1000 for c in self.customers)
            monthly_data['avg_utilization'] = monthly_data['total_balances'] / total_credit if total_credit > 0 else 0
        
        # Calculate net income
        monthly_data['net_income'] = monthly_data['total_revenue'] - monthly_data['total_charge_offs']
        
        return monthly_data
    
    def run_simulation(self):
        """Run the full portfolio simulation"""
        if not self.customers:
            self.generate_initial_portfolio()
        
        self.monthly_results = []
        
        self.logger.info(f"Starting {self.simulation_months}-month simulation...")
        
        for month in range(1, self.simulation_months + 1):
            monthly_result = self.simulate_month(month)
            self.monthly_results.append(monthly_result)
            
            if month % 6 == 0:
                self.logger.info(f"Completed month {month}")
        
        self._calculate_portfolio_metrics()
        self.logger.info("Simulation completed")
    
    def _calculate_portfolio_metrics(self):
        """Calculate overall portfolio performance metrics"""
        if not self.monthly_results:
            return
        
        df = pd.DataFrame(self.monthly_results)
        
        self.portfolio_metrics = {
            'total_revenue': df['total_revenue'].sum(),
            'total_charge_offs': df['total_charge_offs'].sum(),
            'net_income': df['net_income'].sum(),
            'avg_monthly_revenue': df['total_revenue'].mean(),
            'avg_utilization': df['avg_utilization'].mean(),
            'final_portfolio_size': df['total_credit_extended'].iloc[-1],
            'revenue_volatility': df['total_revenue'].std(),
            'charge_off_rate': df['total_charge_offs'].sum() / df['total_balances'].sum(),
            'return_on_assets': df['net_income'].sum() / df['total_balances'].mean(),
        }
        
        # Action distribution
        total_actions = sum([
            sum(month['actions_taken'].values()) for month in self.monthly_results
        ])
        
        if total_actions > 0:
            action_dist = {}
            for action_type in ['do_nothing', 'increase_line', 'decrease_line', 'increase_apr', 'decrease_apr']:
                action_count = sum([month['actions_taken'][action_type] for month in self.monthly_results])
                action_dist[action_type] = action_count / total_actions
            
            self.portfolio_metrics['action_distribution'] = action_dist
    
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate comprehensive performance report"""
        if not self.monthly_results:
            self.run_simulation()
        
        # Monthly performance data
        monthly_df = pd.DataFrame(self.monthly_results)
        
        # Customer-level summary
        customer_data = []
        for customer in self.customers:
            if customer.account_history:
                history_df = pd.DataFrame(customer.account_history)
                customer_summary = {
                    'customer_id': customer.customer_id,
                    'final_balance': customer.balance,
                    'final_credit_line': customer.current_state.credit_line,
                    'final_apr': customer.current_state.apr,
                    'avg_utilization': history_df['utilization'].mean(),
                    'total_purchases': history_df['purchase_amount'].sum(),
                    'total_payments': history_df['payment_amount'].sum(),
                    'months_on_book': customer.months_on_book,
                    'risk_score': customer.risk_score
                }
                customer_data.append(customer_summary)
        
        customer_df = pd.DataFrame(customer_data)
        
        return monthly_df, customer_df
    
    def plot_portfolio_performance(self):
        """Create visualizations of portfolio performance"""
        if not self.monthly_results:
            self.run_simulation()
        
        df = pd.DataFrame(self.monthly_results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PORTICO Model Portfolio Performance', fontsize=16)
        
        # Revenue over time
        axes[0, 0].plot(df['month'], df['total_revenue'])
        axes[0, 0].set_title('Monthly Revenue')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Revenue ($)')
        
        # Charge-offs over time
        axes[0, 1].plot(df['month'], df['total_charge_offs'], color='red')
        axes[0, 1].set_title('Monthly Charge-offs')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Charge-offs ($)')
        
        # Net income
        axes[0, 2].plot(df['month'], df['net_income'], color='green')
        axes[0, 2].set_title('Monthly Net Income')
        axes[0, 2].set_xlabel('Month')
        axes[0, 2].set_ylabel('Net Income ($)')
        
        # Portfolio balances
        axes[1, 0].plot(df['month'], df['total_balances'])
        axes[1, 0].set_title('Total Portfolio Balances')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Balances ($)')
        
        # Utilization rate
        axes[1, 1].plot(df['month'], df['avg_utilization'])
        axes[1, 1].set_title('Average Utilization Rate')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Utilization Rate')
        
        # Action distribution (pie chart for final month)
        if self.portfolio_metrics and 'action_distribution' in self.portfolio_metrics:
            action_dist = self.portfolio_metrics['action_distribution']
            axes[1, 2].pie(action_dist.values(), labels=action_dist.keys(), autopct='%1.1f%%')
            axes[1, 2].set_title('Action Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def compare_with_baseline(self, baseline_simulation_months: int = None):
        """
        Compare PORTICO performance with a no-action baseline
        
        Args:
            baseline_simulation_months: Months to simulate for baseline (default: same as main simulation)
        """
        if baseline_simulation_months is None:
            baseline_simulation_months = self.simulation_months
        
        # Save current results
        portico_results = self.monthly_results.copy()
        portico_metrics = self.portfolio_metrics.copy()
        
        # Create baseline simulator (no actions taken)
        baseline_customers = []
        for customer in self.customers:
            baseline_customer = Customer(
                customer_id=customer.customer_id,
                current_state=State(
                    credit_line=customer.current_state.credit_line,
                    apr=customer.current_state.apr,
                    behavior_vars=customer.current_state.behavior_vars.copy()
                ),
                balance=customer.balance
            )
            baseline_customers.append(baseline_customer)
        
        # Run baseline simulation (no PORTICO actions)
        baseline_results = []
        for month in range(1, baseline_simulation_months + 1):
            monthly_data = {
                'month': month,
                'total_revenue': 0.0,
                'total_charge_offs': 0.0,
                'total_balances': 0.0,
                'net_income': 0.0
            }
            
            for customer in baseline_customers:
                customer.update_behavior()
                customer.make_purchase(month)
                customer.make_payment(month)
                
                if customer.balance > 0:
                    monthly_interest = customer.balance * (customer.current_state.apr / 100) / 12
                    monthly_data['total_revenue'] += monthly_interest
                
                # Charge-offs
                if customer.risk_score > 0.8 and customer.balance > 0:
                    if np.random.random() < 0.01:
                        charge_off_amount = customer.balance * 0.5
                        monthly_data['total_charge_offs'] += charge_off_amount
                        customer.balance *= 0.5
                
                monthly_data['total_balances'] += customer.balance
            
            monthly_data['net_income'] = monthly_data['total_revenue'] - monthly_data['total_charge_offs']
            baseline_results.append(monthly_data)
        
        # Calculate baseline metrics
        baseline_df = pd.DataFrame(baseline_results)
        baseline_metrics = {
            'total_revenue': baseline_df['total_revenue'].sum(),
            'total_charge_offs': baseline_df['total_charge_offs'].sum(),
            'net_income': baseline_df['net_income'].sum(),
        }
        
        # Compare results
        comparison = {
            'portico_net_income': portico_metrics['net_income'],
            'baseline_net_income': baseline_metrics['net_income'],
            'improvement': portico_metrics['net_income'] - baseline_metrics['net_income'],
            'improvement_pct': ((portico_metrics['net_income'] - baseline_metrics['net_income']) / 
                              abs(baseline_metrics['net_income']) * 100) if baseline_metrics['net_income'] != 0 else 0
        }
        
        self.logger.info(f"PORTICO vs Baseline Comparison:")
        self.logger.info(f"PORTICO Net Income: ${comparison['portico_net_income']:,.2f}")
        self.logger.info(f"Baseline Net Income: ${comparison['baseline_net_income']:,.2f}")
        self.logger.info(f"Improvement: ${comparison['improvement']:,.2f} ({comparison['improvement_pct']:.1f}%)")
        
        return comparison, baseline_results
    
    def save_results(self, filepath: str):
        """Save simulation results to file"""
        results_data = {
            'monthly_results': self.monthly_results,
            'portfolio_metrics': self.portfolio_metrics,
            'customer_data': [
                {
                    'customer_id': c.customer_id,
                    'final_state': {
                        'credit_line': c.current_state.credit_line,
                        'apr': c.current_state.apr,
                        'behavior_vars': c.current_state.behavior_vars.tolist()
                    },
                    'balance': c.balance,
                    'history': c.account_history
                } for c in self.customers
            ],
            'simulation_parameters': {
                'n_customers': self.n_customers,
                'simulation_months': self.simulation_months
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")


def main():
    """Example usage of the PORTICO model and portfolio simulator"""
    # Initialize and train PORTICO model
    portico = PorticoModel(
        time_horizon=36,
        discount_factor=0.95,
        update_frequency=6
    )
    
    print("Training PORTICO model...")
    policy_table = portico.solve_mdp()
    print(f"Policy table created with {len(policy_table)} entries")
    
    # Create and run portfolio simulation
    simulator = PortfolioSimulator(
        portico_model=portico,
        n_customers=500,
        simulation_months=36
    )
    
    print("Running portfolio simulation...")
    simulator.run_simulation()
    
    # Generate reports
    monthly_df, customer_df = simulator.generate_performance_report()
    print("\nPortfolio Performance Metrics:")
    for metric, value in simulator.portfolio_metrics.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for k, v in value.items():
                print(f"  {k}: {v:.3f}")
        else:
            print(f"{metric}: {value:.3f}")
    
    # Compare with baseline
    comparison, baseline_results = simulator.compare_with_baseline()
    
    # Save results
    simulator.save_results('portico_simulation_results.json')
    portico.save_model('portico_model.pkl')
    
    # Generate plots
    simulator.plot_portfolio_performance()


if __name__ == "__main__":
    main() 