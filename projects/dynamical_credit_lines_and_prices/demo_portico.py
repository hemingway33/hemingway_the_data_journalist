"""
PORTICO Model Demonstration
Shows how the PORTICO model makes dynamic credit line and pricing decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from portico_model import PorticoModel, State, Action, ActionType
from portfolio_simulator import PortfolioSimulator, Customer

def demonstrate_single_customer_decisions():
    """Demonstrate PORTICO decisions for individual customer scenarios"""
    print("=== PORTICO Model: Single Customer Decision Examples ===\n")
    
    # Initialize and train model (simplified for demo)
    portico = PorticoModel(time_horizon=12, discount_factor=0.95, update_frequency=6)
    portico.generate_states()
    
    # Create sample policy decisions (simplified for demonstration)
    # In practice, this would be from the full MDP solution
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Low-Risk Customer',
            'state': State(credit_line=3, apr=2.5, behavior_vars=np.array([2, 3, 3, 2, 3, 2])),
            'description': 'Customer with good payment history and low utilization'
        },
        {
            'name': 'High-Risk Customer', 
            'state': State(credit_line=5, apr=4.0, behavior_vars=np.array([1, 1, 2, 1, 2, 1])),
            'description': 'Customer with poor payment history and high utilization'
        },
        {
            'name': 'Growing Customer',
            'state': State(credit_line=2, apr=3.0, behavior_vars=np.array([3, 4, 3, 4, 3, 3])),
            'description': 'Customer showing improved behavior patterns'
        },
        {
            'name': 'Declining Customer',
            'state': State(credit_line=8, apr=2.0, behavior_vars=np.array([2, 1, 1, 2, 1, 2])),
            'description': 'Customer with deteriorating behavior'
        }
    ]
    
    for scenario in scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Current State:")
        print(f"  - Credit Line: ${scenario['state'].credit_line * 1000:,}")
        print(f"  - APR: {scenario['state'].apr:.1f}%")
        print(f"  - Behavior Score: {np.mean(scenario['state'].behavior_vars):.2f}/4.0")
        
        # Calculate NCF for current state
        ncf = portico.calculate_ncf(scenario['state'])
        print(f"  - Monthly NCF: ${ncf:.2f}")
        
        # Get available actions
        actions = portico.get_available_actions(scenario['state'])
        print(f"Available Actions:")
        
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            new_state = portico.apply_action(scenario['state'], action)
            new_ncf = portico.calculate_ncf(new_state)
            
            # Simple immediate value calculation for demo
            value = new_ncf
            
            action_desc = f"{action.action_type.name}"
            if action.magnitude > 0:
                action_desc += f" (${action.magnitude:,.0f})" if 'LINE' in action.action_type.name else f" ({action.magnitude:.1f}%)"
            
            print(f"  - {action_desc}: Expected NCF = ${new_ncf:.2f}")
            
            if value > best_value:
                best_value = value
                best_action = action
        
        print(f"Recommended Action: {best_action.action_type.name}")
        if best_action.magnitude > 0:
            if 'LINE' in best_action.action_type.name:
                print(f"  - Amount: ${best_action.magnitude:,.0f}")
            else:
                print(f"  - Amount: {best_action.magnitude:.1f}%")
        
        print("-" * 60)


def demonstrate_portfolio_optimization():
    """Demonstrate portfolio-level optimization"""
    print("\n=== PORTICO Model: Portfolio Optimization Demo ===\n")
    
    # Create simplified model for faster demo
    portico = PorticoModel(
        time_horizon=12,
        discount_factor=0.95, 
        update_frequency=6
    )
    
    print("Training PORTICO model (simplified)...")
    # Generate smaller state space for demo
    portico.credit_line_max = 5
    portico.behavior_var_levels = [1, 2, 3]
    policy_table = portico.solve_mdp()
    print(f"Trained model with {len(policy_table)} policy entries")
    
    # Create portfolio simulator
    simulator = PortfolioSimulator(
        portico_model=portico,
        n_customers=100,  # Smaller portfolio for demo
        simulation_months=12
    )
    
    print("\nRunning 12-month portfolio simulation...")
    simulator.run_simulation()
    
    # Display results
    print("\nPortfolio Performance Summary:")
    metrics = simulator.portfolio_metrics
    print(f"Total Revenue: ${metrics['total_revenue']:,.2f}")
    print(f"Total Charge-offs: ${metrics['total_charge_offs']:,.2f}")
    print(f"Net Income: ${metrics['net_income']:,.2f}")
    print(f"Average Utilization: {metrics['avg_utilization']:.1%}")
    print(f"Return on Assets: {metrics['return_on_assets']:.1%}")
    
    if 'action_distribution' in metrics:
        print("\nAction Distribution:")
        for action, pct in metrics['action_distribution'].items():
            print(f"  {action}: {pct:.1%}")
    
    # Compare with baseline
    print("\nComparing with baseline (no optimization)...")
    comparison, _ = simulator.compare_with_baseline()
    print(f"Improvement over baseline: ${comparison['improvement']:,.2f} ({comparison['improvement_pct']:.1f}%)")


def demonstrate_state_value_analysis():
    """Analyze value function across different states"""
    print("\n=== PORTICO Model: State Value Analysis ===\n")
    
    portico = PorticoModel(time_horizon=12, discount_factor=0.95)
    portico.credit_line_max = 5
    portico.behavior_var_levels = [1, 2, 3]
    portico.solve_mdp()
    
    # Create sample states for analysis
    states_to_analyze = []
    for credit_line in range(1, 6):
        for apr in [1.5, 3.0, 4.5]:
            for behavior_level in [1, 2, 3]:
                behavior_vars = np.full(6, behavior_level)
                state = State(
                    credit_line=credit_line,
                    apr=apr,
                    behavior_vars=behavior_vars
                )
                states_to_analyze.append(state)
    
    # Calculate values for analysis
    analysis_data = []
    for state in states_to_analyze:
        ncf = portico.calculate_ncf(state)
        risk_score = np.mean(state.behavior_vars) / 4
        
        # Try to find optimal action
        optimal_action = Action(ActionType.DO_NOTHING)
        try:
            optimal_action = portico.get_optimal_action(state)
        except:
            pass
        
        analysis_data.append({
            'Credit_Line': state.credit_line,
            'APR': state.apr,
            'Behavior_Score': np.mean(state.behavior_vars),
            'Risk_Score': risk_score,
            'Monthly_NCF': ncf,
            'Optimal_Action': optimal_action.action_type.name
        })
    
    df = pd.DataFrame(analysis_data)
    
    print("State Value Analysis Summary:")
    print(f"Average NCF by Credit Line Level:")
    credit_line_analysis = df.groupby('Credit_Line')['Monthly_NCF'].agg(['mean', 'std', 'count'])
    print(credit_line_analysis)
    
    print(f"\nAverage NCF by Behavior Score:")
    behavior_analysis = df.groupby('Behavior_Score')['Monthly_NCF'].agg(['mean', 'std', 'count'])
    print(behavior_analysis)
    
    print(f"\nAction Distribution by Risk Level:")
    df['Risk_Category'] = pd.cut(df['Risk_Score'], bins=[0, 0.4, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
    action_by_risk = df.groupby(['Risk_Category', 'Optimal_Action']).size().unstack(fill_value=0)
    action_by_risk_pct = action_by_risk.div(action_by_risk.sum(axis=1), axis=0) * 100
    print(action_by_risk_pct)


def create_decision_visualization():
    """Create visualizations of PORTICO decision patterns"""
    print("\n=== PORTICO Model: Decision Visualization ===\n")
    
    # Generate decision data
    portico = PorticoModel(time_horizon=12, discount_factor=0.95)
    portico.credit_line_max = 5
    portico.behavior_var_levels = [1, 2, 3]
    portico.solve_mdp()
    
    # Create decision matrix
    decision_data = []
    for credit_line in range(1, 6):
        for behavior_score in [1, 2, 3]:
            for apr in [1.5, 3.0, 4.5]:
                state = State(
                    credit_line=credit_line,
                    apr=apr,
                    behavior_vars=np.full(6, behavior_score)
                )
                
                try:
                    optimal_action = portico.get_optimal_action(state)
                    action_code = optimal_action.action_type.value
                except:
                    action_code = 0
                
                decision_data.append({
                    'Credit_Line': credit_line,
                    'Behavior_Score': behavior_score,
                    'APR': apr,
                    'Action': action_code,
                    'NCF': portico.calculate_ncf(state)
                })
    
    df = pd.DataFrame(decision_data)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PORTICO Model Decision Analysis', fontsize=16)
    
    # 1. NCF by Credit Line and Behavior Score
    pivot_ncf = df.pivot_table(values='NCF', index='Credit_Line', columns='Behavior_Score', aggfunc='mean')
    sns.heatmap(pivot_ncf, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0,0])
    axes[0,0].set_title('Average NCF by Credit Line and Behavior Score')
    
    # 2. Action Distribution by Behavior Score
    action_counts = df.groupby(['Behavior_Score', 'Action']).size().unstack(fill_value=0)
    action_counts.plot(kind='bar', stacked=True, ax=axes[0,1])
    axes[0,1].set_title('Action Distribution by Behavior Score')
    axes[0,1].set_xlabel('Behavior Score')
    axes[0,1].legend(['Do Nothing', 'Increase Line', 'Decrease Line', 'Increase APR', 'Decrease APR'])
    
    # 3. NCF Distribution
    axes[1,0].hist(df['NCF'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Distribution of Monthly NCF')
    axes[1,0].set_xlabel('Monthly NCF ($)')
    axes[1,0].set_ylabel('Frequency')
    
    # 4. Credit Line vs APR decision pattern
    pivot_action = df.pivot_table(values='Action', index='Credit_Line', columns='APR', aggfunc='mean')
    sns.heatmap(pivot_action, annot=True, fmt='.1f', cmap='viridis', ax=axes[1,1])
    axes[1,1].set_title('Average Action Code by Credit Line and APR')
    
    plt.tight_layout()
    plt.savefig('projects/dynamical_credit_lines_and_prices/portico_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Decision visualization saved as 'portico_analysis.png'")


def demonstrate_business_rules():
    """Demonstrate how business rules constrain decisions"""
    print("\n=== PORTICO Model: Business Rules Demo ===\n")
    
    portico = PorticoModel()
    
    # Test scenarios that should trigger business rules
    test_scenarios = [
        {
            'name': 'High-Risk Customer Seeking Credit Increase',
            'state': State(credit_line=3, apr=4.0, behavior_vars=np.array([1, 1, 1, 1, 1, 1])),
            'action': Action(ActionType.INCREASE_CREDIT_LINE, 1000)
        },
        {
            'name': 'Large Credit Line Increase Request',
            'state': State(credit_line=5, apr=2.5, behavior_vars=np.array([3, 3, 3, 3, 3, 3])),
            'action': Action(ActionType.INCREASE_CREDIT_LINE, 5000)
        },
        {
            'name': 'Large APR Increase',
            'state': State(credit_line=3, apr=2.0, behavior_vars=np.array([2, 2, 2, 2, 2, 2])),
            'action': Action(ActionType.INCREASE_APR, 5.0)
        },
        {
            'name': 'Normal Credit Increase',
            'state': State(credit_line=3, apr=2.5, behavior_vars=np.array([3, 3, 3, 3, 3, 3])),
            'action': Action(ActionType.INCREASE_CREDIT_LINE, 1000)
        }
    ]
    
    for scenario in test_scenarios:
        print(f"Scenario: {scenario['name']}")
        print(f"Current State: Credit Line ${scenario['state'].credit_line * 1000:,}, APR {scenario['state'].apr:.1f}%")
        print(f"Risk Score: {np.mean(scenario['state'].behavior_vars)/4:.2f}")
        print(f"Proposed Action: {scenario['action'].action_type.name} ${scenario['action'].magnitude:,.0f}")
        
        allowed = portico.apply_business_rules(scenario['state'], scenario['action'])
        print(f"Action Allowed: {'✓ YES' if allowed else '✗ NO'}")
        
        if not allowed:
            risk_score = np.mean(scenario['state'].behavior_vars) / 4
            if (risk_score > portico.business_rules['risk_threshold'] and 
                scenario['action'].action_type == ActionType.INCREASE_CREDIT_LINE):
                print("  Reason: Customer exceeds risk threshold")
            elif (scenario['action'].action_type == ActionType.INCREASE_CREDIT_LINE and 
                  scenario['action'].magnitude > portico.business_rules['max_line_increase']):
                print("  Reason: Increase amount exceeds maximum allowed")
            elif (scenario['action'].action_type in [ActionType.INCREASE_APR, ActionType.DECREASE_APR] and 
                  scenario['action'].magnitude > portico.business_rules['max_apr_change']):
                print("  Reason: APR change exceeds maximum allowed")
        
        print("-" * 50)


def main():
    """Run all PORTICO demonstrations"""
    print("PORTICO MODEL DEMONSTRATION")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Individual customer decisions
        demonstrate_single_customer_decisions()
        
        # Portfolio optimization
        demonstrate_portfolio_optimization()
        
        # State value analysis
        demonstrate_state_value_analysis()
        
        # Business rules
        demonstrate_business_rules()
        
        # Create visualizations
        create_decision_visualization()
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("PORTICO DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    main() 