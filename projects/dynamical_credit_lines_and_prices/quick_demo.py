#!/usr/bin/env python3
"""
Quick demonstration of PORTICO model decision-making
"""

from portico_model import PorticoModel, State, ActionType
from portfolio_simulator import PortfolioSimulator
import numpy as np

def demo_customer_decisions():
    """Demonstrate PORTICO decisions for sample customers"""
    print("🏦 PORTICO MODEL DEMONSTRATION")
    print("=" * 50)
    
    # Create a simplified model for quick demo
    portico = PorticoModel(
        time_horizon=12,
        discount_factor=0.95,
        update_frequency=6
    )
    # Simplify for faster computation
    portico.credit_line_max = 5
    portico.behavior_var_levels = [1, 2, 3]
    
    print("Training PORTICO model...")
    policy_table = portico.solve_mdp()
    print(f"✓ Trained with {len(policy_table)} policy entries\n")
    
    # Example customer scenarios
    customers = [
        {
            'name': 'Alice (Good Customer)',
            'state': State(credit_line=2, apr=2.5, behavior_vars=np.array([3, 3, 3, 3, 3, 3])),
            'description': 'Low risk, good payment history'
        },
        {
            'name': 'Bob (Risky Customer)',
            'state': State(credit_line=3, apr=4.0, behavior_vars=np.array([1, 1, 1, 1, 1, 1])),
            'description': 'High risk, poor payment history'
        },
        {
            'name': 'Carol (Average Customer)',
            'state': State(credit_line=3, apr=3.0, behavior_vars=np.array([2, 2, 2, 2, 2, 2])),
            'description': 'Medium risk, average behavior'
        }
    ]
    
    for customer in customers:
        print(f"Customer: {customer['name']}")
        print(f"Description: {customer['description']}")
        
        state = customer['state']
        print(f"Current State:")
        print(f"  💳 Credit Line: ${state.credit_line * 1000:,}")
        print(f"  📊 APR: {state.apr:.1f}%")
        print(f"  ⭐ Behavior Score: {np.mean(state.behavior_vars):.1f}/3.0")
        
        # Calculate current profitability
        ncf = portico.calculate_ncf(state)
        print(f"  💰 Monthly NCF: ${ncf:.2f}")
        
        # Get optimal action
        try:
            optimal_action = portico.get_optimal_action(state)
            
            # Check business rules
            if portico.apply_business_rules(state, optimal_action):
                new_state = portico.apply_action(state, optimal_action)
                new_ncf = portico.calculate_ncf(new_state)
                
                print(f"🎯 Recommended Action: {optimal_action.action_type.name}")
                if optimal_action.magnitude > 0:
                    if 'LINE' in optimal_action.action_type.name:
                        print(f"  📈 Amount: ${optimal_action.magnitude:,.0f}")
                    else:
                        print(f"  📈 Amount: {optimal_action.magnitude:.1f}%")
                
                if optimal_action.action_type != ActionType.DO_NOTHING:
                    print(f"  💳 New Credit Line: ${new_state.credit_line * 1000:,}")
                    print(f"  📊 New APR: {new_state.apr:.1f}%")
                    print(f"  💰 Expected NCF: ${new_ncf:.2f}")
                    print(f"  📈 NCF Change: ${new_ncf - ncf:+.2f}")
            else:
                print(f"🚫 Recommended Action: {optimal_action.action_type.name} (BLOCKED by business rules)")
                print(f"  ⚠️  Risk threshold or limit exceeded")
                
        except Exception as e:
            print(f"🎯 Recommended Action: DO_NOTHING (no policy match)")
        
        print("-" * 50)

def demo_portfolio_simulation():
    """Quick portfolio simulation demo"""
    print("\n📊 PORTFOLIO SIMULATION DEMO")
    print("=" * 30)
    
    # Create simplified model
    portico = PorticoModel(time_horizon=6, discount_factor=0.95, update_frequency=3)
    portico.credit_line_max = 3
    portico.behavior_var_levels = [1, 2, 3]
    
    print("Training model...")
    portico.solve_mdp()
    
    # Create small portfolio
    simulator = PortfolioSimulator(
        portico_model=portico,
        n_customers=50,  # Small portfolio for demo
        simulation_months=6
    )
    
    print("Running 6-month simulation with 50 customers...")
    simulator.run_simulation()
    
    # Show results
    metrics = simulator.portfolio_metrics
    print(f"\n📈 RESULTS:")
    print(f"Total Revenue: ${metrics['total_revenue']:,.2f}")
    print(f"Total Charge-offs: ${metrics['total_charge_offs']:,.2f}")
    print(f"Net Income: ${metrics['net_income']:,.2f}")
    print(f"Average Utilization: {metrics['avg_utilization']:.1%}")
    
    if 'action_distribution' in metrics:
        print(f"\n🎯 ACTIONS TAKEN:")
        for action, pct in metrics['action_distribution'].items():
            if pct > 0:
                print(f"  {action.replace('_', ' ').title()}: {pct:.1%}")
    
    # Compare with baseline
    print(f"\n⚖️  BASELINE COMPARISON:")
    comparison, _ = simulator.compare_with_baseline()
    improvement = comparison['improvement_pct']
    if improvement > 0:
        print(f"  🎉 PORTICO improved performance by {improvement:.1f}%!")
    else:
        print(f"  📉 PORTICO underperformed by {abs(improvement):.1f}%")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demo_customer_decisions()
        demo_portfolio_simulation()
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("For a full demo with visualizations, run: python demo_portico.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc() 