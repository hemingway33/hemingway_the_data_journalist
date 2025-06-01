#!/usr/bin/env python3
"""
Quick test script for PORTICO model implementation
"""

from portico_model import PorticoModel, State, ActionType
import numpy as np

def test_portico_basic():
    """Test basic PORTICO model functionality"""
    print('Testing PORTICO Model Implementation...')

    # Create a simple model
    portico = PorticoModel(time_horizon=6, discount_factor=0.95, update_frequency=3)
    portico.credit_line_max = 3  # Smaller state space for testing
    portico.behavior_var_levels = [1, 2]  # Reduced complexity

    # Generate states
    states = portico.generate_states()
    print(f'✓ Generated {len(states)} states')

    # Test NCF calculation
    test_state = State(credit_line=2, apr=3.0, behavior_vars=np.array([2, 2, 2, 2, 2, 2]))
    ncf = portico.calculate_ncf(test_state)
    print(f'✓ Test NCF calculation: ${ncf:.2f}')

    # Test action generation
    actions = portico.get_available_actions(test_state)
    print(f'✓ Available actions: {len(actions)}')
    for action in actions:
        print(f'  - {action.action_type.name} (magnitude: {action.magnitude})')

    # Test state transition
    action = actions[1] if len(actions) > 1 else actions[0]
    new_state = portico.apply_action(test_state, action)
    print(f'✓ State transition test passed')
    print(f'  Original: Credit Line {test_state.credit_line}, APR {test_state.apr:.1f}%')
    print(f'  New: Credit Line {new_state.credit_line}, APR {new_state.apr:.1f}%')

    # Test business rules
    risky_state = State(credit_line=3, apr=4.0, behavior_vars=np.array([1, 1, 1, 1, 1, 1]))
    risky_action = next((a for a in portico.get_available_actions(risky_state) 
                        if a.action_type == ActionType.INCREASE_CREDIT_LINE), None)
    
    if risky_action:
        allowed = portico.apply_business_rules(risky_state, risky_action)
        print(f'✓ Business rules test: High-risk credit increase {"allowed" if allowed else "blocked"}')

    print('\n✓ PORTICO model basic functionality test passed!')
    return True

if __name__ == "__main__":
    test_portico_basic() 