"""
PORTICO Model: Dynamic Credit Line and Pricing Management
Implementation based on "Managing Credit Lines and Prices for Bank One Credit Cards"

This module implements the PORTICO Markov Decision Process (MDP) model for
optimal credit line and price decisions in credit card portfolios.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize
import pickle


class ActionType(Enum):
    """Types of actions available in the PORTICO model"""
    DO_NOTHING = 0
    INCREASE_CREDIT_LINE = 1
    DECREASE_CREDIT_LINE = 2
    INCREASE_APR = 3
    DECREASE_APR = 4


@dataclass
class State:
    """Represents a state in the PORTICO MDP model"""
    credit_line: int  # Credit line level (control variable)
    apr: float       # APR level (control variable)
    behavior_vars: np.ndarray  # Behavior variables (Beh1-Beh6)
    
    def __hash__(self):
        return hash((self.credit_line, self.apr, tuple(self.behavior_vars)))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self.credit_line == other.credit_line and 
                self.apr == other.apr and 
                np.array_equal(self.behavior_vars, other.behavior_vars))


@dataclass
class Action:
    """Represents an action in the PORTICO model"""
    action_type: ActionType
    magnitude: float = 0.0  # Amount of change (e.g., $1000 for credit line, 2.5% for APR)


@dataclass
class PolicyEntry:
    """Entry in the policy table"""
    state: State
    optimal_action: Action
    expected_value: float


class PorticoModel:
    """
    PORTICO MDP Model for Dynamic Credit Line and Pricing Management
    
    This class implements the Markov Decision Process model described in the paper,
    including state management, transition matrices, reward calculation, and
    policy optimization using value iteration.
    """
    
    def __init__(self, 
                 time_horizon: int = 36,
                 discount_factor: float = 0.95,
                 update_frequency: int = 6,  # months between decision epochs
                 n_behavior_vars: int = 6):
        """
        Initialize the PORTICO model
        
        Args:
            time_horizon: Planning horizon in months (default 36)
            discount_factor: One-period discount factor β
            update_frequency: Months between decision epochs
            n_behavior_vars: Number of behavior variables
        """
        self.time_horizon = time_horizon
        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.n_behavior_vars = n_behavior_vars
        
        # Model components
        self.states: List[State] = []
        self.transition_matrices: Dict[ActionType, np.ndarray] = {}
        self.policy_table: Dict[State, PolicyEntry] = {}
        self.value_function: Dict[State, float] = {}
        
        # Constraints and boundaries
        self.credit_line_min = 1
        self.credit_line_max = 10
        self.apr_min = 1.0
        self.apr_max = 5.0
        self.behavior_var_levels = [1, 2, 3, 4]  # Discrete levels for behavior variables
        
        # Business rules and constraints
        self.business_rules = {
            'max_line_increase': 3000,
            'max_apr_change': 2.5,
            'risk_threshold': 0.8,  # Risk score threshold for approvals
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_states(self) -> List[State]:
        """
        Generate all possible states in the MDP
        
        Returns:
            List of all states
        """
        states = []
        
        for credit_line in range(self.credit_line_min, self.credit_line_max + 1):
            for apr_level in range(1, 6):  # APR levels 1-5
                # Generate behavior variable combinations
                for beh_combo in self._generate_behavior_combinations():
                    state = State(
                        credit_line=credit_line,
                        apr=apr_level + self.apr_min - 1,
                        behavior_vars=np.array(beh_combo)
                    )
                    states.append(state)
        
        self.states = states
        self.logger.info(f"Generated {len(states)} states")
        return states
    
    def _generate_behavior_combinations(self) -> List[List[int]]:
        """Generate all combinations of behavior variables"""
        from itertools import product
        
        combinations = list(product(self.behavior_var_levels, repeat=self.n_behavior_vars))
        return [list(combo) for combo in combinations]
    
    def get_available_actions(self, state: State) -> List[Action]:
        """
        Get available actions for a given state
        
        Args:
            state: Current state
            
        Returns:
            List of available actions
        """
        actions = [Action(ActionType.DO_NOTHING)]
        
        # Credit line actions
        if state.credit_line < self.credit_line_max:
            actions.append(Action(ActionType.INCREASE_CREDIT_LINE, 1000))
            actions.append(Action(ActionType.INCREASE_CREDIT_LINE, 3000))
        
        if state.credit_line > self.credit_line_min:
            actions.append(Action(ActionType.DECREASE_CREDIT_LINE, 1000))
        
        # APR actions
        if state.apr < self.apr_max:
            actions.append(Action(ActionType.INCREASE_APR, 2.5))
        
        if state.apr > self.apr_min:
            actions.append(Action(ActionType.DECREASE_APR, 2.5))
        
        return actions
    
    def apply_action(self, state: State, action: Action) -> State:
        """
        Apply an action to a state and return the new state
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            New state after applying action
        """
        new_credit_line = state.credit_line
        new_apr = state.apr
        
        if action.action_type == ActionType.INCREASE_CREDIT_LINE:
            new_credit_line = min(state.credit_line + 1, self.credit_line_max)
        elif action.action_type == ActionType.DECREASE_CREDIT_LINE:
            new_credit_line = max(state.credit_line - 1, self.credit_line_min)
        elif action.action_type == ActionType.INCREASE_APR:
            new_apr = min(state.apr + 0.5, self.apr_max)
        elif action.action_type == ActionType.DECREASE_APR:
            new_apr = max(state.apr - 0.5, self.apr_min)
        
        return State(
            credit_line=new_credit_line,
            apr=new_apr,
            behavior_vars=state.behavior_vars.copy()
        )
    
    def calculate_ncf(self, state: State) -> float:
        """
        Calculate Net Cash Flow (NCF) for a given state
        
        Args:
            state: Current state
            
        Returns:
            Net cash flow for the state
        """
        # Base revenue from credit line utilization
        utilization_rate = 0.3 + 0.1 * (state.behavior_vars[0] - 1) / 3  # Based on behavior
        monthly_balance = state.credit_line * 1000 * utilization_rate
        interest_revenue = monthly_balance * (state.apr / 100) / 12
        
        # Costs and risk adjustments
        risk_score = np.mean(state.behavior_vars) / 4  # Normalized risk score
        charge_off_rate = 0.02 * risk_score  # Higher behavior scores = higher risk
        charge_off_cost = monthly_balance * charge_off_rate
        
        # Operational costs
        operational_cost = 50 + 10 * state.credit_line  # Fixed + variable costs
        
        # Net cash flow
        ncf = interest_revenue - charge_off_cost - operational_cost
        
        # Penalty for inactive accounts (low behavior scores)
        if np.mean(state.behavior_vars) < 2:
            ncf -= 100  # Penalty for inactive accounts
        
        return ncf
    
    def create_transition_matrix(self, action_type: ActionType) -> np.ndarray:
        """
        Create transition matrix for a given action type
        
        Args:
            action_type: Type of action
            
        Returns:
            Transition probability matrix
        """
        n_states = len(self.states)
        transition_matrix = np.zeros((n_states, n_states))
        
        # Create state index mapping
        state_to_idx = {state: i for i, state in enumerate(self.states)}
        
        for i, current_state in enumerate(self.states):
            # Behavior transitions (stochastic part)
            current_behavior_mean = np.mean(current_state.behavior_vars)
            
            for j, next_state in enumerate(self.states):
                # Only consider states with same control variables for behavior transitions
                if (current_state.credit_line == next_state.credit_line and 
                    current_state.apr == next_state.apr):
                    
                    # Calculate transition probability based on behavior similarity
                    behavior_diff = np.linalg.norm(
                        current_state.behavior_vars - next_state.behavior_vars
                    )
                    
                    # Higher probability for similar behavior states
                    if behavior_diff == 0:
                        prob = 0.4  # Stay in same behavior state
                    elif behavior_diff == 1:
                        prob = 0.3  # Move to adjacent behavior state
                    elif behavior_diff == 2:
                        prob = 0.2  # Move to nearby behavior state
                    else:
                        prob = 0.1 / max(1, behavior_diff - 2)  # Low prob for distant states
                    
                    transition_matrix[i, j] = prob
        
        # Normalize rows to ensure they sum to 1
        for i in range(n_states):
            row_sum = np.sum(transition_matrix[i, :])
            if row_sum > 0:
                transition_matrix[i, :] /= row_sum
            else:
                # If no transitions defined, stay in same state
                transition_matrix[i, i] = 1.0
        
        return transition_matrix
    
    def solve_mdp(self) -> Dict[State, PolicyEntry]:
        """
        Solve the MDP using value iteration to find optimal policy
        
        Returns:
            Optimal policy table
        """
        if not self.states:
            self.generate_states()
        
        # Initialize value function
        for state in self.states:
            self.value_function[state] = self.calculate_ncf(state)
        
        # Create transition matrices for different action types
        for action_type in ActionType:
            self.transition_matrices[action_type] = self.create_transition_matrix(action_type)
        
        self.logger.info("Starting value iteration...")
        
        # Value iteration
        for iteration in range(self.time_horizon):
            new_value_function = {}
            
            for state in self.states:
                if iteration % self.update_frequency == 0:
                    # Update epoch - can take actions
                    best_value = float('-inf')
                    best_action = Action(ActionType.DO_NOTHING)
                    
                    for action in self.get_available_actions(state):
                        # Calculate expected value for this action
                        next_state = self.apply_action(state, action)
                        immediate_reward = self.calculate_ncf(next_state)
                        
                        # Expected future value
                        expected_future_value = 0.0
                        state_idx = self.states.index(state)
                        
                        for j, future_state in enumerate(self.states):
                            transition_prob = self.transition_matrices[action.action_type][state_idx, j]
                            expected_future_value += (transition_prob * 
                                                    self.value_function.get(future_state, 0))
                        
                        total_value = immediate_reward + self.discount_factor * expected_future_value
                        
                        if total_value > best_value:
                            best_value = total_value
                            best_action = action
                    
                    new_value_function[state] = best_value
                    self.policy_table[state] = PolicyEntry(state, best_action, best_value)
                else:
                    # Non-update period - no action taken
                    immediate_reward = self.calculate_ncf(state)
                    state_idx = self.states.index(state)
                    
                    expected_future_value = 0.0
                    for j, future_state in enumerate(self.states):
                        transition_prob = self.transition_matrices[ActionType.DO_NOTHING][state_idx, j]
                        expected_future_value += (transition_prob * 
                                                self.value_function.get(future_state, 0))
                    
                    new_value_function[state] = (immediate_reward + 
                                               self.discount_factor * expected_future_value)
            
            self.value_function = new_value_function
            
            if iteration % 6 == 0:
                self.logger.info(f"Completed iteration {iteration}")
        
        self.logger.info("Value iteration completed")
        return self.policy_table
    
    def get_optimal_action(self, state: State) -> Action:
        """
        Get optimal action for a given state
        
        Args:
            state: Current state
            
        Returns:
            Optimal action
        """
        if state in self.policy_table:
            return self.policy_table[state].optimal_action
        else:
            # Find closest state if exact match not found
            closest_state = min(self.policy_table.keys(), 
                              key=lambda s: self._state_distance(s, state))
            return self.policy_table[closest_state].optimal_action
    
    def _state_distance(self, state1: State, state2: State) -> float:
        """Calculate distance between two states"""
        credit_diff = abs(state1.credit_line - state2.credit_line)
        apr_diff = abs(state1.apr - state2.apr)
        behavior_diff = np.linalg.norm(state1.behavior_vars - state2.behavior_vars)
        
        return credit_diff + apr_diff + behavior_diff
    
    def apply_business_rules(self, state: State, action: Action) -> bool:
        """
        Check if an action violates business rules
        
        Args:
            state: Current state
            action: Proposed action
            
        Returns:
            True if action is allowed, False otherwise
        """
        # Risk threshold check
        risk_score = np.mean(state.behavior_vars) / 4
        if risk_score > self.business_rules['risk_threshold']:
            if action.action_type == ActionType.INCREASE_CREDIT_LINE:
                return False
        
        # Credit line increase limits
        if (action.action_type == ActionType.INCREASE_CREDIT_LINE and 
            action.magnitude > self.business_rules['max_line_increase']):
            return False
        
        # APR change limits
        if (action.action_type in [ActionType.INCREASE_APR, ActionType.DECREASE_APR] and 
            action.magnitude > self.business_rules['max_apr_change']):
            return False
        
        return True
    
    def save_model(self, filepath: str):
        """Save the trained model to file"""
        model_data = {
            'policy_table': self.policy_table,
            'value_function': self.value_function,
            'states': self.states,
            'transition_matrices': self.transition_matrices,
            'parameters': {
                'time_horizon': self.time_horizon,
                'discount_factor': self.discount_factor,
                'update_frequency': self.update_frequency,
                'n_behavior_vars': self.n_behavior_vars
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.policy_table = model_data['policy_table']
        self.value_function = model_data['value_function']
        self.states = model_data['states']
        self.transition_matrices = model_data['transition_matrices']
        
        params = model_data['parameters']
        self.time_horizon = params['time_horizon']
        self.discount_factor = params['discount_factor']
        self.update_frequency = params['update_frequency']
        self.n_behavior_vars = params['n_behavior_vars']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def generate_policy_report(self) -> pd.DataFrame:
        """
        Generate a policy report showing optimal actions for all states
        
        Returns:
            DataFrame with policy recommendations
        """
        report_data = []
        
        for state, policy_entry in self.policy_table.items():
            report_data.append({
                'Credit_Line': state.credit_line,
                'APR': state.apr,
                'Behavior_1': state.behavior_vars[0],
                'Behavior_2': state.behavior_vars[1],
                'Behavior_3': state.behavior_vars[2],
                'Behavior_4': state.behavior_vars[3],
                'Behavior_5': state.behavior_vars[4],
                'Behavior_6': state.behavior_vars[5],
                'Optimal_Action': policy_entry.optimal_action.action_type.name,
                'Action_Magnitude': policy_entry.optimal_action.magnitude,
                'Expected_Value': policy_entry.expected_value
            })
        
        return pd.DataFrame(report_data) 