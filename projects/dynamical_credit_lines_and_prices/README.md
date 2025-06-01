# PORTICO Model: Dynamic Credit Line and Pricing Management

This implementation recreates the PORTICO (Portfolio Optimization for Risk-adjusted Target Income with Credit Operations) model from the Bank One Credit Cards research paper. The model uses Markov Decision Processes (MDP) to optimize credit line and pricing decisions for credit card portfolios.

## Overview

The PORTICO model addresses the challenge of dynamically managing credit lines and pricing for credit card customers by:

- **Modeling customer states** using control variables (credit line, APR) and behavior variables
- **Optimizing decisions** using value iteration on a Markov Decision Process
- **Incorporating business rules** and risk constraints
- **Maximizing expected Net Present Value (NPV)** over a specified time horizon

## Model Components

### 1. State Representation
Each customer state is defined by:
- **Control Variables**: Credit line level, APR level
- **Behavior Variables**: 6 behavioral indicators (payment patterns, utilization, etc.)

### 2. Actions
Available actions include:
- Do nothing
- Increase/decrease credit line
- Increase/decrease APR

### 3. Rewards
Net Cash Flow (NCF) calculation considering:
- Interest revenue from balances
- Charge-off costs based on risk
- Operational costs
- Penalties for inactive accounts

### 4. Transition Dynamics
Customer behavior evolution modeled through:
- Stochastic behavior variable transitions
- Control variable changes through actions
- Action-independent transition matrices (simplified approach)

## Files

### Core Implementation
- **`portico_model.py`**: Main PORTICO MDP model implementation
- **`portfolio_simulator.py`**: Portfolio-level simulation and management
- **`demo_portico.py`**: Demonstration script showing model capabilities

### Documentation
- **`README.md`**: This file
- **`requirements.txt`**: Python package dependencies

## Installation

1. Install required packages:
```bash
uv add numpy pandas matplotlib seaborn scipy scikit-learn
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from portico_model import PorticoModel
from portfolio_simulator import PortfolioSimulator

# Initialize and train PORTICO model
portico = PorticoModel(
    time_horizon=36,        # 36-month planning horizon
    discount_factor=0.95,   # Monthly discount factor
    update_frequency=6      # Decision epochs every 6 months
)

# Generate states and solve MDP
policy_table = portico.solve_mdp()

# Create portfolio simulator
simulator = PortfolioSimulator(
    portico_model=portico,
    n_customers=1000,
    simulation_months=36
)

# Run simulation
simulator.run_simulation()

# Get results
monthly_df, customer_df = simulator.generate_performance_report()
```

### Running the Demo

```python
python demo_portico.py
```

This will demonstrate:
- Individual customer decision examples
- Portfolio optimization results
- State value analysis
- Business rule enforcement
- Decision visualizations

## Key Features

### 1. Markov Decision Process Framework
- **States**: Discrete representation of customer characteristics
- **Actions**: Credit line and pricing decisions
- **Rewards**: Net cash flow calculations
- **Transitions**: Behavior evolution modeling
- **Value Iteration**: Optimal policy computation

### 2. Business Rule Integration
- Risk threshold constraints for credit increases
- Maximum credit line increase limits
- APR change limitations
- Post-optimization constraint enforcement

### 3. Portfolio Management
- Multi-customer simulation
- Monthly decision cycles
- Performance tracking and reporting
- Baseline comparison capabilities

### 4. Risk Management
- Customer risk scoring based on behavior variables
- Charge-off probability modeling
- Risk-adjusted decision making
- Portfolio-level risk metrics

## Model Parameters

### Default Configuration
- **Time Horizon**: 36 months
- **Discount Factor**: 0.95 (monthly)
- **Update Frequency**: 6 months
- **Credit Line Levels**: 1-10 ($1K-$10K)
- **APR Range**: 1.0% - 5.0%
- **Behavior Variables**: 6 variables, levels 1-4

### Business Rules
- **Maximum Credit Increase**: $3,000
- **Maximum APR Change**: 2.5%
- **Risk Threshold**: 0.8 (normalized)

## Performance Metrics

The model tracks various portfolio performance indicators:

### Revenue Metrics
- Total monthly revenue
- Interest income from balances
- Average revenue per customer
- Revenue volatility

### Risk Metrics
- Total charge-offs
- Charge-off rate
- Customer risk distribution
- Portfolio risk exposure

### Operational Metrics
- Action distribution
- Utilization rates
- Customer retention
- Return on assets

### Decision Quality
- Improvement over baseline (no optimization)
- Value function convergence
- Policy stability across states

## Theoretical Foundation

The PORTICO model is based on:

### Markov Decision Process Theory
- **Bellman Equation**: V_t(s) = max_a {r(s,a) + β∑p(s'|s,a)V_{t+1}(s')}
- **Value Iteration**: Iterative computation of optimal value function
- **Policy Extraction**: Optimal actions from value function

### Financial Optimization
- **Net Present Value**: Discounted sum of future cash flows
- **Risk-Return Tradeoff**: Balancing revenue and charge-off risk
- **Dynamic Programming**: Sequential decision optimization

### Credit Risk Management
- **Behavioral Modeling**: Customer state transitions
- **Portfolio Theory**: Diversification and correlation effects
- **Regulatory Constraints**: Business rule incorporation

## Limitations and Assumptions

### Model Simplifications
- Discrete state space (finite states)
- Action-independent transitions
- Simplified NCF calculations
- Monthly decision granularity

### Business Assumptions
- Static economic environment
- No customer acquisition/attrition
- Perfect information about customer states
- Homogeneous customer segments

### Technical Constraints
- Computational complexity scales with state space size
- Memory requirements for large portfolios
- Convergence depends on parameter choices

## Extensions and Future Work

### Potential Enhancements
1. **Continuous State Spaces**: Function approximation methods
2. **Customer Acquisition**: New customer onboarding decisions
3. **Economic Cycles**: Time-varying parameters
4. **Multi-Product**: Cross-selling optimization
5. **Real-time Learning**: Online model updates

### Research Directions
1. **Deep Reinforcement Learning**: Neural network value functions
2. **Multi-Agent Systems**: Customer interaction modeling
3. **Robust Optimization**: Uncertainty quantification
4. **Behavioral Economics**: Advanced customer modeling

## References

- Original Paper: "Managing Credit Lines and Prices for Bank One Credit Cards"
- Markov Decision Processes: Puterman (1994)
- Dynamic Programming: Bellman (1957)
- Credit Risk Management: Basel II/III frameworks

## License

This implementation is provided for educational and research purposes. Commercial use should comply with applicable regulations and intellectual property rights.

## Contact

For questions or contributions, please refer to the project documentation or submit issues through the appropriate channels. 