# Digital Twin for Loan Portfolio Management & Optimization

A comprehensive digital twin system for loan business management that integrates portfolio optimization, risk management, and reinforcement learning to continuously improve lending decisions and portfolio performance.

## 🎯 **Overview**

This digital twin serves multiple critical functions:

1. **Real-time Portfolio Simulation** - Creates a virtual representation of your loan portfolio
2. **Risk Management & Stress Testing** - Evaluates portfolio performance under various economic scenarios
3. **Business Process Optimization** - Continuously improves loan products and operational processes
4. **Reinforcement Learning Environment** - Trains AI agents to make optimal lending decisions
5. **Performance Prediction** - Forecasts portfolio metrics and identifies optimization opportunities

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│  Business Layer     │  Portfolio Layer    │  Intelligence Layer │
│  ┌─────────────────┐│  ┌─────────────────┐│  ┌─────────────────┐│
│  │ Loan Products   ││  │ Risk Mgmt       ││  │ ML Models       ││
│  │ Credit Policies ││  │ Optimization    ││  │ Decision Engine ││
│  │ Processes       ││  │ Performance     ││  │ Learning        ││
│  └─────────────────┘│  └─────────────────┘│  └─────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  Environment Layer              │  Data Layer                    │
│  ┌─────────────────────────────┐│  ┌─────────────────────────────┐│
│  │ RL Environment              ││  │ Data Connectors             ││
│  │ Multi-Agent Framework       ││  │ Feature Engineering         ││
│  │ Simulation & Training       ││  │ Synthetic Data Generation   ││
│  └─────────────────────────────┘│  └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 **Quick Start**

### Prerequisites

- Python 3.12+
- UV package manager
- Basic understanding of loan portfolio management

### Installation

```bash
# Navigate to the project directory
cd projects/twin_portfolio_agent

# Install dependencies (already configured in parent project)
# Dependencies: gymnasium, scipy, pandas, numpy, torch, etc.

# Test the environment
python twin_env.py
```

### Basic Usage

```python
from twin_env import LoanPortfolioTwinEnv

# Create the digital twin environment
env = LoanPortfolioTwinEnv(
    initial_portfolio_size=1000,
    max_portfolio_size=10000,
    simulation_days=365,
    render_mode="human"
)

# Reset environment
observation, info = env.reset()

# Take actions and observe results
for step in range(100):
    # Action: [credit_policy_adj, pricing_adj, portfolio_rebalancing]
    action = env.action_space.sample()  # Random action for demo
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 10 == 0:
        env.render()  # Display current state
    
    if terminated or truncated:
        break

# Get performance analytics
performance_df = env.get_performance_summary()
print(performance_df.describe())
```

## 📊 **Core Features**

### 1. **Loan Portfolio Simulation**

- **Realistic Loan Generation**: Creates synthetic loans with authentic characteristics
- **Multiple Loan Types**: Consumer, Auto, Mortgage, Business, Credit Card
- **Risk Factor Modeling**: Credit scores, LTV ratios, DTI ratios, behavioral patterns
- **Lifecycle Management**: Origination through charge-off or prepayment

### 2. **Portfolio Risk Management**

- **Value-at-Risk (VaR)**: 95% confidence interval risk assessment
- **Expected Loss Modeling**: Forward-looking loss predictions
- **Stress Testing**: Portfolio performance under adverse scenarios
- **Concentration Risk**: Monitoring portfolio diversification
- **Regulatory Compliance**: Capital adequacy and liquidity requirements

### 3. **Performance Optimization**

- **Multi-Objective Optimization**: Balance return, risk, and regulatory requirements
- **Dynamic Policy Adjustment**: Real-time credit and pricing policy updates
- **Portfolio Rebalancing**: Optimal asset allocation decisions
- **Performance Attribution**: Identify sources of portfolio performance

### 4. **Reinforcement Learning Environment**

- **Gymnasium Compatibility**: Standard RL environment interface
- **Multi-Agent Support**: Specialized agents for different decision domains
- **Continuous Action Space**: Fine-grained policy adjustments
- **Rich Observation Space**: Comprehensive state representation
- **Reward Engineering**: Multi-criteria reward functions

## 🎮 **Action & Observation Spaces**

### Action Space
The environment accepts 3-dimensional continuous actions:

```python
action_space = Box(
    low=[-0.1, -0.02, -0.1],   # [credit_policy, pricing, rebalancing]
    high=[0.1, 0.02, 0.1],     # Maximum ±10% policy, ±2% pricing, ±10% rebalancing
    dtype=np.float32
)
```

- **Credit Policy Adjustment** (-0.1 to 0.1): Modify minimum credit score requirements
- **Pricing Adjustment** (-0.02 to 0.02): Adjust interest rate spreads
- **Portfolio Rebalancing** (-0.1 to 0.1): Modify loan type allocations

### Observation Space
13-dimensional observation vector:

```python
observation_space = Box(
    low=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, -0.1, 0.0, 50.0, 0.0, 5.0],
    high=[1e9, 0.5, 0.3, 1.0, 0.2, 1.0, 0.2, 0.2, 0.1, 0.1, 200.0, 0.1, 50.0],
    dtype=np.float32
)
```

**Portfolio Metrics:**
- Total portfolio value (millions)
- Expected loss rate
- Delinquency rate
- Return on equity
- Value-at-Risk (95%)
- Concentration risk

**Market Conditions:**
- Base interest rate
- Unemployment rate
- GDP growth rate
- Inflation rate
- Housing price index
- Credit spread
- Volatility index

## 🏆 **Reward Function**

The environment uses a multi-objective reward function that balances:

1. **Financial Performance** (40%): Risk-adjusted returns, ROE targets
2. **Risk Management** (30%): VaR limits, expected loss control
3. **Regulatory Compliance** (20%): Capital adequacy, liquidity ratios
4. **Operational Efficiency** (10%): Portfolio growth, processing efficiency

```python
reward = (
    roe_reward +           # Financial performance
    var_penalty +          # Risk management
    delinq_penalty +       # Credit quality
    compliance_bonus +     # Regulatory adherence
    growth_reward          # Business expansion
)
```

## 📈 **Performance Metrics**

### Financial Metrics
- **Return on Equity (ROE)**: Target 15%+
- **Net Interest Margin (NIM)**: Interest income efficiency
- **Return on Assets (ROA)**: Asset utilization effectiveness

### Risk Metrics
- **Value-at-Risk (VaR)**: Maximum loss at 95% confidence
- **Expected Loss**: Forward-looking loss estimates
- **Charge-off Rate**: Actual loss realization
- **Delinquency Rate**: Early warning indicator

### Operational Metrics
- **Portfolio Growth Rate**: Business expansion
- **Processing Efficiency**: Decision automation rate
- **Customer Satisfaction**: Approval rates and service quality

## 🔧 **Configuration**

### Environment Parameters

```python
env = LoanPortfolioTwinEnv(
    initial_portfolio_size=1000,    # Starting number of loans
    max_portfolio_size=10000,       # Maximum portfolio capacity
    simulation_days=365,            # Episode length in days
    render_mode="human"             # Visualization mode
)
```

### Business Policies

```python
@dataclass
class BusinessPolicies:
    max_ltv: float = 0.80              # Maximum loan-to-value ratio
    min_credit_score: float = 600      # Minimum credit score requirement
    max_dti: float = 0.43              # Maximum debt-to-income ratio
    max_concentration_by_type: float = 0.40  # Portfolio concentration limit
    target_roe: float = 0.15           # Target return on equity
    max_var_limit: float = 0.05        # Maximum VaR threshold
    min_capital_ratio: float = 0.08    # Regulatory capital requirement
```

## 🧪 **Integration with Existing Projects**

This digital twin leverages components from your existing codebase:

- **`RL_learner/`**: DQN and MuZero implementations for agent training
- **`loan_portfolio_stress_testing/`**: Advanced stress testing models
- **`active_portfolio_mgmt/`**: Portfolio optimization algorithms
- **`survival_model/`**: Survival analysis for prepayment and default modeling
- **`portfolio_tracking/`**: Performance monitoring and attribution

## 🔄 **Development Roadmap**

### Phase 1: Foundation (✅ Complete)
- [x] Core environment implementation
- [x] Basic loan and portfolio entities
- [x] Synthetic data generation
- [x] RL environment interface

### Phase 2: Business Logic (🚧 In Progress)
- [ ] Advanced loan product definitions
- [ ] Credit policy engine
- [ ] Regulatory compliance framework
- [ ] Process automation

### Phase 3: Intelligence Layer (📋 Planned)
- [ ] ML model integration
- [ ] Decision engine implementation
- [ ] Continual learning pipeline
- [ ] Model monitoring and validation

### Phase 4: Integration (📋 Planned)
- [ ] Real-time data connectors
- [ ] Dashboard and API development
- [ ] Production deployment pipeline
- [ ] A/B testing framework

## 📚 **Documentation**

- **[Project Structure](PROJECT_STRUCTURE.md)**: Detailed system architecture
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)**: Development roadmap and technical details
- **[API Reference](docs/api_reference.md)**: Complete API documentation *(coming soon)*
- **[User Guide](docs/user_guide.md)**: Comprehensive usage guide *(coming soon)*

## 🤝 **Contributing**

This is part of the larger Hemingway data journalism project focused on financial analytics and modeling. Contributions should align with the overall project goals of transparent, explainable financial modeling.

## 📄 **License**

This project is part of the Hemingway data journalist project. See the parent directory LICENSE file for details.

---

**🎯 Next Steps**: Run the environment locally, explore the synthetic data generation, and begin experimenting with different reward functions and agent strategies! 