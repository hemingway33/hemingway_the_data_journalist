# Rejection Inference Methods for Sample Selection Bias

A comprehensive framework for comparing and selecting rejection inference methods to address sample selection bias in consumer loan decision making.

## Overview

This repository provides a modular, extensible framework for:
- **Simulating** realistic loan data with sample selection bias
- **Implementing** various rejection inference methods from conservative to advanced
- **Comparing** methods across multiple performance and interpretability metrics
- **Deciding** which method to use based on business context and constraints

## Key Features

### 🎯 **Method Coverage**
- **Conservative Methods**: Approved-only, regularized approaches
- **Propensity Weighting**: Standard and conservative variants
- **Rejection Inference**: Simple and sophisticated implementations  
- **External Data Methods**: Utilizing alternative credit data
- **Hybrid Approaches**: Combining multiple techniques

### 📊 **Comprehensive Analysis**
- **Performance Metrics**: AUC, accuracy, precision, recall
- **Interpretability Metrics**: Coefficient bias, stability analysis
- **Business Impact**: Regulatory compliance, implementation complexity
- **Visualization**: Comparative charts and trade-off analysis

### 🧭 **Decision Framework**
- **Scenario Rubrics**: Guidelines for different business contexts
- **Method Profiles**: Detailed capability and constraint mapping
- **Automated Recommendations**: Context-aware method selection

## Codebase Structure

```
projects/solving_sample_bias/
├── sample_simulation.py          # Data generation and simulation
├── rejection_inference_methods.py # Method implementations
├── comparative_analysis.py       # Analysis and evaluation tools
├── decision_framework.py         # Method selection guidelines
├── main_analysis.py              # Main orchestrator
├── config.py                     # Configuration management
├── utils.py                      # Common utility functions
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

### Module Descriptions

#### `sample_simulation.py`
- `LoanDataSimulator`: Generates realistic loan applicant features
- `SimulationScenario`: Defines complete scenarios with parameters
- Predefined scenarios for different rejection rates and data availability

#### `rejection_inference_methods.py`
- Abstract base class for all methods
- 8+ implemented methods from basic to advanced
- Factory function for easy method instantiation
- Consistent interface for training and prediction

#### `comparative_analysis.py`
- `RejectionInferenceComparator`: Main comparison engine
- Performance and interpretability metrics
- Visualization generation
- Method recommendation based on priorities

#### `decision_framework.py`
- `BusinessContext`: Structured business constraint representation
- `RejectionInferenceDecisionFramework`: Automated method selection
- Scenario rubrics and method profiles
- Example business contexts (fintech, traditional bank, etc.)

#### `main_analysis.py`
- `RejectionInferenceAnalysisRunner`: Main orchestrator
- Scenario analysis workflows
- Business context analysis
- Comprehensive comparison across scenarios

#### `config.py`
- Centralized configuration management
- Model, simulation, analysis, and business settings
- Support for configuration files and environment variables
- Global configuration instance for consistent settings

#### `utils.py`
- Common utility functions for metrics calculation
- Data validation and quality checks
- Visualization helpers and formatting functions
- File I/O utilities for saving results

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Run Complete Analysis
```python
from main_analysis import RejectionInferenceAnalysisRunner

runner = RejectionInferenceAnalysisRunner()
runner.demonstrate_decision_framework()
```

#### 2. Analyze Specific Business Context
```python
# Analyze traditional bank scenario
results = runner.run_business_context_analysis('traditional_bank')
```

#### 3. Compare Methods for Scenario
```python
# Compare methods for high rejection rate scenario
results = runner.run_scenario_analysis('high_rejection')
```

#### 4. Custom Analysis
```python
from sample_simulation import SimulationScenario
from rejection_inference_methods import create_rejection_inference_methods
from comparative_analysis import RejectionInferenceComparator

# Create custom scenario
scenario = SimulationScenario(
    name="Custom Analysis",
    rejection_rate=0.6,
    n_samples=20000,
    external_predictor_strength=0.0
)

# Generate data and compare methods
df, simulator = scenario.generate_complete_dataset()
methods = create_rejection_inference_methods()
comparator = RejectionInferenceComparator()
results = comparator.compare_methods(methods, df, simulator.get_normalized_feature_columns(), simulator)
```

## Key Scenario Rubrics

### 1. Rejection Rate Impact

| Rejection Rate | Bias Severity | Recommended Approach |
|---|---|---|
| **< 30%** | Low | Approved-only methods sufficient |
| **30-60%** | Moderate | Conservative propensity weighting |
| **> 60%** | High | Aggressive methods + external data |

### 2. Business Context Matrix

| Context | Interpretability | Performance | Regulatory | Recommended Method |
|---|---|---|---|---|
| **Traditional Bank** | High | Medium | High | Conservative Propensity |
| **Fintech Startup** | Medium | High | Medium | Simple Rejection Inference |
| **Online Lender** | Low | High | Medium | External Predictor Enhanced |
| **Credit Union** | High | Medium | High | Approved Only |

### 3. Method Selection Decision Tree

```
1. Is rejection rate < 30%?
   → YES: Use Approved Only or Regularized Approved
   → NO: Continue to question 2

2. Is interpretability critical (regulatory/compliance)?
   → YES: Use Conservative Propensity Weighting  
   → NO: Continue to question 3

3. Is external data available and high quality?
   → YES: Use Hybrid External + Propensity
   → NO: Use Simple Rejection Inference or Propensity Weighting

4. Is computational budget limited?
   → YES: Use simpler methods (Approved Only, Simple Rejection Inference)
   → NO: Use more sophisticated methods (Ensemble, Hybrid approaches)
```

## Method Profiles

### Conservative Methods

#### **Approved Only (Traditional)**
- **Best for**: Low rejection rates, high interpretability needs
- **Interpretability**: ⭐⭐⭐⭐⭐ Excellent
- **Performance**: ⭐⭐ Poor (ignores bias)
- **Complexity**: ⭐ Very Low
- **Regulatory**: ✅ Fully compliant

#### **Regularized Approved Only**
- **Best for**: Low-medium rejection rates, balanced needs
- **Interpretability**: ⭐⭐⭐⭐⭐ Excellent  
- **Performance**: ⭐⭐⭐ Fair (reduced overfitting)
- **Complexity**: ⭐ Very Low
- **Regulatory**: ✅ Fully compliant

### Propensity Methods

#### **Conservative Propensity Weighting**
- **Best for**: Balanced needs, regulatory environments
- **Interpretability**: ⭐⭐⭐⭐ Good
- **Performance**: ⭐⭐⭐ Good (addresses bias)
- **Complexity**: ⭐⭐ Low-Medium
- **Regulatory**: ✅ Generally accepted

#### **Standard Propensity Weighting**
- **Best for**: Medium-high rejection rates, performance focus
- **Interpretability**: ⭐⭐⭐ Fair
- **Performance**: ⭐⭐⭐⭐ Good
- **Complexity**: ⭐⭐ Medium
- **Regulatory**: ⚠️ May need justification

### Rejection Inference Methods

#### **Simple Rejection Inference**
- **Best for**: High rejection rates, performance priority
- **Interpretability**: ⭐⭐ Poor (artificial labels)
- **Performance**: ⭐⭐⭐⭐ Good
- **Complexity**: ⭐⭐ Medium
- **Regulatory**: ❌ High scrutiny risk

#### **External Predictor Enhanced**
- **Best for**: High rejection rates, external data available
- **Interpretability**: ⭐⭐ Poor-Fair
- **Performance**: ⭐⭐⭐⭐⭐ Excellent
- **Complexity**: ⭐⭐⭐⭐ High
- **Regulatory**: ❌ Requires careful documentation

### Advanced Methods

#### **Ensemble Averaging**
- **Best for**: Stability requirements, large datasets
- **Interpretability**: ⭐⭐⭐ Fair
- **Performance**: ⭐⭐⭐ Good
- **Complexity**: ⭐⭐⭐ Medium-High
- **Regulatory**: ✅ Generally acceptable

#### **Hybrid External + Propensity**
- **Best for**: High rejection + external data + regulatory compliance
- **Interpretability**: ⭐⭐⭐ Fair-Good
- **Performance**: ⭐⭐⭐⭐⭐ Excellent
- **Complexity**: ⭐⭐⭐⭐⭐ Very High
- **Regulatory**: ⚠️ Requires expertise

## Implementation Roadmap

### Phase 1: Baseline (Week 1-2)
- Implement Approved Only method
- Establish performance baseline  
- Measure bias impact

### Phase 2: Bias Correction (Week 3-4)
- Implement Conservative Propensity Weighting
- Compare with baseline
- Validate interpretability

### Phase 3: Optimization (Week 5-8)
- If external data available: Test External Predictor methods
- If high rejection rate: Test Simple Rejection Inference  
- If stability critical: Test Ensemble Averaging

### Phase 4: Production (Week 9+)
- Deploy best method based on validation
- Implement monitoring and model stability checks
- Plan for regular retraining and validation

## Key Insights from Analysis

### Performance vs Interpretability Trade-offs
- **Conservative methods** maintain interpretability but may underperform
- **Aggressive methods** achieve better AUC but lose coefficient interpretability  
- **Hybrid approaches** can balance both but increase complexity

### Critical Findings
- **50% rejection rate** is inflection point where bias becomes severe
- **External predictor quality** is crucial - poor data can hurt more than help
- **Coefficient stability** often more important than marginal AUC gains
- **Regulatory requirements** strongly constrain method selection

### Business Impact
- **Traditional banks**: Favor interpretable, stable methods
- **Fintechs**: Can accept complexity for performance gains
- **Regulated entities**: Must prioritize explainability and compliance
- **Resource-constrained**: Should start simple and upgrade iteratively

## Validation and Monitoring

### Model Validation
- **Coefficient comparison** against ground truth
- **Out-of-time testing** for stability
- **Stress testing** under different rejection rates
- **Interpretability assessment** with business stakeholders

### Production Monitoring  
- **Coefficient drift** detection
- **Performance degradation** alerts
- **Data quality** monitoring for external predictors
- **Regulatory compliance** documentation

## Contributing

This framework is designed to be extensible. To add new methods:

1. Inherit from `RejectionInferenceMethod` base class
2. Implement `fit()` and `predict_proba()` methods
3. Add method profile to decision framework
4. Update factory function in `rejection_inference_methods.py`

## Citation

If you use this framework in your research or business applications, please cite:

```
Rejection Inference Methods for Sample Selection Bias in Consumer Lending
https://github.com/your-repo/rejection-inference-framework
```

## License

MIT License - See LICENSE file for details.