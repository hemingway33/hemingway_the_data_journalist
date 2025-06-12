# SCF Accounting Module: BNPL vs Retailer Purchase Financing

## 迪卡侬器材租赁融资方案对比分析系统

This module provides comprehensive financial analysis and comparison of three supply chain financing scenarios:

1. **Consumer BNPL Only** - 消费者分期付款方案
2. **Retailer Purchase Financing Only** - 零售商采购融资方案  
3. **Combined Program** - 组合方案 (推荐)

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.8+ installed and the required dependencies:

```bash
uv pip install -r requirements.txt
```

### Running the Analysis

#### Test Chinese Font Support (optional)
```bash
python test_fonts.py
```

#### For Managers (简化版分析)
```bash
python demo_for_managers.py
```

#### For Detailed Analysis (完整分析)
```bash
python scf_accounting_BNPL.py
```

## 📊 Analysis Results

Based on our Decathlon equipment rental case study, the **Combined Program** shows superior performance:

| 方案 | 风险调整后ROI | 年度净利润 | 推荐指数 |
|------|--------------|-----------|----------|
| **组合方案** | **9.14%** | **¥2,485,524** | ⭐⭐⭐⭐⭐ |
| 零售商融资 | 2.93% | ¥655,644 | ⭐⭐⭐ |
| 消费者BNPL | -2.00% | ¥748,800 | ⭐ |

## 🏗️ Module Architecture

### Core Components

#### 1. `FinancingParameters` 
Configuration class for financing program parameters:
- BNPL interest rates, default rates, processing fees
- Retailer financing rates, payment terms, discounts
- Operating costs and collection costs

#### 2. `DecathlonScenario`
Business scenario parameters specific to Decathlon equipment rental:
- Equipment procurement and rental volumes
- Market adoption rates and pricing
- Cost structure and utilization rates

#### 3. `FinancingScenarioAnalyzer`
Core analysis engine that calculates:
- Financial metrics for each scenario
- Risk-adjusted returns
- Cash flow impacts
- Synergy effects for combined programs

#### 4. `DecathlonFinancingSimulator`
Simulation and reporting engine featuring:
- Monte Carlo simulations (1000 iterations)
- Executive summary generation
- Visualization dashboards
- Excel/CSV report exports

## 📈 Key Features

### Financial Modeling
- **Revenue Stream Analysis**: Interest income, processing fees, early payment discounts
- **Cost Modeling**: Default losses, operational costs, financing costs
- **ROI Calculations**: Both basic and risk-adjusted returns
- **Cash Flow Impact**: Working capital and payment timing effects

### Risk Analysis
- **Monte Carlo Simulation**: 1000 scenarios with parameter variations
- **Risk-Adjusted Returns**: Incorporating default risk and operational risk
- **Probability Analysis**: Success rates and confidence intervals

### Synergy Effects Modeling
- **Cross-selling Opportunities**: Additional revenue from program integration
- **Operational Efficiency**: Scale benefits and cost savings
- **Customer Loyalty**: Retention and lifetime value improvements

## 📋 Scenario Comparison

### Consumer BNPL Only
**优势**: 
- Higher interest income potential
- Direct consumer market access
- Scalable revenue model

**风险**:
- Higher default risk (3%)
- Consumer market volatility
- Regulatory compliance complexity

### Retailer Purchase Financing Only
**优势**:
- Lower B2B default risk
- Stable supplier relationships
- Predictable cash flows

**限制**:
- Limited growth potential
- Dependence on supplier cooperation
- Lower margins

### Combined Program (推荐)
**协同优势**:
- Revenue diversification
- Risk distribution
- Cross-selling opportunities (¥115,200 additional revenue)
- Operational efficiency gains (¥5,880 savings)
- Enhanced customer loyalty (¥960,000 additional revenue)

## 🎯 Business Case: Decathlon Equipment Rental

### Scenario Parameters
- **Monthly Equipment Purchases**: ¥500,000
- **Monthly Rental Transactions**: 2,000
- **Average Rental Value**: ¥800
- **BNPL Adoption Rate**: 30%
- **Equipment Utilization**: 75%

### Expected Outcomes
- **Annual Net Profit**: ¥2,485,524 (Combined Program)
- **Synergy Value**: ¥1,081,080
- **Risk-Adjusted ROI**: 9.14%
- **Profit Probability**: 100% (based on simulation)

## 📊 Output Files

### Generated Reports
1. **Console Output**: Real-time analysis results
2. **CSV Summary**: `decathlon_financing_analysis_summary.csv`
3. **Management Presentation**: `management_presentation.md`
4. **Visualization Dashboard**: High-quality charts with Chinese font support
   - `decathlon_financing_analysis_charts.png` - Main analysis charts
   - `default_sensitivity_analysis.png` - Default rate sensitivity analysis
   - `chinese_font_test.png` - Font compatibility test chart
5. **Default Rate Analysis**: Comprehensive default probability impact analysis
   - `default_rate_detailed_analysis.csv` - Detailed sensitivity data
   - `default_sensitivity_summary.md` - Executive summary and recommendations
   - `default_rate_analysis.py` - Specialized analysis tool

### Report Sections
- **Executive Summary**: Key findings and recommendations
- **Financial Analysis**: Detailed metrics for each scenario
- **Risk Assessment**: Monte Carlo simulation results
- **Implementation Roadmap**: Phased deployment plan

## 🔧 Customization

### Modifying Parameters
Edit the default values in the dataclass definitions:

```python
@dataclass
class FinancingParameters:
    bnpl_interest_rate: float = 0.15  # Adjust interest rate
    bnpl_default_rate: float = 0.03   # Adjust default rate
    # ... other parameters
```

### Adding New Scenarios
Create new business scenario classes:

```python
@dataclass
class CustomScenario:
    # Define your business parameters
    monthly_volume: float = 1000000
    # ... other parameters
```

### Custom Analysis
Extend the `FinancingScenarioAnalyzer` class:

```python
class CustomAnalyzer(FinancingScenarioAnalyzer):
    def calculate_custom_metrics(self):
        # Your custom calculation logic
        pass
```

## 📚 Usage Examples

### Basic Analysis
```python
from scf_accounting_BNPL import DecathlonFinancingSimulator

# Initialize simulator
simulator = DecathlonFinancingSimulator()

# Run analysis
summary = simulator.generate_executive_summary()
print(f"Best scenario: {summary['recommendation']['best_scenario']}")
```

### Custom Parameters
```python
from scf_accounting_BNPL import (
    FinancingParameters, 
    DecathlonScenario, 
    FinancingScenarioAnalyzer
)

# Custom parameters
params = FinancingParameters(
    bnpl_interest_rate=0.18,  # 18% interest rate
    bnpl_default_rate=0.025   # 2.5% default rate
)

scenario = DecathlonScenario(
    monthly_equipment_purchases=750000,  # ¥750K monthly
    consumer_bnpl_adoption=0.4           # 40% adoption
)

# Run custom analysis
analyzer = FinancingScenarioAnalyzer(params, scenario)
results = analyzer.generate_comprehensive_analysis()
```

### Default Rate Sensitivity Analysis
```python
from scf_accounting_BNPL import DecathlonFinancingSimulator

# Initialize simulator
simulator = DecathlonFinancingSimulator()

# Run default rate sensitivity analysis
sensitivity_data = simulator.analyze_default_rate_sensitivity()

# Create detailed analysis and visualization
simulator.create_default_sensitivity_analysis()
simulator.create_default_sensitivity_visualization()

# Run specialized deep analysis
from default_rate_analysis import comprehensive_default_analysis
results_df, sensitivity = comprehensive_default_analysis()
```

## 🎯 Key Insights

### Strategic Recommendations
1. **Implement Combined Program**: Highest risk-adjusted returns (9.14%)
2. **Phased Rollout**: Start with retailer financing, add BNPL
3. **Focus on Synergies**: Leverage cross-selling and efficiency gains
4. **Risk Management**: Maintain robust credit assessment and monitoring

### Critical Success Factors
- **Technology Integration**: Seamless payment processing systems
- **Risk Controls**: Multi-layered credit assessment and monitoring
- **Supplier Relationships**: Strong partnerships for early payment discounts
- **Customer Experience**: User-friendly BNPL interface and support

## 🛠️ Troubleshooting

### Common Issues

**ImportError: No module named 'openpyxl'**
```bash
uv pip install openpyxl
```

**Chinese Character Display**: 
- ✅ Now fully supported with automatic font detection
- System automatically selects appropriate Chinese fonts
- Run `python test_fonts.py` to verify font support
- Charts are saved as high-quality PNG files

**Permission Errors**: File export issues
- Ensure write permissions in the output directory
- Close any open Excel files with the same name

## 📞 Support

For questions or customization requests, please review:
1. The management presentation document
2. The demo script output
3. The detailed analysis results

## 🔮 Future Enhancements

Potential additions:
- **Real-time Dashboard**: Web-based monitoring interface
- **API Integration**: Connect with external credit scoring services
- **Advanced Analytics**: Machine learning-based risk models
- **Multi-currency Support**: International expansion capabilities

---

*Created for Decathlon Equipment Rental Financing Analysis*
*迪卡侬器材租赁融资方案分析系统* 