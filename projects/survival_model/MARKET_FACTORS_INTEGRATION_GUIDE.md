# Market Factors Integration Guide for SME Lending

## Overview

The Market Factors Risk Model provides a comprehensive framework for capturing systematic risks that affect SME loan portfolios beyond individual borrower characteristics. This guide demonstrates how to integrate market factors with your existing survival model and decision engine frameworks.

## Key Features

### 1. Comprehensive Risk Factor Coverage
- **24 Risk Factors** across 4 categories
- **Macroeconomic Factors (8)**: GDP growth, unemployment, inflation, Fed funds rate, confidence indices, PMI
- **Market Factors (6)**: VIX volatility, credit spreads, lending standards, bond yields, real estate, lending volume
- **Industry Factors (5)**: Retail sales, construction spending, technology spending, energy prices, supply chain stress
- **Regional Factors (5)**: Regional unemployment, GDP growth, real estate, business formation, bank health

### 2. Data Management Capabilities
- **Automatic Updates**: From FRED API, Yahoo Finance, and other public sources
- **Manual Updates**: For proprietary or specialized data sources
- **Historical Storage**: SQLite database for trend analysis
- **Real-time Integration**: Daily, weekly, monthly, and quarterly update frequencies

### 3. Risk Assessment Framework
- **Industry-Specific Scoring**: Tailored risk weights for different business sectors
- **Portfolio Adjustments**: Dynamic multipliers for default probability adjustments
- **Comprehensive Reporting**: Detailed risk breakdowns and recommendations

## Integration with Survival Model Framework

### Basic Integration Example

```python
from market_factors import MarketFactorsContainer
from enhanced_decision_engine import JointOptimizationEngine

# Initialize market factors
market_factors = MarketFactorsContainer()

# Update current market conditions (manual or automatic)
market_factors.update_factor_manual('unemployment_rate', 3.7)
market_factors.update_factor_manual('fed_funds_rate', 5.25)
market_factors.update_factor_manual('inflation_rate', 3.2)
market_factors.update_factor_manual('vix_volatility', 18.5)

# Get risk adjustment for specific industry
industry = 'technology'
risk_multiplier = market_factors.get_portfolio_risk_adjustment(industry=industry)

# Initialize decision engine with market factor adjustment
decision_engine = JointOptimizationEngine()

# Apply market factor adjustment to survival model predictions
def adjust_survival_prediction(base_prediction, market_multiplier):
    """
    Adjust survival model prediction based on market factors.
    
    Parameters:
    -----------
    base_prediction : float
        Base default probability from survival model
    market_multiplier : float
        Market factor adjustment multiplier
        
    Returns:
    --------
    float : Adjusted default probability
    """
    adjusted_prediction = base_prediction * market_multiplier
    return min(1.0, max(0.0, adjusted_prediction))  # Bound between 0 and 1

# Example usage in loan evaluation
borrower_data = {
    'loan_amount': 50000,
    'annual_income': 75000,
    'credit_score': 720,
    'debt_to_income': 0.35,
    'years_in_business': 5,
    'industry': 'technology',
    'collateral_value': 60000,
    'payment_history_score': 0.85
}

# Get base survival prediction
base_default_prob = 0.08  # From your survival model

# Apply market factor adjustment
adjusted_default_prob = adjust_survival_prediction(
    base_default_prob, 
    risk_multiplier
)

print(f"Base Default Probability: {base_default_prob:.3f}")
print(f"Market Risk Multiplier: {risk_multiplier:.3f}")
print(f"Adjusted Default Probability: {adjusted_default_prob:.3f}")
```

### Advanced Integration with Decision Engine

```python
class MarketAwareDecisionEngine(JointOptimizationEngine):
    """
    Enhanced decision engine with market factor integration.
    """
    
    def __init__(self, market_factors_container=None):
        super().__init__()
        self.market_factors = market_factors_container or MarketFactorsContainer()
        
    def evaluate_loan_with_market_factors(self, borrower_data, loan_terms):
        """
        Evaluate loan considering both borrower-specific and market factors.
        """
        # Get base evaluation from parent class
        base_evaluation = self.evaluate_4variable_combination(
            borrower_data, 
            loan_terms['amount_strategy'],
            loan_terms['pricing_strategy'], 
            loan_terms['payment_pattern'],
            loan_terms['term_months']
        )
        
        if not base_evaluation['approved']:
            return base_evaluation
            
        # Apply market factor adjustments
        industry = borrower_data.get('industry', 'services')
        region = borrower_data.get('region', None)
        
        market_multiplier = self.market_factors.get_portfolio_risk_adjustment(
            industry=industry, 
            region=region
        )
        
        # Adjust default probability
        adjusted_default_prob = min(1.0, 
            base_evaluation['default_probability'] * market_multiplier
        )
        
        # Recalculate metrics with adjusted default probability
        adjusted_evaluation = self._recalculate_with_market_adjustment(
            base_evaluation, 
            adjusted_default_prob,
            market_multiplier
        )
        
        return adjusted_evaluation
        
    def _recalculate_with_market_adjustment(self, base_eval, adj_default_prob, multiplier):
        """Recalculate loan metrics with market-adjusted default probability."""
        
        # Adjust expected return
        survival_prob = 1 - adj_default_prob
        adj_expected_return = base_eval['expected_return'] * survival_prob
        
        # Adjust profit calculations
        adj_expected_profit = base_eval['expected_profit'] * survival_prob
        
        # Create adjusted evaluation
        adjusted_eval = base_eval.copy()
        adjusted_eval.update({
            'default_probability': adj_default_prob,
            'expected_return': adj_expected_return,
            'expected_profit': adj_expected_profit,
            'market_risk_multiplier': multiplier,
            'market_adjusted': True
        })
        
        # Re-check approval criteria with adjusted metrics
        adjusted_eval['approved'] = self._check_adjusted_approval_criteria(adjusted_eval)
        
        return adjusted_eval
        
    def _check_adjusted_approval_criteria(self, evaluation):
        """Check approval criteria with market-adjusted metrics."""
        criteria = {
            'min_expected_return': evaluation['expected_return'] >= 0.04,
            'max_default_probability': evaluation['default_probability'] <= 0.30,
            'min_expected_profit': evaluation['expected_profit'] >= 300,
            'market_risk_acceptable': evaluation['market_risk_multiplier'] <= 1.5
        }
        
        return all(criteria.values())

# Usage example
market_aware_engine = MarketAwareDecisionEngine()

# Update market conditions
market_aware_engine.market_factors.update_factor_manual('unemployment_rate', 4.2)
market_aware_engine.market_factors.update_factor_manual('credit_spreads', 600)

# Evaluate loan with market factors
borrower = {
    'loan_amount': 75000,
    'annual_income': 90000,
    'credit_score': 740,
    'industry': 'retail',
    'region': 'northeast'
}

loan_terms = {
    'amount_strategy': 'standard',
    'pricing_strategy': 'premium',
    'payment_pattern': 'installment',
    'term_months': 36
}

evaluation = market_aware_engine.evaluate_loan_with_market_factors(
    borrower, 
    loan_terms
)

print(f"Market-Adjusted Evaluation:")
print(f"  Approved: {evaluation['approved']}")
print(f"  Default Probability: {evaluation['default_probability']:.3f}")
print(f"  Expected Return: {evaluation['expected_return']:.3f}")
print(f"  Market Risk Multiplier: {evaluation['market_risk_multiplier']:.3f}")
```

## Periodic Market Factor Updates

### Automated Update Schedule

```python
import schedule
import time
from datetime import datetime

def setup_market_factor_updates(market_factors, fred_api_key=None):
    """Setup automated market factor update schedule."""
    
    # Daily updates (market open)
    schedule.every().day.at("09:30").do(
        update_daily_factors, 
        market_factors, 
        fred_api_key
    )
    
    # Weekly updates (Monday morning)
    schedule.every().monday.at("08:00").do(
        update_weekly_factors,
        market_factors,
        fred_api_key
    )
    
    # Monthly updates (first business day)
    schedule.every().month.do(
        update_monthly_factors,
        market_factors,
        fred_api_key
    )

def update_daily_factors(market_factors, api_key):
    """Update daily market factors."""
    daily_factors = [
        'vix_volatility',
        'fed_funds_rate', 
        'credit_spreads',
        'corporate_bond_yield'
    ]
    
    for factor in daily_factors:
        try:
            market_factors.update_factor_automatic(factor, api_key)
            print(f"Updated {factor} at {datetime.now()}")
        except Exception as e:
            print(f"Failed to update {factor}: {e}")

def update_weekly_factors(market_factors, api_key):
    """Update weekly market factors."""
    # Add weekly-specific updates
    print(f"Weekly market factor update completed at {datetime.now()}")

def update_monthly_factors(market_factors, api_key):
    """Update monthly market factors."""
    monthly_factors = [
        'unemployment_rate',
        'inflation_rate',
        'consumer_confidence',
        'retail_sales_growth',
        'construction_spending'
    ]
    
    for factor in monthly_factors:
        try:
            market_factors.update_factor_automatic(factor, api_key)
            print(f"Updated {factor} at {datetime.now()}")
        except Exception as e:
            print(f"Failed to update {factor}: {e}")

# Start the scheduler
def run_market_factor_scheduler(market_factors, fred_api_key):
    """Run the market factor update scheduler."""
    setup_market_factor_updates(market_factors, fred_api_key)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
```

## Portfolio-Level Risk Management

### Dynamic Portfolio Adjustments

```python
class PortfolioRiskManager:
    """
    Portfolio-level risk management with market factor integration.
    """
    
    def __init__(self, market_factors_container):
        self.market_factors = market_factors_container
        self.portfolio_segments = {
            'retail': {'weight': 0.25, 'max_exposure': 0.35},
            'construction': {'weight': 0.20, 'max_exposure': 0.30},
            'technology': {'weight': 0.30, 'max_exposure': 0.40},
            'manufacturing': {'weight': 0.15, 'max_exposure': 0.25},
            'services': {'weight': 0.10, 'max_exposure': 0.20}
        }
        
    def calculate_portfolio_risk_score(self):
        """Calculate overall portfolio risk score."""
        portfolio_risk = 0.0
        
        for industry, config in self.portfolio_segments.items():
            industry_risk = self.market_factors.calculate_risk_score(industry=industry)
            weighted_risk = industry_risk['risk_score'] * config['weight']
            portfolio_risk += weighted_risk
            
        return portfolio_risk
        
    def get_industry_recommendations(self):
        """Get lending recommendations by industry."""
        recommendations = {}
        
        for industry in self.portfolio_segments.keys():
            risk_analysis = self.market_factors.calculate_risk_score(industry=industry)
            multiplier = self.market_factors.get_portfolio_risk_adjustment(industry=industry)
            
            if multiplier > 1.2:
                recommendation = "RESTRICT"
                action = "Tighten underwriting, reduce exposure"
            elif multiplier > 1.1:
                recommendation = "CAUTION"
                action = "Increase pricing, monitor closely"
            elif multiplier < 0.9:
                recommendation = "EXPAND"
                action = "Consider growth opportunities"
            else:
                recommendation = "MAINTAIN"
                action = "Continue current strategy"
                
            recommendations[industry] = {
                'risk_score': risk_analysis['risk_score'],
                'risk_multiplier': multiplier,
                'recommendation': recommendation,
                'action': action,
                'top_risk_factors': self._get_top_risk_factors(risk_analysis, 3)
            }
            
        return recommendations
        
    def _get_top_risk_factors(self, risk_analysis, top_n=3):
        """Get top risk factors for an industry."""
        factor_contribs = risk_analysis['factor_contributions']
        sorted_factors = sorted(
            factor_contribs.items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )
        
        return [
            {
                'factor': factor_name,
                'description': self.market_factors.factors[factor_name].description,
                'value': contrib['value'],
                'risk_contribution': contrib['weighted_score']
            }
            for factor_name, contrib in sorted_factors[:top_n]
        ]

# Usage example
portfolio_manager = PortfolioRiskManager(market_factors)

# Get current portfolio risk assessment
portfolio_risk = portfolio_manager.calculate_portfolio_risk_score()
industry_recommendations = portfolio_manager.get_industry_recommendations()

print(f"Overall Portfolio Risk Score: {portfolio_risk:.3f}")
print("\nIndustry Recommendations:")
for industry, rec in industry_recommendations.items():
    print(f"\n{industry.upper()}:")
    print(f"  Risk Score: {rec['risk_score']:.3f}")
    print(f"  Recommendation: {rec['recommendation']}")
    print(f"  Action: {rec['action']}")
    print(f"  Top Risk Factors:")
    for factor in rec['top_risk_factors']:
        print(f"    • {factor['description']}: {factor['value']:.2f}")
```

## API Integration Examples

### FRED API Integration

```python
# To use FRED API for automatic updates:
FRED_API_KEY = "your_fred_api_key_here"

# Update all FRED factors automatically
market_factors.update_all_factors(api_key=FRED_API_KEY)

# Update specific factor
market_factors.update_factor_automatic('unemployment_rate', FRED_API_KEY)
```

### Yahoo Finance Integration

```python
# Yahoo Finance updates (no API key required)
market_factors.update_factor_automatic('vix_volatility')

# Manual VIX update
import yfinance as yf
vix = yf.Ticker("^VIX")
current_vix = vix.history(period="1d")['Close'].iloc[-1]
market_factors.update_factor_manual('vix_volatility', current_vix)
```

## Best Practices

### 1. Update Frequency Management
- **Daily**: VIX, Fed funds rate, credit spreads, bond yields
- **Weekly**: Market sentiment indicators
- **Monthly**: Economic indicators, employment data, industry metrics
- **Quarterly**: GDP, business confidence, lending standards

### 2. Risk Threshold Management
```python
# Define risk thresholds for different market conditions
RISK_THRESHOLDS = {
    'low_risk': {'max_score': 0.3, 'multiplier_range': (0.75, 1.0)},
    'moderate_risk': {'max_score': 0.7, 'multiplier_range': (1.0, 1.2)},
    'high_risk': {'max_score': 1.0, 'multiplier_range': (1.2, 1.5)}
}

def get_risk_category(risk_score):
    """Categorize risk level based on score."""
    if risk_score <= RISK_THRESHOLDS['low_risk']['max_score']:
        return 'low_risk'
    elif risk_score <= RISK_THRESHOLDS['moderate_risk']['max_score']:
        return 'moderate_risk'
    else:
        return 'high_risk'
```

### 3. Historical Analysis
```python
# Analyze historical risk patterns
def analyze_risk_trends(market_factors, factor_name, months=12):
    """Analyze risk factor trends over time."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)
    
    history = market_factors.get_factor_history(
        factor_name,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if not history.empty:
        trend_analysis = {
            'current_value': history['value'].iloc[-1],
            'average': history['value'].mean(),
            'volatility': history['value'].std(),
            'trend': 'increasing' if history['value'].iloc[-1] > history['value'].mean() else 'decreasing',
            'percentile_rank': (history['value'] <= history['value'].iloc[-1]).mean()
        }
        return trend_analysis
    
    return None

# Example usage
unemployment_trend = analyze_risk_trends(market_factors, 'unemployment_rate', 24)
print(f"Unemployment Trend Analysis: {unemployment_trend}")
```

## Conclusion

The Market Factors Risk Model provides a robust framework for incorporating systematic risks into your SME lending decisions. By integrating macroeconomic, market, industry, and regional factors, you can:

1. **Enhance Risk Assessment**: More accurate default probability predictions
2. **Dynamic Portfolio Management**: Real-time adjustments based on market conditions
3. **Industry-Specific Strategies**: Tailored approaches for different business sectors
4. **Automated Risk Monitoring**: Continuous updates from public data sources
5. **Comprehensive Reporting**: Detailed risk breakdowns and actionable insights

This integration transforms your lending platform from a static model to a dynamic, market-aware decision engine that adapts to changing economic conditions and provides superior risk management capabilities. 