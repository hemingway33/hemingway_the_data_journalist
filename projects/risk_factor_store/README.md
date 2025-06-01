# Credit Risk Factor Store - Barra-Style Factor Model

A comprehensive multi-factor risk model system inspired by Barra's factor modeling methodology, designed for sophisticated risk decomposition, factor exposure analysis, and portfolio risk attribution at both individual and portfolio levels.

## Features

### 🔧 Multi-Factor Risk Model Framework
- **Systematic Risk Factors**: Country, industry, style, and macroeconomic factors
- **Factor Exposures**: Standardized exposures for each entity to systematic factors
- **Factor Returns**: Time-series of factor returns with volatility modeling
- **Idiosyncratic Risk**: Entity-specific risk not explained by systematic factors
- **Factor Hierarchies**: Structured factor taxonomies (Global → Regional → Country)

### 📈 Factor Exposure Analysis
- **Fundamental Exposures**: Based on financial statement data and ratios
- **Statistical Exposures**: Derived from return time-series analysis
- **Dynamic Factor Loadings**: Time-varying exposures using rolling windows
- **Cross-Sectional Normalization**: Standardized exposures across universe
- **Missing Data Handling**: Robust estimation for incomplete data

### 🎯 Risk Decomposition & Attribution
- **Total Risk Breakdown**: Systematic vs idiosyncratic risk components
- **Factor Contribution Analysis**: Risk attribution to individual factors
- **Marginal Risk Contribution**: Impact of position changes on portfolio risk
- **Active Risk Analysis**: Tracking error decomposition against benchmarks
- **Historical Risk Attribution**: Time-series analysis of risk drivers

### 📊 Covariance Matrix Estimation
- **Factor Covariance Matrix**: Structured approach using factor models
- **Shrinkage Estimators**: Robust covariance estimation with regularization
- **Volatility Regime Modeling**: Time-varying volatility with GARCH models
- **Cross-Asset Correlations**: Consistent correlation structure across asset classes
- **Stress Testing Framework**: Scenario-based covariance adjustments

### 🔍 Data Quality Monitoring
- Automated data validation
- Quality scoring and alerts
- Data lineage tracking
- Outlier detection

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Ingestion      │    │   Processing    │
│                 │    │  Pipeline       │    │   & Storage     │
│ • FRED API      │───▶│                 │───▶│                 │
│ • Yahoo Finance │    │ • Data Cleaning │    │ • PostgreSQL    │
│ • News APIs     │    │ • Validation    │    │ • Redis Cache   │
│ • Web Scraping  │    │ • Transformation│    │ • Time Series   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Risk Engine    │    │                 │
│                 │    │                 │    │                 │
│ • REST APIs     │◀───│ • Credit Scoring│◀───│                 │
│ • WebSocket     │    │ • Portfolio     │    │                 │
│ • Documentation │    │   Analytics     │    │                 │
│ • Authentication│    │ • Stress Testing│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Technology Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy
- **Database**: PostgreSQL with TimescaleDB extension
- **Cache**: Redis for real-time data
- **ML Framework**: scikit-learn, XGBoost, PyTorch
- **Data Sources**: FRED API, Yahoo Finance, News APIs
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker + Docker Compose

## Quick Start

### Prerequisites

1. **Python 3.12+**
2. **PostgreSQL** (with TimescaleDB extension recommended)
3. **Redis** 
4. **API Keys** (optional but recommended):
   - FRED API key
   - Alpha Vantage API key
   - News API key

### Installation

1. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Set up the database**:
   ```bash
   python projects/risk_factor_store/setup.py
   ```

4. **Start the API server**:
   ```bash
   python -m projects.risk_factor_store.api.main
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Configuration

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/risk_factors
DATABASE_ECHO=false

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here

# Risk Models
DEFAULT_PD_THRESHOLD=0.05
PORTFOLIO_VAR_CONFIDENCE=0.95
LOOKBACK_DAYS=252

# Monitoring
ENABLE_PROMETHEUS=true
LOG_LEVEL=INFO
```

## API Usage

### Credit Scoring

Score an individual entity:

```python
import requests

# Score a corporate entity
response = requests.post("http://localhost:8000/api/v1/credit-scoring/score", json={
    "entity_id": "CORP_001",
    "entity_features": {
        "age_years": 15,
        "total_assets": 50000000,
        "debt_to_equity": 0.6,
        "current_ratio": 1.2,
        "roe": 0.15,
        "industry_risk_score": 0.3,
        "payment_history_score": 0.8,
        "credit_utilization": 0.4
    },
    "model_type": "xgboost"
})

score = response.json()
print(f"Probability of Default: {score['probability_of_default']:.2%}")
print(f"Risk Rating: {score['risk_rating']}")
```

### Portfolio Analytics

Generate a portfolio risk report:

```python
# Get portfolio risk report
response = requests.get("http://localhost:8000/api/v1/portfolio-analytics/PORTFOLIO_001/report")
report = response.json()

print(f"Total Exposure: ${report['basic_statistics']['total_exposure']:,.2f}")
print(f"Expected Loss: ${report['credit_metrics']['expected_loss']:,.2f}")
print(f"VaR (95%): ${report['var_metrics']['var']:,.2f}")
```

## Data Sources

### Economic Data (FRED)
- GDP Growth Rate
- Unemployment Rate  
- Inflation Rate
- Federal Funds Rate
- Treasury Yields
- Credit Spreads

### Market Data (Yahoo Finance)
- Stock Market Indices (S&P 500, NASDAQ, Dow Jones)
- Volatility Index (VIX)
- Bond Indices
- Commodity Prices
- Currency Rates

### Credit Data
- Payment histories
- Credit utilization
- Account information
- Credit inquiries

## Risk Models

### Individual Credit Scoring

The system uses multiple machine learning models:

1. **Logistic Regression**: Baseline linear model
2. **XGBoost**: Gradient boosting for non-linear relationships
3. **Random Forest**: Ensemble method for robustness

Features include:
- Entity-specific factors (financials, demographics)
- Macroeconomic conditions
- Market stress indicators
- Industry and geographic risk factors

### Portfolio Risk Analytics

- **Value at Risk (VaR)**: Parametric and Monte Carlo methods
- **Expected Shortfall**: Conditional VaR for tail risk
- **Stress Testing**: Historical and hypothetical scenarios
- **Concentration Risk**: Herfindahl index and top-N concentration

## Development

### Project Structure

```
projects/risk_factor_store/
├── api/                    # FastAPI application
│   ├── endpoints/         # API route handlers
│   ├── schemas.py         # Pydantic models
│   └── main.py           # FastAPI app
├── core/                  # Core configuration
│   ├── config.py         # Settings management
│   └── database.py       # Database connection
├── data_ingestion/        # Data collection pipeline
│   ├── base_collector.py # Base collector class
│   ├── fred_collector.py # FRED data collector
│   └── market_collector.py # Market data collector
├── models/               # Database models
│   └── risk_factors.py   # SQLAlchemy models
├── risk_engine/          # Risk calculation engines
│   ├── credit_scoring.py # Individual scoring
│   └── portfolio_analytics.py # Portfolio analysis
├── setup.py             # Database setup script
└── README.md            # This file
```

### Running Tests

```bash
pytest projects/risk_factor_store/tests/
```

### Database Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Production Considerations

1. **Database**: Use PostgreSQL with TimescaleDB for time-series optimization
2. **Caching**: Redis cluster for high availability
3. **Load Balancing**: Nginx or AWS ALB
4. **Monitoring**: Prometheus + Grafana for metrics
5. **Security**: API authentication, rate limiting, input validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the API documentation at `/redoc` 