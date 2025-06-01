# Credit Policy Agent

An intelligent system that autonomously manages, reviews, and optimizes credit policies for lending institutions. The agent continuously monitors performance metrics, evaluates data sources, and makes informed decisions to balance compliance, risk management, and profitability.

## 🎯 Key Features

### 🔧 Policy Rule Engine
- **Dynamic Rule Management**: Create, update, and version control credit policy rules
- **Real-time Evaluation**: Instant loan application assessment against current policies
- **A/B Testing**: Test policy changes with controlled traffic splits
- **Rule Validation**: Comprehensive validation before applying policy changes

### 📊 Performance Optimization
- **Multi-objective Optimization**: Balance compliance, risk, and profit objectives
- **Continuous Monitoring**: Track key performance indicators in real-time
- **Automated Alerts**: Get notified when performance degrades
- **Impact Simulation**: Preview policy changes before implementation

### 🤖 Model Management
- **Performance Monitoring**: Track credit score model accuracy and drift
- **Automated Retraining**: Trigger model updates when performance drops
- **Model Comparison**: Evaluate multiple models and select the best performer
- **Explainable AI**: Understand model decisions and feature importance

### 📈 Data Source Evaluation
- **Alternative Data Integration**: Incorporate non-traditional data sources
- **Quality Assessment**: Monitor data freshness and completeness
- **Cost-Benefit Analysis**: Evaluate ROI of different data providers
- **Automated Onboarding**: Seamlessly integrate new data sources

### 💬 Client Interaction Intelligence
- **Smart Interview Questions**: Generate targeted questions based on risk profile
- **Information Requests**: Dynamically request additional client information
- **Communication Workflow**: Automate client interaction processes
- **Risk-based Collection**: Prioritize information gathering by risk level

### 🔐 **Secure Explanations & Model Protection**
- **Customer-Safe Explanations**: Provide clear reasoning without exposing model internals
- **Multi-Level Access**: Different explanation depths for customers, staff, admins, and auditors
- **FCRA Compliance**: Adverse action notices that meet regulatory requirements
- **Security Validation**: Automatic detection and prevention of sensitive data leakage
- **Model Weight Protection**: Safeguards ensure model parameters never reach customers

## 🏗️ Architecture

```
Credit Policy Agent
├── Core Agent (Orchestration)
├── Policy Engine (Rule Management)
├── Optimization Engine (Performance Tuning)
├── Model Manager (ML Model Lifecycle)
├── Data Source Manager (External Data)
├── Client Interaction Manager (Communication)
├── Monitoring & Metrics (Performance Tracking)
└── REST API (External Interface)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- PostgreSQL 14+ (optional, defaults to SQLite)
- Redis 6+ (optional, for caching)

### Installation

1. **Clone and navigate to the project:**
```bash
cd projects/credit_policy_agent
```

2. **Install dependencies using uv:**
```bash
uv pip install -r requirements.txt
```

3. **Run the test example:**
```bash
python test_example.py
```

4. **Start the API server:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation
Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Agent Status**: http://localhost:8000/api/v1/status

## 📋 Usage Examples

### Evaluate a Loan Application

```python
import asyncio
from src.core.agent import CreditPolicyAgent, LoanApplication

async def evaluate_loan():
    agent = CreditPolicyAgent()
    await agent.start()
    
    application = LoanApplication(
        application_id="APP001",
        applicant_id="USER001",
        loan_amount=25000.0,
        loan_purpose="home_improvement",
        credit_score=720,
        income=65000.0,
        employment_status="employed",
        debt_to_income_ratio=0.35
    )
    
    decision = await agent.evaluate_application(application)
    print(f"Decision: {'Approved' if decision.approved else 'Rejected'}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Interest Rate: {decision.interest_rate:.2%}")
    
    await agent.stop()

asyncio.run(evaluate_loan())
```

### Update Policy Rules

```python
async def update_policy():
    agent = CreditPolicyAgent()
    await agent.start()
    
    new_rules = {
        "min_credit_score_premium": {
            "rule_type": "credit_score",
            "field": "application.credit_score",
            "operator": "gte",
            "value": 750,
            "weight": 1.5,
            "description": "Premium tier minimum credit score"
        }
    }
    
    success = await agent.update_policy_rules(new_rules)
    print(f"Policy update: {'Success' if success else 'Failed'}")
    
    await agent.stop()
```

### REST API Usage

```bash
# Evaluate loan application
curl -X POST "http://localhost:8000/api/v1/applications/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "APP001",
    "applicant_id": "USER001", 
    "loan_amount": 25000,
    "loan_purpose": "home_improvement",
    "credit_score": 720,
    "income": 65000,
    "employment_status": "employed",
    "debt_to_income_ratio": 0.35
  }'

# Get current policy
curl "http://localhost:8000/api/v1/policies/current"

# Update policy rules
curl -X PUT "http://localhost:8000/api/v1/policies/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "rules": {
      "new_rule": {
        "rule_type": "income",
        "field": "application.income", 
        "operator": "gte",
        "value": 30000,
        "weight": 1.0,
        "description": "Minimum income requirement"
      }
    }
  }'

# Get customer-safe explanation (never exposes model internals)
curl "http://localhost:8000/api/v1/applications/APP001/explanation"

# Get internal explanation for staff (with operational details)
curl "http://localhost:8000/api/v1/applications/APP001/explanation/internal?user_role=internal_staff"

# Validate explanation security
curl -X POST "http://localhost:8000/api/v1/explanations/validate-security" \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "declined",
    "message": "Credit score below requirements"
  }'
```

## 🔐 Secure Explanations

The system includes sophisticated explainable AI features that provide transparency while protecting sensitive model information:

### Customer Explanations
Customer-facing explanations **NEVER** expose model internals:

```json
{
  "decision": "declined",
  "message": "We are unable to approve your loan application at this time.",
  "primary_factors": [
    "Credit score does not meet our minimum requirements",
    "Debt-to-income ratio exceeds our guidelines"
  ],
  "improvement_suggestions": [
    "Improve credit score by paying bills on time",
    "Pay down existing debts to reduce debt-to-income ratio"
  ],
  "your_rights": [
    "You have the right to obtain a free copy of your credit report"
  ]
}
```

### Multi-Level Access Control
- **👤 Customer**: Safe explanations with actionable advice, no sensitive data
- **👨‍💼 Internal Staff**: Operational details but still protects core model secrets
- **👨‍💻 Administrator**: System management details for troubleshooting
- **📋 Auditor**: Full compliance information for regulatory purposes

### Protected Information
The system automatically prevents exposure of:
- ❌ Model weights and coefficients  
- ❌ Internal scoring thresholds
- ❌ Algorithm implementation details
- ❌ Training data characteristics
- ❌ Feature importance values
- ❌ Raw risk scores and parameters

### Security Features
- 🔒 **Automatic Sanitization**: Removes sensitive data based on user role
- 🛡️ **Security Validation**: Scans explanations for potential data leakage
- 📋 **FCRA Compliance**: Generates adverse action notices per regulations
- 🚫 **Model Protection**: Multiple layers prevent exposure of proprietary algorithms

## ⚙️ Configuration

The system uses environment variables for configuration. Create a `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Database
DATABASE_URL=sqlite:///./credit_policy.db

# Optimization
OPTIMIZATION_INTERVAL_HOURS=6
COMPLIANCE_WEIGHT=0.4
RISK_WEIGHT=0.4
PROFIT_WEIGHT=0.2

# Model Management
AUTO_RETRAIN_ENABLED=true
MODEL_VALIDATION_THRESHOLD=0.8

# Monitoring
LOG_LEVEL=INFO
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_example.py
```

This will:
- ✅ Start the agent
- 📋 Evaluate sample loan applications
- 🔧 Test policy rule updates
- 📊 Display agent status and metrics
- ✅ Clean shutdown

## 📊 Default Policy Rules

The system comes with sensible default credit policy rules:

| Rule | Description | Threshold |
|------|-------------|-----------|
| **Minimum Credit Score** | FICO score requirement | ≥ 580 |
| **Maximum Debt-to-Income** | DTI ratio limit | ≤ 43% |
| **Minimum Income** | Annual income requirement | ≥ $25,000 |
| **Employment Status** | Valid employment types | Employed, Self-employed |
| **Maximum Risk Score** | Model risk score limit | ≤ 80% |

## 🔮 Future Enhancements

### Advanced AI Features
- **Reinforcement Learning**: Dynamic policy learning from outcomes
- **Explainable AI**: Enhanced decision transparency
- **Federated Learning**: Privacy-preserving multi-institutional learning
- **Graph Neural Networks**: Relationship-based risk assessment

### Integration Opportunities
- **Open Banking APIs**: Enhanced data access
- **Blockchain Integration**: Immutable audit trails
- **IoT Data Sources**: Alternative risk indicators
- **Social Media Analysis**: Behavioral risk factors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions, issues, or contributions:
- 📧 Email: support@creditpolicyagent.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/credit-policy-agent/issues)
- 📖 Documentation: [Full Documentation](https://docs.creditpolicyagent.com)

---

**Built with ❤️ for the future of intelligent lending** 