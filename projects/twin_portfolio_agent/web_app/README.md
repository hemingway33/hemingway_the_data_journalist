# Multi-Agent Digital Twin Portfolio Management Web Application

A modern web interface for the Multi-Agent Digital Twin Portfolio Management System, built with React and FastAPI.

## 🌟 Features

### Frontend (React)
- **Dashboard**: Real-time portfolio metrics and performance visualization
- **Agent Management**: Create and manage user agents with different roles and preferences
- **Portfolio Metrics**: Detailed portfolio analytics and optimization requests
- **Risk Management**: Risk assessment and monitoring tools
- **Agent Communication**: Real-time agent messaging and market alerts
- **System Monitoring**: System health and performance tracking

### Backend (FastAPI)
- **RESTful API**: Complete API for portfolio management operations
- **WebSocket Support**: Real-time communication between agents and frontend
- **Multi-Agent Integration**: Direct integration with the digital twin environment
- **Portfolio Optimization**: AI-driven portfolio optimization endpoints
- **Market Simulation**: Market alert simulation and stress testing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB APPLICATION                         │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React + Material-UI)                            │
│  ├── Dashboard (Charts, Metrics)                           │
│  ├── Agent Management (Create/Manage Agents)               │
│  ├── Portfolio Analytics (Optimization)                    │
│  ├── Real-time Communication (WebSocket)                   │
│  └── Responsive Design (Mobile-friendly)                   │
├─────────────────────────────────────────────────────────────┤
│  Backend (FastAPI + WebSocket)                             │
│  ├── RESTful API Endpoints                                 │
│  ├── WebSocket Manager                                     │
│  ├── Agent Communication Bridge                            │
│  └── Multi-Agent Environment Integration                   │
├─────────────────────────────────────────────────────────────┤
│  Digital Twin Core                                         │
│  ├── Multi-Agent Twin Environment                          │
│  ├── Portfolio Management Agent                            │
│  ├── User Agents (Personalization)                        │
│  └── Loan Portfolio Simulation                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 16+
- UV package manager (recommended)

### Backend Setup

```bash
# Navigate to backend directory
cd web_app/backend

# Install Python dependencies
uv pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd web_app/frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at `http://localhost:3000`

## 📋 API Documentation

### Endpoints

#### System
- `GET /api/system/status` - Get system status and metrics
- `GET /health` - Health check endpoint

#### Agents
- `POST /api/agents/create` - Create a new user agent
- `GET /api/agents` - List all active agents

#### Portfolio
- `GET /api/portfolio/metrics` - Get portfolio metrics
- `GET /api/portfolio/history` - Get portfolio performance history
- `POST /api/portfolio/optimize` - Request portfolio optimization

#### Simulation
- `POST /api/simulation/step` - Execute simulation step
- `POST /api/alerts/simulate` - Simulate market alerts

#### WebSocket
- `WS /ws/{client_id}` - Real-time communication endpoint

### Example API Usage

```javascript
// Create a new user agent
const response = await fetch('/api/agents/create', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'pm_001',
    role: 'portfolio_manager',
    risk_tolerance: 'moderate',
    preferred_metrics: ['ROE', 'VaR'],
    dashboard_layout: 'executive'
  })
});

// Request portfolio optimization
const optimization = await fetch('/api/portfolio/optimize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'pm_001',
    preferences: {
      target_roe: 0.18,
      max_risk_tolerance: 0.05
    },
    constraints: {
      regulatory_limits: true,
      liquidity_requirements: 0.15
    }
  })
});
```

## 🎨 User Interface

### Dashboard
- **Real-time Metrics**: Portfolio value, ROE, VaR, active agents
- **Performance Charts**: Historical performance trends
- **Market Conditions**: Economic indicators and market data
- **Simulation Controls**: Run simulation steps and view results

### Agent Management
- **Create Agents**: Configure user agents with specific roles and preferences
- **Agent Roles**: Portfolio Manager, Risk Manager, Credit Officer, Compliance Officer
- **Customization**: Risk tolerance, preferred metrics, dashboard layouts
- **Capabilities View**: See agent capabilities and message statistics

### Portfolio Analytics
- **Detailed Metrics**: Comprehensive portfolio composition and KPIs
- **Loan Details**: Individual loan information with risk assessment
- **Optimization Interface**: Request personalized portfolio optimization
- **Performance Tracking**: Monitor optimization results and agent recommendations

### Real-time Communication
- **Agent Messages**: Live agent communications and system events
- **Market Alerts**: Simulated market events and risk notifications
- **Alert Simulator**: Generate test alerts with configurable severity
- **Connection Status**: Real-time WebSocket connection monitoring

## 🔧 Configuration

### Environment Variables

#### Backend
```bash
# Backend configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=info
```

#### Frontend
```bash
# Frontend configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

### Customization

#### Agent Roles
Available agent roles and their specializations:
- **Portfolio Manager**: Portfolio optimization and performance tracking
- **Risk Manager**: Risk assessment and mitigation strategies
- **Credit Officer**: Credit decisions and loan approval processes
- **Compliance Officer**: Regulatory compliance and reporting

#### Risk Tolerance Levels
- **Conservative**: Lower risk, stable returns
- **Moderate**: Balanced risk-return profile
- **Aggressive**: Higher risk, potentially higher returns

#### Dashboard Layouts
- **Standard**: Basic metrics and charts
- **Executive**: High-level summary for executives
- **Detailed**: Comprehensive analytics view
- **Risk Focused**: Risk-centric metrics and alerts

## 📊 Key Features Demonstration

### Multi-Agent Communication
```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/client_123');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'agent_created':
      // Handle new agent creation
      break;
    case 'optimization_complete':
      // Handle optimization results
      break;
    case 'market_alert':
      // Handle market alerts
      break;
  }
};
```

### Portfolio Optimization
The system enables personalized portfolio optimization through user agents:
1. User creates an agent with specific preferences
2. Agent requests optimization with custom parameters
3. Portfolio Management Agent generates recommendations
4. Results are communicated back through the multi-agent system

### Real-time Updates
- Live portfolio metrics updates
- Real-time agent message broadcasting
- Instant market alert notifications
- Continuous system status monitoring

## 🛠️ Development

### Project Structure
```
web_app/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html       # Main HTML file
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── contexts/        # React contexts
│   │   ├── pages/           # Application pages
│   │   ├── services/        # API services
│   │   ├── App.js           # Main App component
│   │   └── index.js         # Application entry point
│   └── package.json         # Node.js dependencies
└── README.md                # This file
```

### Adding New Features

1. **Backend**: Add new endpoints in `main.py`
2. **Frontend**: Create components in appropriate directories
3. **API Integration**: Update services in `src/services/api.js`
4. **Real-time Features**: Extend WebSocket handling in contexts

### Testing

```bash
# Backend testing
cd backend
python -m pytest

# Frontend testing
cd frontend
npm test
```

## 🔒 Security Considerations

- **CORS Configuration**: Properly configured for development and production
- **WebSocket Security**: Client ID validation and connection management
- **Input Validation**: Pydantic models for API request validation
- **Error Handling**: Comprehensive error handling and logging

## 🚀 Deployment

### Production Deployment

#### Backend (Docker)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend (Docker)
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html
```

### Environment Setup
- Configure production environment variables
- Set up reverse proxy (nginx)
- Enable HTTPS for production
- Configure monitoring and logging

## 📈 Performance

- **WebSocket Optimization**: Efficient message broadcasting
- **React Optimization**: Memoization and lazy loading
- **API Caching**: Intelligent data caching strategies
- **Chart Performance**: Optimized chart rendering for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of the Digital Twin Portfolio Management System. See the main project README for license information.

---

**Built with ❤️ using React, FastAPI, and Material-UI** 