# Getting Started with Multi-Agent Digital Twin Portfolio Management

Welcome to the Multi-Agent Digital Twin Portfolio Management Web Application! This guide will help you get up and running quickly.

## 🚀 Quick Start (Recommended)

The fastest way to get started is using our master launch script:

```bash
# Navigate to the web app directory
cd projects/twin_portfolio_agent/web_app

# Launch the complete system
python launch.py
```

This will automatically:
- ✅ Check dependencies
- 📦 Install required packages  
- 🔥 Start the backend API server
- ⚛️ Start the React frontend
- 🌐 Open your web browser
- 📊 Make everything ready to use

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.8+** (3.12+ recommended)
- **Node.js 16+** (for React frontend)
- **npm** (comes with Node.js)

### Installing Prerequisites

#### On macOS (using Homebrew):
```bash
brew install python3 node
```

#### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3 python3-pip nodejs npm
```

#### On Windows:
1. Install Python from [python.org](https://python.org)
2. Install Node.js from [nodejs.org](https://nodejs.org)

## 🛠️ Manual Setup (Alternative)

If you prefer manual setup or the quick start doesn't work:

### 1. Run Setup Script
```bash
./setup.sh
```

### 2. Start Services Separately

#### Terminal 1 - Backend:
```bash
./start_backend.sh
```

#### Terminal 2 - Frontend:
```bash
./start_frontend.sh
```

### 3. Or Start Both Together:
```bash
./start_all.sh
```

## 🌐 Accessing the Application

Once started, access the application at:

- **Frontend Web App**: http://localhost:3000
- **Backend API**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🎯 First Steps in the Application

### 1. Dashboard Overview
- View real-time portfolio metrics
- Monitor active agents
- See performance charts
- Check market conditions

### 2. Create Your First Agent
1. Navigate to **Agent Management**
2. Click **Create Agent**
3. Choose a role:
   - **Portfolio Manager**: Optimization and performance tracking
   - **Risk Manager**: Risk assessment and mitigation
   - **Credit Officer**: Credit decisions and loan management
   - **Compliance Officer**: Regulatory compliance
4. Configure preferences (risk tolerance, metrics, etc.)
5. Click **Create Agent**

### 3. Request Portfolio Optimization
1. Go to **Portfolio Metrics**
2. Click **Request Optimization**
3. Select your agent
4. Set optimization parameters
5. Submit request and view results

### 4. Monitor Real-time Communications
1. Visit **Agent Communication**
2. See live agent messages
3. Simulate market alerts
4. Monitor system responses

## 🎮 Demo Mode

Try our automated demonstration:

```bash
# Automated demo (runs by itself)
python demo.py

# Interactive demo (step-by-step)
python demo.py --mode interactive
```

The demo will:
- Create sample agents
- Run portfolio optimizations  
- Simulate market scenarios
- Generate performance reports

## 🧪 Testing the System

Run our comprehensive test suite:

```bash
python test_webapp.py
```

This tests:
- ✅ All API endpoints
- 🔌 WebSocket connections
- ⚡ Performance metrics
- 🛡️ Error handling
- 📊 System integration

## 🐳 Docker Deployment (Optional)

For production-like deployment:

```bash
# Build and start with Docker
./start_docker.sh

# Or manually
docker-compose up --build
```

## 📱 Mobile Support

The application is responsive and works on:
- 📱 Mobile phones
- 📲 Tablets  
- 💻 Desktop computers
- 🖥️ Large displays

## 🔧 Configuration

### Environment Variables

#### Backend (.env):
```bash
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=info
```

#### Frontend (.env):
```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

### Agent Configuration
Customize agent behavior by setting:
- **Risk Tolerance**: Conservative, Moderate, Aggressive
- **Preferred Metrics**: ROE, VaR, Sharpe ratio, etc.
- **Dashboard Layout**: Standard, Executive, Detailed, Risk-focused
- **Alert Frequency**: Real-time, Hourly, Daily, Weekly

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Kill processes on ports 3000 and 8000
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

#### Dependencies Not Installing
```bash
# Clear npm cache
npm cache clean --force

# Recreate Python virtual environment
rm -rf backend/.venv
python3 -m venv backend/.venv
```

#### WebSocket Connection Issues
- Check if backend is running on port 8000
- Verify firewall settings
- Try refreshing the browser

#### Frontend Not Loading
- Wait 30-60 seconds for initial build
- Check for JavaScript errors in browser console
- Ensure npm dependencies installed correctly

### Getting Help

1. **Check Logs**: Look at terminal output for error messages
2. **API Status**: Visit http://localhost:8000/health
3. **Browser Console**: Press F12 to see JavaScript errors
4. **Test Suite**: Run `python test_webapp.py` to diagnose issues

## 📊 System Architecture

```
┌─────────────────────────────────────┐
│           Frontend (React)          │
│  - Dashboard, Agent Management      │
│  - Real-time Updates (WebSocket)    │
│  - Material-UI Components           │
└─────────────────┬───────────────────┘
                  │ HTTP/WebSocket
┌─────────────────▼───────────────────┐
│           Backend (FastAPI)         │
│  - REST API Endpoints              │
│  - WebSocket Manager               │
│  - Multi-Agent Integration         │
└─────────────────┬───────────────────┘
                  │ Python Integration
┌─────────────────▼───────────────────┐
│       Multi-Agent Twin Core        │
│  - Portfolio Management Agent      │
│  - User Agents (Personalization)   │
│  - Loan Portfolio Simulation       │
└─────────────────────────────────────┘
```

## 🎯 Key Features

### Real-time Portfolio Management
- Live portfolio metrics updates
- Interactive performance charts
- Market condition monitoring
- Risk assessment dashboards

### Multi-Agent Personalization  
- Create agents with specific roles
- Personalized optimization requests
- Customizable risk preferences
- Agent-specific dashboards

### Advanced Communication
- Real-time agent messaging
- Market alert simulation
- WebSocket-based updates
- System-wide notifications

### Professional UI/UX
- Modern Material-UI design
- Responsive mobile layout
- Interactive data visualization
- Intuitive navigation

## 📈 Performance Tips

### For Best Performance:
1. **Use Chrome or Firefox** for optimal WebSocket support
2. **Close unused browser tabs** to free memory
3. **Keep terminal open** to monitor system logs
4. **Use SSD storage** for faster Node.js builds
5. **Ensure stable internet** for real-time features

## 🔒 Security Notes

### Development Mode:
- CORS is configured for localhost only
- WebSocket connections are unencrypted
- No authentication required

### Production Deployment:
- Configure HTTPS/WSS encryption
- Implement proper authentication
- Set production CORS origins
- Use environment-specific settings

## 📚 Next Steps

1. **Explore the Interface**: Try all menu options and features
2. **Create Multiple Agents**: Test different roles and preferences  
3. **Run Optimizations**: See how agents provide different recommendations
4. **Monitor Performance**: Watch real-time metrics and alerts
5. **Read the Code**: Explore the source code for customization
6. **Extend Features**: Add new agent types or optimization strategies

## 💡 Tips for Success

- **Start Simple**: Create one agent first, then add more
- **Monitor Logs**: Keep terminal open to see system activity
- **Use Demo Mode**: Run demos to understand capabilities
- **Test Frequently**: Use the test suite to verify functionality
- **Experiment**: Try different agent configurations and see results

## 🤝 Contributing

Want to improve the system?
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_webapp.py`
5. Submit a pull request

---

**🎉 Congratulations!** You're now ready to explore the Multi-Agent Digital Twin Portfolio Management System. Start by creating your first agent and see the power of AI-driven portfolio optimization in action!

For detailed technical documentation, see [README.md](README.md). 