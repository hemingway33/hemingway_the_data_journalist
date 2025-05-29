"""
FastAPI Backend for Multi-Agent Digital Twin Portfolio Management

This module provides the web API and WebSocket interface for the multi-agent
digital twin system, enabling real-time agent communication and portfolio management.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from multi_agent_twin_env import (
    MultiAgentTwinEnv, UserPreferences, AgentRole, MessageType,
    AgentMessage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Multi-Agent Digital Twin Portfolio API",
    description="API for Multi-Agent Loan Portfolio Management System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
multi_env = None
websocket_connections: Dict[str, WebSocket] = {}
agent_sessions: Dict[str, Dict] = {}

# Pydantic models for API
class UserPreferencesModel(BaseModel):
    user_id: str
    role: str
    risk_tolerance: str = "moderate"
    preferred_metrics: List[str] = ["ROE", "VaR"]
    dashboard_layout: str = "standard"
    alert_frequency: str = "daily"
    communication_style: str = "detailed"
    time_horizon: str = "quarterly"

class OptimizationRequest(BaseModel):
    user_id: str
    preferences: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None

class MarketAlertRequest(BaseModel):
    alert_type: str
    severity: str = "medium"

class SystemStatus(BaseModel):
    status: str
    total_agents: int
    user_agents: int
    portfolio_value: float
    current_roe: float
    var_95: float
    delinquency_rate: float

# Initialize environment on startup
@app.on_event("startup")
async def startup_event():
    global multi_env
    multi_env = MultiAgentTwinEnv(
        initial_portfolio_size=500,
        max_portfolio_size=5000,
        simulation_days=365
    )
    multi_env.reset()
    logger.info("Multi-Agent Twin Environment initialized")

# WebSocket Manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connection established for client: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket connection closed for client: {client_id}")
    
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: str):
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                self.disconnect(client_id)

manager = WebSocketManager()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Multi-Agent Digital Twin Portfolio API", "status": "active"}

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status and metrics"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    # Get current portfolio metrics
    observation = multi_env._get_observation()
    analytics = multi_env.get_agent_analytics()
    
    return SystemStatus(
        status="active",
        total_agents=analytics["total_agents"],
        user_agents=analytics["user_agents"],
        portfolio_value=float(observation[0] * 1e6),  # Convert back from millions
        current_roe=float(observation[3]),
        var_95=float(observation[4]),
        delinquency_rate=float(observation[2])
    )

@app.post("/api/agents/create")
async def create_user_agent(preferences: UserPreferencesModel):
    """Create a new user agent with specified preferences"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    try:
        # Convert role string to enum
        role = AgentRole(preferences.role.lower())
        
        user_prefs = UserPreferences(
            user_id=preferences.user_id,
            role=role,
            risk_tolerance=preferences.risk_tolerance,
            preferred_metrics=preferences.preferred_metrics,
            dashboard_layout=preferences.dashboard_layout,
            alert_frequency=preferences.alert_frequency,
            communication_style=preferences.communication_style,
            time_horizon=preferences.time_horizon
        )
        
        user_agent = multi_env.add_user_agent(user_prefs)
        
        # Store agent session
        agent_sessions[preferences.user_id] = {
            "agent_id": user_agent.agent_id,
            "role": role.value,
            "created_at": datetime.now().isoformat(),
            "preferences": preferences.dict()
        }
        
        # Broadcast agent creation
        await manager.broadcast(json.dumps({
            "type": "agent_created",
            "data": {
                "user_id": preferences.user_id,
                "role": role.value,
                "agent_id": user_agent.agent_id
            }
        }))
        
        return {
            "status": "success",
            "agent_id": user_agent.agent_id,
            "message": f"User agent created for {preferences.user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating user agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/agents")
async def list_agents():
    """List all active agents"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    analytics = multi_env.get_agent_analytics()
    
    return {
        "analytics": analytics,
        "agent_sessions": agent_sessions
    }

@app.post("/api/portfolio/optimize")
async def optimize_portfolio(request: OptimizationRequest):
    """Request personalized portfolio optimization"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    if request.user_id not in agent_sessions:
        raise HTTPException(status_code=404, detail="User agent not found")
    
    try:
        result = multi_env.request_personalized_optimization(
            user_id=request.user_id,
            preferences=request.preferences,
            constraints=request.constraints
        )
        
        # Broadcast optimization result
        await manager.broadcast(json.dumps({
            "type": "optimization_complete",
            "data": {
                "user_id": request.user_id,
                "result": result
            }
        }))
        
        return {
            "status": "success",
            "optimization_result": result
        }
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/alerts/simulate")
async def simulate_market_alert(request: MarketAlertRequest):
    """Simulate a market alert and broadcast to agents"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    try:
        alert_result = multi_env.simulate_market_alert(
            alert_type=request.alert_type,
            severity=request.severity
        )
        
        # Broadcast alert to all connected clients
        await manager.broadcast(json.dumps({
            "type": "market_alert",
            "data": alert_result
        }))
        
        return {
            "status": "success",
            "alert": alert_result
        }
        
    except Exception as e:
        logger.error(f"Error simulating market alert: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/portfolio/metrics")
async def get_portfolio_metrics():
    """Get current portfolio performance metrics"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    observation = multi_env._get_observation()
    performance_history = multi_env.get_performance_summary()
    
    # Get latest performance data
    latest_performance = performance_history.tail(1).to_dict('records')[0] if len(performance_history) > 0 else {}
    
    metrics = {
        "current_metrics": {
            "portfolio_value": float(observation[0] * 1e6),
            "expected_loss_rate": float(observation[1]),
            "delinquency_rate": float(observation[2]),
            "roe": float(observation[3]),
            "var_95": float(observation[4]),
            "concentration_risk": float(observation[5]),
            "base_interest_rate": float(observation[6]),
            "unemployment_rate": float(observation[7]),
            "gdp_growth": float(observation[8])
        },
        "portfolio_composition": {
            "total_loans": len(multi_env.portfolio.loans),
            "total_value": float(multi_env.portfolio.total_value),
            "total_exposure": float(multi_env.portfolio.total_exposure),
            "expected_loss": float(multi_env.portfolio.expected_loss)
        },
        "latest_performance": latest_performance,
        "market_conditions": {
            "base_interest_rate": float(multi_env.market_conditions.base_interest_rate),
            "unemployment_rate": float(multi_env.market_conditions.unemployment_rate),
            "gdp_growth": float(multi_env.market_conditions.gdp_growth),
            "inflation_rate": float(multi_env.market_conditions.inflation_rate),
            "volatility_index": float(multi_env.market_conditions.volatility_index)
        }
    }
    
    return metrics

@app.get("/api/portfolio/history")
async def get_portfolio_history():
    """Get portfolio performance history"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    performance_df = multi_env.get_performance_summary()
    
    if performance_df.empty:
        return {"history": []}
    
    # Convert DataFrame to list of dictionaries
    history = performance_df.to_dict('records')
    
    return {"history": history}

@app.post("/api/simulation/step")
async def simulation_step():
    """Execute one simulation step"""
    if not multi_env:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    try:
        # Take a random action for demonstration
        action = multi_env.action_space.sample()
        obs, reward, terminated, truncated, info = multi_env.step(action)
        
        # Broadcast simulation update
        await manager.broadcast(json.dumps({
            "type": "simulation_step",
            "data": {
                "day": multi_env.current_day,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            }
        }))
        
        return {
            "status": "success",
            "current_day": multi_env.current_day,
            "reward": float(reward),
            "terminated": terminated,
            "info": info
        }
        
    except Exception as e:
        logger.error(f"Error in simulation step: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    # Send initial system status
    try:
        status = await get_system_status()
        await manager.send_personal_message(
            json.dumps({
                "type": "system_status",
                "data": status.dict()
            }), 
            client_id
        )
    except Exception as e:
        logger.error(f"Error sending initial status: {e}")
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong"}), 
                    client_id
                )
            elif message.get("type") == "request_metrics":
                try:
                    metrics = await get_portfolio_metrics()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "metrics_update",
                            "data": metrics
                        }), 
                        client_id
                    )
                except Exception as e:
                    logger.error(f"Error sending metrics: {e}")
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 