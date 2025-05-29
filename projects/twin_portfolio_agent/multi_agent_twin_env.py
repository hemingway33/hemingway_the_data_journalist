"""
Multi-Agent Digital Twin Environment for Loan Portfolio Management

This module extends the core digital twin environment with multi-agent capabilities,
enabling user agents to communicate with the portfolio management agent for
highly customized solutions and services.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import json
from abc import ABC, abstractmethod

# Import base components from the original twin_env
from twin_env import (
    LoanPortfolioTwinEnv, LoanStatus, LoanType, Loan, Portfolio, 
    MarketConditions, BusinessPolicies
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    INSIGHT = "insight"
    ALERT = "alert"


class AgentRole(Enum):
    """Agent role types"""
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"
    CREDIT_OFFICER = "credit_officer"
    COMPLIANCE_OFFICER = "compliance_officer"
    PORTFOLIO_AGENT = "portfolio_agent"


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    requires_response: bool = False


@dataclass
class UserPreferences:
    """User preferences and customization settings"""
    user_id: str
    role: AgentRole
    risk_tolerance: str = "moderate"
    preferred_metrics: List[str] = field(default_factory=lambda: ["ROE", "VaR"])
    dashboard_layout: str = "standard"
    alert_frequency: str = "daily"
    communication_style: str = "detailed"
    custom_thresholds: Dict[str, float] = field(default_factory=dict)
    time_horizon: str = "quarterly"
    sector_preferences: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.message_queue = []
        self.capabilities = []
        self.is_active = True
        
    @abstractmethod
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and optionally return response"""
        pass
    
    def send_message(self, broker, message: AgentMessage):
        """Send message through the message broker"""
        broker.route_message(message)
    
    def receive_message(self, message: AgentMessage):
        """Receive and queue message for processing"""
        self.message_queue.append(message)


class UserAgent(BaseAgent):
    """User agent that provides personalized services"""
    
    def __init__(self, agent_id: str, role: AgentRole, user_preferences: UserPreferences):
        super().__init__(agent_id, role)
        self.user_preferences = user_preferences
        self.decision_history = []
        self.learning_data = {}
        self.capabilities = [
            "personalization", "preference_learning", "custom_alerts",
            "dashboard_customization", "recommendation_filtering"
        ]
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process messages based on user preferences"""
        
        if message.message_type == MessageType.ALERT:
            return self._customize_alert(message)
        elif message.message_type == MessageType.RESPONSE:
            return self._personalize_response(message)
        elif message.message_type == MessageType.NOTIFICATION:
            return self._filter_notification(message)
        
        return None
    
    def _customize_alert(self, message: AgentMessage) -> AgentMessage:
        """Customize alert based on user preferences"""
        content = message.content.copy()
        
        # Adjust urgency based on user's alert frequency preference
        if self.user_preferences.alert_frequency == "real_time":
            content["delivery_method"] = "immediate"
        elif self.user_preferences.alert_frequency == "daily":
            content["delivery_method"] = "digest"
        
        # Filter metrics based on user preferences
        if "metrics" in content:
            filtered_metrics = {
                k: v for k, v in content["metrics"].items() 
                if k in self.user_preferences.preferred_metrics
            }
            content["metrics"] = filtered_metrics
        
        # Adapt communication style
        if self.user_preferences.communication_style == "concise":
            content["message"] = self._make_concise(content.get("message", ""))
        
        return AgentMessage(
            message_id=f"{message.message_id}_customized",
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type=MessageType.NOTIFICATION,
            content=content,
            timestamp=datetime.now()
        )
    
    def _personalize_response(self, message: AgentMessage) -> AgentMessage:
        """Personalize response based on user context"""
        content = message.content.copy()
        
        # Add user-specific context
        content["user_context"] = {
            "role": self.user_preferences.role.value,
            "risk_tolerance": self.user_preferences.risk_tolerance,
            "time_horizon": self.user_preferences.time_horizon
        }
        
        # Filter recommendations based on preferences
        if "recommendations" in content:
            filtered_recs = self._filter_recommendations(content["recommendations"])
            content["recommendations"] = filtered_recs
        
        return AgentMessage(
            message_id=f"{message.message_id}_personalized",
            sender_id=self.agent_id,
            recipient_id=self.user_preferences.user_id,
            message_type=MessageType.RESPONSE,
            content=content,
            timestamp=datetime.now()
        )
    
    def _filter_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Filter recommendations based on user preferences"""
        filtered = []
        
        for rec in recommendations:
            # Check against user's risk tolerance
            if self._matches_risk_tolerance(rec):
                # Check against sector preferences
                if self._matches_sector_preferences(rec):
                    # Adjust recommendation format
                    filtered.append(self._format_recommendation(rec))
        
        return filtered
    
    def _matches_risk_tolerance(self, recommendation: Dict) -> bool:
        """Check if recommendation matches user's risk tolerance"""
        rec_risk = recommendation.get("risk_level", "moderate")
        user_risk = self.user_preferences.risk_tolerance
        
        risk_mapping = {
            "conservative": ["conservative"],
            "moderate": ["conservative", "moderate"],
            "aggressive": ["conservative", "moderate", "aggressive"]
        }
        
        return rec_risk in risk_mapping.get(user_risk, ["moderate"])
    
    def _matches_sector_preferences(self, recommendation: Dict) -> bool:
        """Check if recommendation matches user's sector preferences"""
        if not self.user_preferences.sector_preferences:
            return True  # No preferences means accept all
        
        rec_sectors = recommendation.get("sectors", [])
        return any(sector in self.user_preferences.sector_preferences 
                  for sector in rec_sectors)
    
    def _format_recommendation(self, recommendation: Dict) -> Dict:
        """Format recommendation according to user preferences"""
        formatted = recommendation.copy()
        
        # Add personalized confidence scoring
        formatted["personalized_confidence"] = self._calculate_personal_confidence(recommendation)
        
        # Add reasoning based on user preferences
        formatted["reasoning"] = self._generate_personalized_reasoning(recommendation)
        
        return formatted
    
    def _calculate_personal_confidence(self, recommendation: Dict) -> float:
        """Calculate confidence score based on user's historical preferences"""
        base_confidence = recommendation.get("confidence", 0.5)
        
        # Adjust based on historical acceptance of similar recommendations
        historical_multiplier = self._get_historical_multiplier(recommendation)
        
        return min(base_confidence * historical_multiplier, 1.0)
    
    def _generate_personalized_reasoning(self, recommendation: Dict) -> str:
        """Generate reasoning tailored to user's role and preferences"""
        if self.user_preferences.role == AgentRole.PORTFOLIO_MANAGER:
            return f"This recommendation aligns with your {self.user_preferences.risk_tolerance} risk tolerance and {self.user_preferences.time_horizon} investment horizon."
        elif self.user_preferences.role == AgentRole.RISK_MANAGER:
            return f"Risk assessment shows this fits within your preferred risk parameters with VaR impact of {recommendation.get('var_impact', 'N/A')}."
        else:
            return "Recommendation based on your historical preferences and current market conditions."
    
    def update_preferences_from_feedback(self, feedback: Dict[str, Any]):
        """Update user preferences based on feedback and actions"""
        # This would implement preference learning logic
        # For now, just log the feedback
        self.learning_data[datetime.now().isoformat()] = feedback
        logger.info(f"Updated preferences for user {self.user_preferences.user_id}")
    
    def _make_concise(self, message: str) -> str:
        """Make message more concise for users who prefer brief communication"""
        # Simple implementation - in practice would use NLP
        sentences = message.split('. ')
        return '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else message
    
    def _get_historical_multiplier(self, recommendation: Dict) -> float:
        """Get multiplier based on historical acceptance of similar recommendations"""
        # Simplified implementation
        return 1.0  # Would analyze historical data in practice
    
    def _filter_notification(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Filter notifications based on user preferences"""
        # Check if notification is relevant to user's role and preferences
        content = message.content
        
        if "priority" in content:
            if content["priority"] < self.user_preferences.custom_thresholds.get("min_priority", 1):
                return None  # Filter out low priority notifications
        
        return message


class PortfolioManagementAgent(BaseAgent):
    """Central portfolio management agent that coordinates with user agents"""
    
    def __init__(self, twin_env: LoanPortfolioTwinEnv):
        super().__init__("portfolio_agent", AgentRole.PORTFOLIO_AGENT)
        self.twin_env = twin_env
        self.user_agents = {}
        self.coordination_data = {}
        self.capabilities = [
            "portfolio_optimization", "risk_assessment", "market_analysis",
            "compliance_monitoring", "coordination", "consensus_building"
        ]
    
    def register_user_agent(self, user_agent: UserAgent):
        """Register a user agent for coordination"""
        self.user_agents[user_agent.agent_id] = user_agent
        logger.info(f"Registered user agent {user_agent.agent_id} with role {user_agent.role.value}")
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process messages from user agents"""
        
        if message.message_type == MessageType.REQUEST:
            return self._handle_optimization_request(message)
        elif message.message_type == MessageType.INSIGHT:
            return self._handle_user_insight(message)
        
        return None
    
    def _handle_optimization_request(self, message: AgentMessage) -> AgentMessage:
        """Handle optimization request from user agent"""
        content = message.content
        
        # Extract user preferences and constraints
        user_preferences = content.get("preferences", {})
        constraints = content.get("constraints", {})
        
        # Get current portfolio state
        portfolio_state = self._get_portfolio_state()
        
        # Generate personalized recommendations
        recommendations = self._generate_personalized_recommendations(
            user_preferences, constraints, portfolio_state
        )
        
        # Create response
        response_content = {
            "request_id": content.get("request_id"),
            "recommendations": recommendations,
            "portfolio_metrics": self._get_relevant_metrics(user_preferences),
            "market_context": self._get_market_context(),
            "rationale": self._generate_rationale(recommendations, user_preferences)
        }
        
        return AgentMessage(
            message_id=f"opt_response_{datetime.now().timestamp()}",
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            content=response_content,
            timestamp=datetime.now()
        )
    
    def _handle_user_insight(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle insights shared by user agents"""
        content = message.content
        
        # Store insight for future use
        insight_type = content.get("pattern")
        self.coordination_data[insight_type] = content
        
        # Acknowledge receipt
        return AgentMessage(
            message_id=f"ack_{message.message_id}",
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            content={"status": "insight_received", "integration_status": "pending"},
            timestamp=datetime.now()
        )
    
    def _generate_personalized_recommendations(
        self, 
        user_preferences: Dict, 
        constraints: Dict, 
        portfolio_state: Dict
    ) -> List[Dict]:
        """Generate recommendations tailored to user preferences"""
        
        # Base recommendations from the twin environment
        base_action = self.twin_env.action_space.sample()
        
        # Adjust based on user preferences
        if user_preferences.get("risk_tolerance") == "conservative":
            base_action[0] = np.clip(base_action[0], -0.05, 0.05)  # Limit policy changes
            base_action[1] = np.clip(base_action[1], -0.01, 0.01)  # Limit rate changes
        elif user_preferences.get("risk_tolerance") == "aggressive":
            base_action[0] = np.clip(base_action[0], -0.1, 0.1)   # Allow larger changes
        
        # Generate multiple scenarios
        recommendations = []
        
        for i in range(3):  # Generate 3 different scenarios
            scenario_action = base_action + np.random.normal(0, 0.01, size=base_action.shape)
            scenario_action = np.clip(scenario_action, 
                                    self.twin_env.action_space.low, 
                                    self.twin_env.action_space.high)
            
            # Simulate outcome
            obs, reward, _, _, info = self.twin_env.step(scenario_action)
            
            recommendations.append({
                "scenario_id": f"scenario_{i+1}",
                "action": scenario_action.tolist(),
                "expected_reward": float(reward),
                "expected_roe": float(obs[3]),
                "expected_var": float(obs[4]),
                "confidence": 0.8 + np.random.normal(0, 0.1),
                "risk_level": self._assess_risk_level(scenario_action),
                "sectors": ["technology", "healthcare", "finance"],  # Simplified
                "rationale": f"This scenario targets {user_preferences.get('target_roe', 0.15):.1%} ROE with {user_preferences.get('risk_tolerance', 'moderate')} risk."
            })
        
        return sorted(recommendations, key=lambda x: x["expected_reward"], reverse=True)
    
    def _assess_risk_level(self, action: np.ndarray) -> str:
        """Assess risk level of an action"""
        action_magnitude = np.linalg.norm(action)
        
        if action_magnitude < 0.05:
            return "conservative"
        elif action_magnitude < 0.1:
            return "moderate"
        else:
            return "aggressive"
    
    def _get_portfolio_state(self) -> Dict:
        """Get current portfolio state"""
        obs = self.twin_env._get_observation()
        return {
            "portfolio_value": float(obs[0]),
            "expected_loss_rate": float(obs[1]),
            "delinquency_rate": float(obs[2]),
            "roe": float(obs[3]),
            "var_95": float(obs[4]),
            "concentration_risk": float(obs[5])
        }
    
    def _get_relevant_metrics(self, user_preferences: Dict) -> Dict:
        """Get metrics relevant to user preferences"""
        all_metrics = self._get_portfolio_state()
        preferred_metrics = user_preferences.get("preferred_metrics", list(all_metrics.keys()))
        
        return {k: v for k, v in all_metrics.items() if k in preferred_metrics}
    
    def _get_market_context(self) -> Dict:
        """Get current market context"""
        return {
            "interest_rate": float(self.twin_env.market_conditions.base_interest_rate),
            "unemployment_rate": float(self.twin_env.market_conditions.unemployment_rate),
            "gdp_growth": float(self.twin_env.market_conditions.gdp_growth),
            "volatility_index": float(self.twin_env.market_conditions.volatility_index)
        }
    
    def _generate_rationale(self, recommendations: List[Dict], user_preferences: Dict) -> str:
        """Generate rationale for recommendations"""
        best_rec = recommendations[0] if recommendations else {}
        
        return f"""
        Based on your {user_preferences.get('risk_tolerance', 'moderate')} risk tolerance and 
        preference for {user_preferences.get('time_horizon', 'quarterly')} performance, 
        the recommended scenario achieves {best_rec.get('expected_roe', 0):.1%} ROE 
        with {best_rec.get('expected_var', 0):.1%} VaR under current market conditions.
        """.strip()
    
    def broadcast_alert(self, alert_content: Dict) -> List[AgentMessage]:
        """Broadcast alert to all registered user agents"""
        messages = []
        
        for agent_id, user_agent in self.user_agents.items():
            message = AgentMessage(
                message_id=f"alert_{datetime.now().timestamp()}_{agent_id}",
                sender_id=self.agent_id,
                recipient_id=agent_id,
                message_type=MessageType.ALERT,
                content=alert_content,
                timestamp=datetime.now(),
                priority=alert_content.get("priority", 1)
            )
            messages.append(message)
        
        return messages


class AgentMessageBroker:
    """Message broker for agent communication"""
    
    def __init__(self):
        self.agents = {}
        self.message_history = []
        self.routing_rules = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the broker"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} with role {agent.role.value}")
    
    def route_message(self, message: AgentMessage):
        """Route message to recipient(s)"""
        self.message_history.append(message)
        
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.agents:
                self.agents[message.recipient_id].receive_message(message)
        else:
            # Broadcast message
            for agent in self.agents.values():
                if agent.agent_id != message.sender_id:
                    agent.receive_message(message)
    
    def process_messages(self):
        """Process all queued messages"""
        responses = []
        
        for agent in self.agents.values():
            while agent.message_queue:
                message = agent.message_queue.pop(0)
                response = agent.process_message(message)
                if response:
                    responses.append(response)
        
        # Route responses
        for response in responses:
            self.route_message(response)
        
        return len(responses)


class MultiAgentTwinEnv(LoanPortfolioTwinEnv):
    """Enhanced twin environment with multi-agent capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize multi-agent components
        self.message_broker = AgentMessageBroker()
        self.portfolio_agent = PortfolioManagementAgent(self)
        self.user_agents = {}
        
        # Register portfolio agent
        self.message_broker.register_agent(self.portfolio_agent)
    
    def add_user_agent(self, user_preferences: UserPreferences) -> UserAgent:
        """Add a user agent to the system"""
        agent_id = f"{user_preferences.role.value}_{user_preferences.user_id}"
        
        user_agent = UserAgent(agent_id, user_preferences.role, user_preferences)
        
        # Register with broker and portfolio agent
        self.message_broker.register_agent(user_agent)
        self.portfolio_agent.register_user_agent(user_agent)
        
        self.user_agents[agent_id] = user_agent
        
        return user_agent
    
    def request_personalized_optimization(
        self, 
        user_id: str, 
        preferences: Dict, 
        constraints: Dict = None
    ) -> Dict:
        """Request personalized optimization from user agent"""
        
        # Find user agent
        user_agent = None
        for agent in self.user_agents.values():
            if agent.user_preferences.user_id == user_id:
                user_agent = agent
                break
        
        if not user_agent:
            raise ValueError(f"User agent not found for user_id: {user_id}")
        
        # Create optimization request
        request = AgentMessage(
            message_id=f"opt_req_{datetime.now().timestamp()}",
            sender_id=user_agent.agent_id,
            recipient_id=self.portfolio_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "request_id": f"req_{user_id}_{datetime.now().timestamp()}",
                "preferences": preferences,
                "constraints": constraints or {},
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            requires_response=True
        )
        
        # Send request
        self.message_broker.route_message(request)
        
        # Process messages to get response
        self.message_broker.process_messages()
        
        # Get response from user agent's queue
        responses = [msg for msg in user_agent.message_queue 
                    if msg.message_type == MessageType.RESPONSE]
        
        if responses:
            response = responses[-1]  # Get latest response
            return response.content
        else:
            return {"error": "No response received"}
    
    def simulate_market_alert(self, alert_type: str, severity: str = "medium"):
        """Simulate a market alert and observe agent responses"""
        alert_content = {
            "alert_type": alert_type,
            "severity": severity,
            "message": f"Market alert: {alert_type} detected with {severity} severity",
            "priority": 2 if severity == "high" else 1,
            "metrics": {
                "ROE": self.portfolio.return_on_equity,
                "VaR": self.portfolio.var_95,
                "delinquency_rate": self.portfolio.delinquency_rate
            },
            "recommended_actions": ["review_portfolio", "assess_risk_exposure"]
        }
        
        # Broadcast alert
        alert_messages = self.portfolio_agent.broadcast_alert(alert_content)
        
        # Route messages
        for message in alert_messages:
            self.message_broker.route_message(message)
        
        # Process responses
        self.message_broker.process_messages()
        
        return alert_content
    
    def get_agent_analytics(self) -> Dict:
        """Get analytics on agent performance and interactions"""
        return {
            "total_agents": len(self.message_broker.agents),
            "user_agents": len(self.user_agents),
            "total_messages": len(self.message_broker.message_history),
            "message_types": {
                msg_type.value: len([msg for msg in self.message_broker.message_history 
                                   if msg.message_type == msg_type])
                for msg_type in MessageType
            },
            "agent_capabilities": {
                agent_id: agent.capabilities 
                for agent_id, agent in self.message_broker.agents.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create multi-agent twin environment
    multi_env = MultiAgentTwinEnv(
        initial_portfolio_size=100,
        max_portfolio_size=1000,
        simulation_days=30
    )
    
    # Initialize environment
    multi_env.reset()
    
    # Add user agents with different preferences
    pm_preferences = UserPreferences(
        user_id="pm_001",
        role=AgentRole.PORTFOLIO_MANAGER,
        risk_tolerance="moderate",
        preferred_metrics=["ROE", "Sharpe_ratio", "VaR"],
        dashboard_layout="executive",
        alert_frequency="real_time",
        time_horizon="quarterly"
    )
    
    rm_preferences = UserPreferences(
        user_id="rm_001", 
        role=AgentRole.RISK_MANAGER,
        risk_tolerance="conservative",
        preferred_metrics=["VaR", "expected_loss", "delinquency_rate"],
        dashboard_layout="risk_focused",
        alert_frequency="daily",
        communication_style="detailed"
    )
    
    # Add agents to environment
    pm_agent = multi_env.add_user_agent(pm_preferences)
    rm_agent = multi_env.add_user_agent(rm_preferences)
    
    print("Multi-Agent Digital Twin Environment Initialized")
    print(f"Portfolio Agent: {multi_env.portfolio_agent.agent_id}")
    print(f"User Agents: {list(multi_env.user_agents.keys())}")
    
    # Test personalized optimization request
    print("\n--- Testing Personalized Optimization ---")
    
    optimization_result = multi_env.request_personalized_optimization(
        user_id="pm_001",
        preferences={
            "target_roe": 0.18,
            "max_risk_tolerance": 0.05,
            "time_horizon": "6_months"
        },
        constraints={
            "regulatory_limits": True,
            "liquidity_requirements": 0.15
        }
    )
    
    print("Optimization Result:", json.dumps(optimization_result, indent=2, default=str))
    
    # Test market alert
    print("\n--- Testing Market Alert ---")
    alert_result = multi_env.simulate_market_alert("interest_rate_increase", "high")
    print("Alert broadcasted:", alert_result["alert_type"])
    
    # Get analytics
    print("\n--- Agent Analytics ---")
    analytics = multi_env.get_agent_analytics()
    print("System Analytics:", json.dumps(analytics, indent=2))
    
    print("\n--- Multi-Agent System Demonstration Complete ---") 