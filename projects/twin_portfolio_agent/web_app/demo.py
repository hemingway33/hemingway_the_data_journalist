#!/usr/bin/env python3
"""
Multi-Agent Digital Twin Portfolio Management Demo Script

This script demonstrates the capabilities of the multi-agent system by:
1. Starting the multi-agent environment
2. Creating sample user agents
3. Running portfolio optimizations
4. Simulating market alerts
5. Generating performance reports

Usage: python demo.py [--mode interactive|automated]
"""

import asyncio
import json
import logging
import time
import argparse
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from multi_agent_twin_env import (
        MultiAgentTwinEnv, UserPreferences, AgentRole
    )
    from twin_env import LoanPortfolioTwinEnv
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure you're running this from the correct directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioDemoRunner:
    """Demo runner for the multi-agent portfolio management system"""
    
    def __init__(self, mode: str = "automated"):
        self.mode = mode
        self.env = None
        self.demo_agents = []
        self.demo_results = {}
        
    def print_banner(self, text: str, char: str = "="):
        """Print a formatted banner"""
        width = 80
        padding = (width - len(text) - 2) // 2
        print(f"{char * width}")
        print(f"{char}{' ' * padding}{text}{' ' * padding}{char}")
        print(f"{char * width}")
        print()
    
    def print_section(self, text: str):
        """Print a section header"""
        print(f"\n🔹 {text}")
        print("-" * (len(text) + 3))
    
    def wait_for_user(self, message: str = "Press Enter to continue..."):
        """Wait for user input in interactive mode"""
        if self.mode == "interactive":
            input(f"\n{message}")
        else:
            print(f"\n⏳ {message}")
            time.sleep(2)
    
    def initialize_environment(self):
        """Initialize the multi-agent environment"""
        self.print_section("Initializing Multi-Agent Environment")
        
        try:
            self.env = MultiAgentTwinEnv(
                initial_portfolio_size=200,
                max_portfolio_size=2000,
                simulation_days=30
            )
            
            # Reset environment
            observation, info = self.env.reset()
            
            print(f"✅ Environment initialized successfully")
            print(f"   - Portfolio Size: {len(self.env.portfolio.loans)} loans")
            print(f"   - Portfolio Value: ${self.env.portfolio.total_value:,.2f}")
            print(f"   - Current ROE: {self.env.portfolio.return_on_equity:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            return False
    
    def create_demo_agents(self):
        """Create demonstration user agents"""
        self.print_section("Creating Demo User Agents")
        
        # Define demo agents with different roles and preferences
        demo_agent_configs = [
            {
                "user_id": "portfolio_manager_alice",
                "role": AgentRole.PORTFOLIO_MANAGER,
                "risk_tolerance": "moderate",
                "preferred_metrics": ["ROE", "VaR", "Sharpe_ratio"],
                "dashboard_layout": "executive",
                "alert_frequency": "real_time",
                "communication_style": "detailed",
                "time_horizon": "quarterly"
            },
            {
                "user_id": "risk_manager_bob",
                "role": AgentRole.RISK_MANAGER,
                "risk_tolerance": "conservative",
                "preferred_metrics": ["VaR", "expected_loss", "concentration_risk"],
                "dashboard_layout": "risk_focused",
                "alert_frequency": "hourly",
                "communication_style": "summary",
                "time_horizon": "monthly"
            },
            {
                "user_id": "credit_officer_carol",
                "role": AgentRole.CREDIT_OFFICER,
                "risk_tolerance": "aggressive",
                "preferred_metrics": ["delinquency_rate", "expected_loss", "ROE"],
                "dashboard_layout": "detailed",
                "alert_frequency": "daily",
                "communication_style": "technical",
                "time_horizon": "weekly"
            }
        ]
        
        for config in demo_agent_configs:
            try:
                preferences = UserPreferences(**config)
                agent = self.env.add_user_agent(preferences)
                self.demo_agents.append((agent, preferences))
                
                print(f"✅ Created {config['role'].value}: {config['user_id']}")
                print(f"   - Risk Tolerance: {config['risk_tolerance']}")
                print(f"   - Preferred Metrics: {', '.join(config['preferred_metrics'][:3])}")
                
            except Exception as e:
                logger.error(f"Failed to create agent {config['user_id']}: {e}")
        
        print(f"\n📊 Total agents created: {len(self.demo_agents)}")
        
    def demonstrate_portfolio_optimization(self):
        """Demonstrate personalized portfolio optimization"""
        self.print_section("Portfolio Optimization Demonstration")
        
        if not self.demo_agents:
            print("❌ No agents available for optimization")
            return
        
        # Run optimization for each agent
        for agent, preferences in self.demo_agents:
            print(f"\n🎯 Optimizing portfolio for {preferences.user_id}...")
            
            # Create optimization preferences based on agent role
            if preferences.role == AgentRole.PORTFOLIO_MANAGER:
                opt_preferences = {
                    "target_roe": 0.18,
                    "max_risk_tolerance": 0.06,
                    "time_horizon": "quarterly"
                }
            elif preferences.role == AgentRole.RISK_MANAGER:
                opt_preferences = {
                    "target_roe": 0.12,
                    "max_risk_tolerance": 0.03,
                    "time_horizon": "monthly"
                }
            else:  # Credit Officer
                opt_preferences = {
                    "target_roe": 0.20,
                    "max_risk_tolerance": 0.08,
                    "time_horizon": "weekly"
                }
            
            try:
                result = self.env.request_personalized_optimization(
                    user_id=preferences.user_id,
                    preferences=opt_preferences,
                    constraints={
                        "regulatory_limits": True,
                        "liquidity_requirements": 0.15
                    }
                )
                
                print(f"   ✅ Optimization completed")
                print(f"   - Target ROE: {opt_preferences['target_roe']:.1%}")
                print(f"   - Max Risk: {opt_preferences['max_risk_tolerance']:.1%}")
                print(f"   - Status: {result['status']}")
                
                # Store results
                self.demo_results[preferences.user_id] = result
                
            except Exception as e:
                logger.error(f"Optimization failed for {preferences.user_id}: {e}")
            
            self.wait_for_user("Continue to next optimization...")
    
    def simulate_market_scenarios(self):
        """Simulate various market scenarios and alerts"""
        self.print_section("Market Scenario Simulation")
        
        scenarios = [
            {
                "name": "Interest Rate Increase",
                "alert_type": "interest_rate_increase",
                "severity": "high",
                "description": "Federal Reserve raises interest rates by 50 basis points"
            },
            {
                "name": "Economic Downturn",
                "alert_type": "economic_downturn",
                "severity": "medium",
                "description": "GDP growth slows, unemployment rises"
            },
            {
                "name": "Credit Spread Widening",
                "alert_type": "credit_spread_widening",
                "severity": "medium",
                "description": "Corporate credit spreads widen due to market uncertainty"
            },
            {
                "name": "Market Volatility Spike",
                "alert_type": "market_volatility",
                "severity": "high",
                "description": "VIX spikes above 30 due to geopolitical tensions"
            }
        ]
        
        for scenario in scenarios:
            print(f"\n🚨 Simulating: {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            
            try:
                # Simulate the market alert
                alert_result = self.env.simulate_market_alert(
                    alert_type=scenario["alert_type"],
                    severity=scenario["severity"]
                )
                
                print(f"   ✅ Alert generated: {scenario['severity']} severity")
                print(f"   - Alert Type: {scenario['alert_type']}")
                print(f"   - Message: {alert_result.get('message', 'Market condition alert')}")
                
                # Show agent responses
                messages = self.env.get_recent_messages(limit=5)
                if messages:
                    print(f"   - Agent Messages: {len(messages)} responses")
                
            except Exception as e:
                logger.error(f"Failed to simulate {scenario['name']}: {e}")
            
            self.wait_for_user("Continue to next scenario...")
    
    def run_simulation_steps(self, steps: int = 5):
        """Run simulation steps and track performance"""
        self.print_section(f"Running {steps} Simulation Steps")
        
        initial_metrics = self.env._get_observation()
        print(f"📊 Initial Portfolio Metrics:")
        print(f"   - Value: ${initial_metrics[0] * 1e6:,.2f}")
        print(f"   - ROE: {initial_metrics[3]:.2%}")
        print(f"   - VaR 95%: {initial_metrics[4]:.2%}")
        print(f"   - Delinquency Rate: {initial_metrics[2]:.2%}")
        
        for step in range(steps):
            print(f"\n🔄 Simulation Step {step + 1}/{steps}")
            
            # Take a random action (in practice, this would be agent-driven)
            action = self.env.action_space.sample()
            
            try:
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                print(f"   - Reward: {reward:.4f}")
                print(f"   - Portfolio Value: ${obs[0] * 1e6:,.2f}")
                print(f"   - ROE: {obs[3]:.2%}")
                print(f"   - Portfolio Size: {info.get('portfolio_size', 'N/A')} loans")
                
                if terminated or truncated:
                    print("   ⚠️  Simulation terminated")
                    break
                    
            except Exception as e:
                logger.error(f"Simulation step {step + 1} failed: {e}")
            
            if self.mode == "interactive" and step < steps - 1:
                self.wait_for_user("Continue to next step...")
            else:
                time.sleep(1)
    
    def generate_performance_report(self):
        """Generate and display performance report"""
        self.print_section("Performance Report")
        
        try:
            # Get portfolio analytics
            analytics = self.env.get_agent_analytics()
            performance_df = self.env.get_performance_summary()
            
            print(f"📈 Portfolio Performance Summary:")
            print(f"   - Total Agents: {analytics['total_agents']}")
            print(f"   - User Agents: {analytics['user_agents']}")
            print(f"   - Total Messages: {analytics['total_messages']}")
            
            if not performance_df.empty:
                latest = performance_df.iloc[-1]
                print(f"\n📊 Latest Performance Metrics:")
                print(f"   - Day: {latest.get('day', 'N/A')}")
                print(f"   - Portfolio Value: ${latest.get('portfolio_value', 0):,.2f}")
                print(f"   - ROE: {latest.get('roe', 0):.2%}")
                print(f"   - VaR 95%: {latest.get('var_95', 0):.2%}")
                print(f"   - Delinquency Rate: {latest.get('delinquency_rate', 0):.2%}")
                print(f"   - Reward: {latest.get('reward', 0):.4f}")
            
            # Agent-specific results
            if self.demo_results:
                print(f"\n🎯 Agent Optimization Results:")
                for user_id, result in self.demo_results.items():
                    print(f"   - {user_id}: {result.get('status', 'Unknown')}")
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"demo_report_{timestamp}.json"
            
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "analytics": analytics,
                "performance_history": performance_df.to_dict('records') if not performance_df.empty else [],
                "optimization_results": self.demo_results,
                "demo_agents": [
                    {
                        "user_id": prefs.user_id,
                        "role": prefs.role.value,
                        "risk_tolerance": prefs.risk_tolerance
                    }
                    for _, prefs in self.demo_agents
                ]
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"\n💾 Report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
    
    def run_demo(self):
        """Run the complete demonstration"""
        self.print_banner("Multi-Agent Digital Twin Portfolio Management Demo")
        
        print(f"🎯 Demo Mode: {self.mode.upper()}")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.mode == "interactive":
            print("\n💡 This demo will walk you through the multi-agent system capabilities.")
            print("   Press Enter at each step to continue...")
        else:
            print("\n🤖 Running automated demo with 2-second delays between steps...")
        
        self.wait_for_user("Ready to start the demo?")
        
        # Demo sequence
        steps = [
            ("Environment Initialization", self.initialize_environment),
            ("Agent Creation", self.create_demo_agents),
            ("Portfolio Optimization", self.demonstrate_portfolio_optimization),
            ("Market Simulation", self.simulate_market_scenarios),
            ("Simulation Steps", lambda: self.run_simulation_steps(3)),
            ("Performance Report", self.generate_performance_report)
        ]
        
        for step_name, step_func in steps:
            try:
                success = step_func()
                if success is False:
                    print(f"❌ {step_name} failed, stopping demo")
                    break
            except KeyboardInterrupt:
                print(f"\n\n🛑 Demo interrupted by user")
                break
            except Exception as e:
                logger.error(f"{step_name} failed: {e}")
                if self.mode == "interactive":
                    if input("Continue with demo? (y/n): ").lower() != 'y':
                        break
                else:
                    print("Continuing automated demo...")
        
        self.print_banner("Demo Completed! 🎉", "🎉")
        print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📚 Next Steps:")
        print("   - Start the web application: ./start_all.sh")
        print("   - Explore the frontend at: http://localhost:3000")
        print("   - Check API docs at: http://localhost:8000/docs")
        print("   - Review the demo report file generated above")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Digital Twin Portfolio Management Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "automated"],
        default="automated",
        help="Demo mode: interactive (wait for user input) or automated (run continuously)"
    )
    
    args = parser.parse_args()
    
    try:
        demo = PortfolioDemoRunner(mode=args.mode)
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 