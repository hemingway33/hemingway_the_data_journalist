#!/usr/bin/env python3
"""
Multi-Agent Digital Twin Web Application Test Suite

This script tests the web application functionality including:
1. Backend API endpoints
2. WebSocket connections
3. Frontend-backend integration
4. Multi-agent system integration

Usage: python test_webapp.py [--host HOST] [--port PORT]
"""

import asyncio
import json
import requests
import websockets
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebAppTester:
    """Comprehensive web application test suite"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}"
        self.test_results = {
            "api_tests": {},
            "websocket_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "start_time": datetime.now().isoformat()
        }
        self.session = requests.Session()
        self.session.timeout = 10
    
    def print_banner(self, text: str, char: str = "="):
        """Print a formatted banner"""
        width = 80
        padding = (width - len(text) - 2) // 2
        print(f"{char * width}")
        print(f"{char}{' ' * padding}{text}{' ' * padding}{char}")
        print(f"{char * width}")
        print()
    
    def print_test_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """Print test result with formatting"""
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f" ({duration:.3f}s)" if duration > 0 else ""
        print(f"{status} {test_name}{duration_str}")
        if message:
            print(f"    {message}")
    
    def test_server_health(self) -> bool:
        """Test if the server is running and healthy"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            if success:
                data = response.json()
                message += f", Response: {data.get('status', 'unknown')}"
            
            self.print_test_result("Server Health Check", success, message, duration)
            self.test_results["api_tests"]["health_check"] = {
                "success": success,
                "status_code": response.status_code,
                "duration": duration,
                "response": response.json() if success else None
            }
            return success
            
        except Exception as e:
            self.print_test_result("Server Health Check", False, str(e))
            self.test_results["api_tests"]["health_check"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_system_status(self) -> bool:
        """Test system status endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/system/status")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            if success:
                data = response.json()
                message += f", Agents: {data.get('total_agents', 0)}, Value: ${data.get('portfolio_value', 0):,.0f}"
            
            self.print_test_result("System Status", success, message, duration)
            self.test_results["api_tests"]["system_status"] = {
                "success": success,
                "status_code": response.status_code,
                "duration": duration,
                "response": response.json() if success else None
            }
            return success
            
        except Exception as e:
            self.print_test_result("System Status", False, str(e))
            self.test_results["api_tests"]["system_status"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_portfolio_metrics(self) -> bool:
        """Test portfolio metrics endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/portfolio/metrics")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            if success:
                data = response.json()
                current_metrics = data.get("current_metrics", {})
                message += f", ROE: {current_metrics.get('roe', 0):.2%}, VaR: {current_metrics.get('var_95', 0):.2%}"
            
            self.print_test_result("Portfolio Metrics", success, message, duration)
            self.test_results["api_tests"]["portfolio_metrics"] = {
                "success": success,
                "status_code": response.status_code,
                "duration": duration,
                "response": response.json() if success else None
            }
            return success
            
        except Exception as e:
            self.print_test_result("Portfolio Metrics", False, str(e))
            self.test_results["api_tests"]["portfolio_metrics"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_agent_creation(self) -> Optional[str]:
        """Test agent creation endpoint"""
        test_agent_data = {
            "user_id": f"test_agent_{int(time.time())}",
            "role": "portfolio_manager",
            "risk_tolerance": "moderate",
            "preferred_metrics": ["ROE", "VaR"],
            "dashboard_layout": "standard",
            "alert_frequency": "daily"
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/agents/create",
                json=test_agent_data
            )
            duration = time.time() - start_time
            
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            agent_id = None
            
            if success:
                data = response.json()
                agent_id = data.get("agent_id")
                message += f", Agent ID: {agent_id}"
            
            self.print_test_result("Agent Creation", success, message, duration)
            self.test_results["api_tests"]["agent_creation"] = {
                "success": success,
                "status_code": response.status_code,
                "duration": duration,
                "agent_data": test_agent_data,
                "response": response.json() if success else None
            }
            return agent_id
            
        except Exception as e:
            self.print_test_result("Agent Creation", False, str(e))
            self.test_results["api_tests"]["agent_creation"] = {
                "success": False,
                "error": str(e)
            }
            return None
    
    def test_portfolio_optimization(self, user_id: str) -> bool:
        """Test portfolio optimization endpoint"""
        optimization_data = {
            "user_id": user_id,
            "preferences": {
                "target_roe": 0.15,
                "max_risk_tolerance": 0.05,
                "time_horizon": "quarterly"
            },
            "constraints": {
                "regulatory_limits": True,
                "liquidity_requirements": 0.15
            }
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/portfolio/optimize",
                json=optimization_data
            )
            duration = time.time() - start_time
            
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            if success:
                data = response.json()
                message += f", Status: {data.get('status', 'unknown')}"
            
            self.print_test_result("Portfolio Optimization", success, message, duration)
            self.test_results["api_tests"]["portfolio_optimization"] = {
                "success": success,
                "status_code": response.status_code,
                "duration": duration,
                "request_data": optimization_data,
                "response": response.json() if success else None
            }
            return success
            
        except Exception as e:
            self.print_test_result("Portfolio Optimization", False, str(e))
            self.test_results["api_tests"]["portfolio_optimization"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_alert_simulation(self) -> bool:
        """Test alert simulation endpoint"""
        alert_data = {
            "alert_type": "interest_rate_increase",
            "severity": "medium"
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/alerts/simulate",
                json=alert_data
            )
            duration = time.time() - start_time
            
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            if success:
                data = response.json()
                message += f", Alert: {data.get('alert', {}).get('alert_type', 'unknown')}"
            
            self.print_test_result("Alert Simulation", success, message, duration)
            self.test_results["api_tests"]["alert_simulation"] = {
                "success": success,
                "status_code": response.status_code,
                "duration": duration,
                "request_data": alert_data,
                "response": response.json() if success else None
            }
            return success
            
        except Exception as e:
            self.print_test_result("Alert Simulation", False, str(e))
            self.test_results["api_tests"]["alert_simulation"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_simulation_step(self) -> bool:
        """Test simulation step endpoint"""
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/simulation/step")
            duration = time.time() - start_time
            
            success = response.status_code == 200
            message = f"Status: {response.status_code}"
            if success:
                data = response.json()
                message += f", Day: {data.get('current_day', 'N/A')}, Reward: {data.get('reward', 0):.4f}"
            
            self.print_test_result("Simulation Step", success, message, duration)
            self.test_results["api_tests"]["simulation_step"] = {
                "success": success,
                "status_code": response.status_code,
                "duration": duration,
                "response": response.json() if success else None
            }
            return success
            
        except Exception as e:
            self.print_test_result("Simulation Step", False, str(e))
            self.test_results["api_tests"]["simulation_step"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection"""
        client_id = f"test_client_{int(time.time())}"
        messages_received = []
        
        try:
            start_time = time.time()
            
            async with websockets.connect(f"{self.ws_url}/ws/{client_id}") as websocket:
                # Send ping
                await websocket.send(json.dumps({"type": "ping"}))
                
                # Wait for pong and initial messages
                for _ in range(3):  # Wait for up to 3 messages
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        messages_received.append(data)
                        if data.get("type") == "pong":
                            break
                    except asyncio.TimeoutError:
                        break
                
                duration = time.time() - start_time
                success = len(messages_received) > 0
                message = f"Messages received: {len(messages_received)}"
                if messages_received:
                    message += f", Types: {[msg.get('type', 'unknown') for msg in messages_received]}"
                
                self.print_test_result("WebSocket Connection", success, message, duration)
                self.test_results["websocket_tests"]["connection"] = {
                    "success": success,
                    "duration": duration,
                    "messages_received": len(messages_received),
                    "message_types": [msg.get("type") for msg in messages_received]
                }
                return success
                
        except Exception as e:
            self.print_test_result("WebSocket Connection", False, str(e))
            self.test_results["websocket_tests"]["connection"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    async def test_websocket_messaging(self) -> bool:
        """Test WebSocket messaging functionality"""
        client_id = f"test_client_msg_{int(time.time())}"
        
        try:
            start_time = time.time()
            
            async with websockets.connect(f"{self.ws_url}/ws/{client_id}") as websocket:
                # Request metrics
                await websocket.send(json.dumps({"type": "request_metrics"}))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    duration = time.time() - start_time
                    success = data.get("type") == "metrics_update"
                    message = f"Response type: {data.get('type', 'unknown')}"
                    
                    self.print_test_result("WebSocket Messaging", success, message, duration)
                    self.test_results["websocket_tests"]["messaging"] = {
                        "success": success,
                        "duration": duration,
                        "response_type": data.get("type"),
                        "has_data": "data" in data
                    }
                    return success
                    
                except asyncio.TimeoutError:
                    self.print_test_result("WebSocket Messaging", False, "Timeout waiting for response")
                    self.test_results["websocket_tests"]["messaging"] = {
                        "success": False,
                        "error": "Timeout"
                    }
                    return False
                
        except Exception as e:
            self.print_test_result("WebSocket Messaging", False, str(e))
            self.test_results["websocket_tests"]["messaging"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_api_performance(self) -> bool:
        """Test API performance with multiple requests"""
        endpoints = [
            "/api/system/status",
            "/api/portfolio/metrics",
            "/api/agents"
        ]
        
        performance_results = {}
        overall_success = True
        
        for endpoint in endpoints:
            try:
                times = []
                successes = 0
                
                for _ in range(5):  # 5 requests per endpoint
                    start_time = time.time()
                    response = self.session.get(f"{self.base_url}{endpoint}")
                    duration = time.time() - start_time
                    times.append(duration)
                    if response.status_code == 200:
                        successes += 1
                
                avg_time = sum(times) / len(times)
                max_time = max(times)
                success_rate = successes / len(times)
                
                endpoint_success = success_rate >= 0.8 and avg_time < 2.0
                overall_success = overall_success and endpoint_success
                
                performance_results[endpoint] = {
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "success_rate": success_rate,
                    "success": endpoint_success
                }
                
                message = f"Avg: {avg_time:.3f}s, Max: {max_time:.3f}s, Success: {success_rate:.1%}"
                self.print_test_result(f"Performance {endpoint}", endpoint_success, message)
                
            except Exception as e:
                overall_success = False
                performance_results[endpoint] = {"success": False, "error": str(e)}
                self.print_test_result(f"Performance {endpoint}", False, str(e))
        
        self.test_results["performance_tests"] = performance_results
        return overall_success
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid requests"""
        error_tests = [
            {
                "name": "Invalid Agent Role",
                "endpoint": "/api/agents/create",
                "method": "POST",
                "data": {"user_id": "test", "role": "invalid_role"},
                "expected_status": 400
            },
            {
                "name": "Missing Required Fields",
                "endpoint": "/api/portfolio/optimize",
                "method": "POST",
                "data": {"user_id": "nonexistent"},
                "expected_status": [400, 404]
            },
            {
                "name": "Invalid Endpoint",
                "endpoint": "/api/nonexistent",
                "method": "GET",
                "data": None,
                "expected_status": 404
            }
        ]
        
        all_passed = True
        
        for test in error_tests:
            try:
                start_time = time.time()
                
                if test["method"] == "POST":
                    response = self.session.post(
                        f"{self.base_url}{test['endpoint']}",
                        json=test["data"]
                    )
                else:
                    response = self.session.get(f"{self.base_url}{test['endpoint']}")
                
                duration = time.time() - start_time
                expected_status = test["expected_status"]
                if isinstance(expected_status, list):
                    success = response.status_code in expected_status
                else:
                    success = response.status_code == expected_status
                
                all_passed = all_passed and success
                message = f"Status: {response.status_code}, Expected: {expected_status}"
                self.print_test_result(f"Error Handling - {test['name']}", success, message, duration)
                
            except Exception as e:
                all_passed = False
                self.print_test_result(f"Error Handling - {test['name']}", False, str(e))
        
        self.test_results["integration_tests"]["error_handling"] = {"success": all_passed}
        return all_passed
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.test_results["end_time"] = datetime.now().isoformat()
        
        # Calculate summary statistics
        api_tests = self.test_results["api_tests"]
        api_success_count = sum(1 for test in api_tests.values() if test.get("success", False))
        api_total = len(api_tests)
        
        ws_tests = self.test_results["websocket_tests"]
        ws_success_count = sum(1 for test in ws_tests.values() if test.get("success", False))
        ws_total = len(ws_tests)
        
        perf_tests = self.test_results["performance_tests"]
        perf_success_count = sum(1 for test in perf_tests.values() if test.get("success", False))
        perf_total = len(perf_tests)
        
        summary = {
            "api_tests": {"passed": api_success_count, "total": api_total},
            "websocket_tests": {"passed": ws_success_count, "total": ws_total},
            "performance_tests": {"passed": perf_success_count, "total": perf_total},
            "overall_success": (api_success_count == api_total and 
                              ws_success_count == ws_total and 
                              perf_success_count == perf_total)
        }
        
        self.test_results["summary"] = summary
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n📊 Test Summary:")
        print(f"   API Tests: {api_success_count}/{api_total} passed")
        print(f"   WebSocket Tests: {ws_success_count}/{ws_total} passed")
        print(f"   Performance Tests: {perf_success_count}/{perf_total} passed")
        print(f"   Overall Result: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
        print(f"\n💾 Detailed report saved to: {report_file}")
    
    async def run_all_tests(self):
        """Run all test suites"""
        self.print_banner("Multi-Agent Digital Twin Web App Test Suite")
        
        print(f"🎯 Testing server at: {self.base_url}")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # API Tests
        print("🔧 API Endpoint Tests")
        print("-" * 20)
        
        if not self.test_server_health():
            print("❌ Server is not running or not healthy. Stopping tests.")
            return
        
        self.test_system_status()
        self.test_portfolio_metrics()
        
        # Create test agent and use for subsequent tests
        test_agent_id = self.test_agent_creation()
        if test_agent_id:
            time.sleep(1)  # Wait for agent to be created
            self.test_portfolio_optimization(test_agent_id)
        
        self.test_alert_simulation()
        self.test_simulation_step()
        
        # WebSocket Tests
        print("\n🔌 WebSocket Tests")
        print("-" * 17)
        
        await self.test_websocket_connection()
        await self.test_websocket_messaging()
        
        # Performance Tests
        print("\n⚡ Performance Tests")
        print("-" * 19)
        
        self.test_api_performance()
        
        # Error Handling Tests
        print("\n🛡️  Error Handling Tests")
        print("-" * 22)
        
        self.test_error_handling()
        
        # Generate report
        print("\n📋 Generating Test Report")
        print("-" * 25)
        
        self.generate_test_report()
        
        print("\n" + "=" * 80)
        print("✅ Test suite completed!")
        print("=" * 80)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Digital Twin Web Application Test Suite"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Backend server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Backend server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    try:
        tester = WebAppTester(host=args.host, port=args.port)
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 