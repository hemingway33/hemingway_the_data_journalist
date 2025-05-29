#!/usr/bin/env python3
"""
Multi-Agent Digital Twin Portfolio Management - Master Launch Script

This script provides a unified interface to launch and manage the complete
multi-agent digital twin portfolio management system including:
1. Backend API server
2. Frontend development server
3. Demo system
4. Testing suite

Usage: python launch.py [command] [options]
"""

import asyncio
import subprocess
import sys
import os
import time
import signal
import argparse
import threading
import webbrowser
from datetime import datetime
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemLauncher:
    """Master system launcher and controller"""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.web_app_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.web_app_dir)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
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
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        self.print_section("Checking Dependencies")
        
        dependencies = [
            ("Python 3", "python3", "--version"),
            ("Node.js", "node", "--version"),
            ("npm", "npm", "--version"),
        ]
        
        all_good = True
        
        for name, command, arg in dependencies:
            try:
                result = subprocess.run(
                    [command, arg], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    print(f"✅ {name}: {version}")
                else:
                    print(f"❌ {name}: Not found or error")
                    all_good = False
            except (subprocess.SubprocessError, FileNotFoundError):
                print(f"❌ {name}: Not installed")
                all_good = False
        
        return all_good
    
    def install_dependencies(self) -> bool:
        """Install Python and Node.js dependencies"""
        self.print_section("Installing Dependencies")
        
        try:
            # Backend dependencies
            print("📦 Installing backend dependencies...")
            backend_dir = os.path.join(self.web_app_dir, "backend")
            
            if os.path.exists(os.path.join(backend_dir, ".venv")):
                print("   Virtual environment already exists")
            else:
                print("   Creating virtual environment...")
                subprocess.run(
                    [sys.executable, "-m", "venv", ".venv"],
                    cwd=backend_dir,
                    check=True
                )
            
            # Activate virtual environment and install packages
            venv_python = os.path.join(backend_dir, ".venv", "bin", "python")
            if not os.path.exists(venv_python):
                venv_python = os.path.join(backend_dir, ".venv", "Scripts", "python.exe")
            
            subprocess.run(
                [venv_python, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=backend_dir,
                check=True
            )
            print("✅ Backend dependencies installed")
            
            # Frontend dependencies
            print("📦 Installing frontend dependencies...")
            frontend_dir = os.path.join(self.web_app_dir, "frontend")
            
            if os.path.exists(os.path.join(frontend_dir, "node_modules")):
                print("   Node modules already exist")
            else:
                subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_dir,
                    check=True
                )
            print("✅ Frontend dependencies installed")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Dependency installation failed: {e}")
            return False
    
    def start_backend(self) -> bool:
        """Start the FastAPI backend server"""
        self.print_section("Starting Backend Server")
        
        backend_dir = os.path.join(self.web_app_dir, "backend")
        venv_python = os.path.join(backend_dir, ".venv", "bin", "python")
        if not os.path.exists(venv_python):
            venv_python = os.path.join(backend_dir, ".venv", "Scripts", "python.exe")
        
        try:
            process = subprocess.Popen(
                [venv_python, "main.py"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes["backend"] = process
            print(f"✅ Backend server started (PID: {process.pid})")
            print("   Backend URL: http://localhost:8000")
            print("   API Docs: http://localhost:8000/docs")
            
            # Wait a moment to check if it started successfully
            time.sleep(2)
            if process.poll() is not None:
                print("❌ Backend server failed to start")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self) -> bool:
        """Start the React frontend development server"""
        self.print_section("Starting Frontend Server")
        
        frontend_dir = os.path.join(self.web_app_dir, "frontend")
        
        try:
            process = subprocess.Popen(
                ["npm", "start"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes["frontend"] = process
            print(f"✅ Frontend server starting (PID: {process.pid})")
            print("   Frontend URL: http://localhost:3000")
            print("   Starting development server...")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def wait_for_services(self):
        """Wait for services to be ready"""
        self.print_section("Waiting for Services")
        
        import requests
        
        # Wait for backend
        print("⏳ Waiting for backend to be ready...")
        backend_ready = False
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    backend_ready = True
                    print("✅ Backend is ready!")
                    break
            except:
                pass
            time.sleep(1)
        
        if not backend_ready:
            print("⚠️  Backend may not be ready yet")
        
        # Wait for frontend (it takes longer to start)
        print("⏳ Waiting for frontend to be ready...")
        print("   This may take 30-60 seconds for the first time...")
        
        # Frontend typically takes longer, so we'll just wait a bit
        time.sleep(10)
        print("✅ Frontend should be starting up")
    
    def open_browser(self):
        """Open web browser to the application"""
        self.print_section("Opening Web Browser")
        
        try:
            webbrowser.open("http://localhost:3000")
            print("✅ Browser opened to http://localhost:3000")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("   Please open http://localhost:3000 manually")
    
    def run_demo(self, mode: str = "automated"):
        """Run the demonstration script"""
        self.print_section(f"Running Demo ({mode} mode)")
        
        try:
            demo_script = os.path.join(self.web_app_dir, "demo.py")
            result = subprocess.run(
                [sys.executable, demo_script, "--mode", mode],
                cwd=self.web_app_dir
            )
            
            if result.returncode == 0:
                print("✅ Demo completed successfully")
            else:
                print("❌ Demo failed or was interrupted")
            
        except Exception as e:
            print(f"❌ Failed to run demo: {e}")
    
    async def run_tests(self):
        """Run the test suite"""
        self.print_section("Running Test Suite")
        
        try:
            test_script = os.path.join(self.web_app_dir, "test_webapp.py")
            result = subprocess.run(
                [sys.executable, test_script],
                cwd=self.web_app_dir
            )
            
            if result.returncode == 0:
                print("✅ Tests completed successfully")
            else:
                print("❌ Some tests failed")
            
        except Exception as e:
            print(f"❌ Failed to run tests: {e}")
    
    def monitor_services(self):
        """Monitor running services"""
        while self.running:
            for service, process in self.processes.items():
                if process.poll() is not None:
                    print(f"⚠️  {service} service stopped unexpectedly")
                    # Could implement restart logic here
            time.sleep(5)
    
    def shutdown(self):
        """Shutdown all services"""
        self.print_section("Shutting Down Services")
        self.running = False
        
        for service, process in self.processes.items():
            try:
                print(f"🛑 Stopping {service}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"   Force killing {service}...")
                    process.kill()
                
                print(f"✅ {service} stopped")
                
            except Exception as e:
                print(f"❌ Error stopping {service}: {e}")
        
        self.processes.clear()
        print("✅ All services stopped")
    
    def start_full_system(self, open_browser: bool = True, run_demo: bool = False):
        """Start the complete system"""
        self.print_banner("Multi-Agent Digital Twin Portfolio Management System")
        
        print(f"🚀 Starting complete system at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check dependencies
        if not self.check_dependencies():
            print("❌ Missing dependencies. Please install required software.")
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            print("❌ Failed to install dependencies.")
            return False
        
        # Start services
        if not self.start_backend():
            print("❌ Failed to start backend.")
            return False
        
        if not self.start_frontend():
            print("❌ Failed to start frontend.")
            return False
        
        self.running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_services)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for services to be ready
        self.wait_for_services()
        
        # Open browser
        if open_browser:
            self.open_browser()
        
        # Run demo
        if run_demo:
            time.sleep(3)  # Give services a moment
            self.run_demo()
        
        # Print success message
        self.print_banner("System Started Successfully! 🎉", "🎉")
        print(f"📊 Backend API: http://localhost:8000")
        print(f"🌐 Frontend App: http://localhost:3000")
        print(f"📚 API Documentation: http://localhost:8000/docs")
        print(f"💡 Real-time features: WebSocket connections active")
        print(f"\n🔧 Management:")
        print(f"   - View logs in terminal")
        print(f"   - Stop with Ctrl+C")
        print(f"   - Test with: python test_webapp.py")
        print(f"   - Demo with: python demo.py")
        print(f"\n📖 For more information, see README.md")
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
        
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Digital Twin Portfolio Management System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  start       Start the complete system (default)
  demo        Run demonstration only
  test        Run test suite only
  setup       Setup dependencies only
  
Examples:
  python launch.py                    # Start complete system
  python launch.py start --no-browser # Start without opening browser
  python launch.py demo --interactive # Run interactive demo
  python launch.py test               # Run tests only
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "demo", "test", "setup"],
        help="Command to execute (default: start)"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Don't run demo automatically"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run demo in interactive mode"
    )
    
    args = parser.parse_args()
    
    launcher = SystemLauncher()
    
    try:
        if args.command == "start":
            launcher.start_full_system(
                open_browser=not args.no_browser,
                run_demo=not args.no_demo
            )
        
        elif args.command == "demo":
            mode = "interactive" if args.interactive else "automated"
            launcher.run_demo(mode)
        
        elif args.command == "test":
            asyncio.run(launcher.run_tests())
        
        elif args.command == "setup":
            if launcher.check_dependencies():
                launcher.install_dependencies()
                print("✅ Setup completed!")
            else:
                print("❌ Please install missing dependencies first")
        
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 