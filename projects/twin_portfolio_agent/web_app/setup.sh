#!/bin/bash

# Multi-Agent Digital Twin Portfolio Management Web App Setup Script
# This script sets up both the backend and frontend components

set -e  # Exit on any error

echo "🚀 Setting up Multi-Agent Digital Twin Portfolio Web Application..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.12+ is available
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Found Python $PYTHON_VERSION"
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
            print_success "Python version is compatible"
        else
            print_error "Python 3.8+ is required. Please upgrade Python."
            exit 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
}

# Check if Node.js is available
check_nodejs() {
    print_status "Checking Node.js version..."
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Found Node.js $NODE_VERSION"
        if node -e 'process.exit(process.version.slice(1).split(".")[0] >= 16 ? 0 : 1)'; then
            print_success "Node.js version is compatible"
        else
            print_warning "Node.js 16+ is recommended for best compatibility"
        fi
    else
        print_error "Node.js is not installed. Please install Node.js 16+ first."
        exit 1
    fi
}

# Check if UV is available, install if not
check_uv() {
    print_status "Checking UV package manager..."
    if command -v uv &> /dev/null; then
        print_success "UV is already installed"
    else
        print_status "UV not found. Installing UV..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        if command -v uv &> /dev/null; then
            print_success "UV installed successfully"
        else
            print_warning "UV installation may have failed. Falling back to pip."
        fi
    fi
}

# Setup backend
setup_backend() {
    print_status "Setting up backend (FastAPI)..."
    
    cd backend
    
    print_status "Installing Python dependencies..."
    if command -v uv &> /dev/null; then
        uv venv
        source .venv/bin/activate
        uv pip install -r requirements.txt
    else
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
    
    print_success "Backend dependencies installed"
    cd ..
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend (React)..."
    
    cd frontend
    
    print_status "Installing Node.js dependencies..."
    if command -v npm &> /dev/null; then
        npm install
    else
        print_error "npm is not available. Please install Node.js with npm."
        exit 1
    fi
    
    print_success "Frontend dependencies installed"
    cd ..
}

# Create environment files
create_env_files() {
    print_status "Creating environment configuration files..."
    
    # Backend environment
    if [ ! -f "backend/.env" ]; then
        cat > backend/.env << EOF
# FastAPI Backend Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=info

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
EOF
        print_success "Created backend/.env"
    else
        print_warning "backend/.env already exists, skipping..."
    fi
    
    # Frontend environment
    if [ ! -f "frontend/.env" ]; then
        cat > frontend/.env << EOF
# React Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
GENERATE_SOURCEMAP=false
EOF
        print_success "Created frontend/.env"
    else
        print_warning "frontend/.env already exists, skipping..."
    fi
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🔥 Starting FastAPI Backend..."
cd backend
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
python main.py
EOF
    chmod +x start_backend.sh
    
    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "⚛️  Starting React Frontend..."
cd frontend
npm start
EOF
    chmod +x start_frontend.sh
    
    # Combined startup script
    cat > start_all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Multi-Agent Digital Twin Portfolio Web Application..."
echo ""

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start backend in background
echo "🔥 Starting FastAPI Backend..."
./start_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "⚛️  Starting React Frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

echo ""
echo "✅ Services started successfully!"
echo "📊 Backend API: http://localhost:8000"
echo "🌐 Frontend App: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
EOF
    chmod +x start_all.sh
    
    print_success "Created startup scripts"
}

# Create Docker configuration
create_docker_config() {
    print_status "Creating Docker configuration..."
    
    # Backend Dockerfile
    cat > backend/Dockerfile << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    
    # Frontend Dockerfile
    cat > frontend/Dockerfile << 'EOF'
# Build stage
FROM node:16-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
EOF
    
    # Nginx configuration for frontend
    cat > frontend/nginx.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    
    root /usr/share/nginx/html;
    index index.html index.htm;
    
    # Handle client-side routing
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # API proxy
    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket proxy
    location /ws/ {
        proxy_pass http://backend:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
}
EOF
    
    # Docker Compose file
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - FASTAPI_HOST=0.0.0.0
      - FASTAPI_PORT=8000
      - ENVIRONMENT=production
    volumes:
      - ./backend:/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - NODE_ENV=production
    restart: unless-stopped

networks:
  default:
    driver: bridge
EOF
    
    # Docker startup script
    cat > start_docker.sh << 'EOF'
#!/bin/bash
echo "🐳 Starting Multi-Agent Portfolio with Docker..."

# Build and start services
docker-compose up --build

echo "✅ Docker services started!"
echo "🌐 Frontend: http://localhost:3000"
echo "📊 Backend: http://localhost:8000"
EOF
    chmod +x start_docker.sh
    
    print_success "Created Docker configuration"
}

# Main setup process
main() {
    echo "════════════════════════════════════════════════════════════════"
    echo "   Multi-Agent Digital Twin Portfolio Management Setup"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check prerequisites
    check_python
    check_nodejs
    check_uv
    
    echo ""
    echo "📦 Installing dependencies..."
    
    # Setup components
    setup_backend
    setup_frontend
    
    echo ""
    echo "⚙️  Creating configuration files..."
    
    # Create configuration
    create_env_files
    create_startup_scripts
    create_docker_config
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    print_success "Setup completed successfully! 🎉"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "🚀 Quick Start Options:"
    echo ""
    echo "1. Development Mode (separate terminals):"
    echo "   ./start_backend.sh    # Terminal 1"
    echo "   ./start_frontend.sh   # Terminal 2"
    echo ""
    echo "2. Development Mode (single terminal):"
    echo "   ./start_all.sh"
    echo ""
    echo "3. Production Mode (Docker):"
    echo "   ./start_docker.sh"
    echo ""
    echo "📊 Access URLs:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend:  http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo ""
    echo "📚 For more information, see README.md"
    echo ""
}

# Run main function
main "$@" 