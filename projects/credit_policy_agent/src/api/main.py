"""Main FastAPI application for the Credit Policy Agent."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import get_config
from ..core.agent import CreditPolicyAgent


# Global agent instance
agent: CreditPolicyAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent
    
    # Startup
    config = get_config()
    agent = CreditPolicyAgent(config)
    
    try:
        await agent.start()
        logging.info("Credit Policy Agent started successfully")
        yield
    finally:
        # Shutdown
        if agent:
            await agent.stop()
        logging.info("Credit Policy Agent stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    
    app = FastAPI(
        title="Credit Policy Agent API",
        description="Intelligent credit policy management and loan evaluation system",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers - import here to avoid circular imports
    from .routes import router
    app.include_router(router, prefix="/api/v1")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Credit Policy Agent API",
            "version": "0.1.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        global agent
        
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        try:
            status = await agent.get_agent_status()
            return {
                "status": "healthy" if status["is_running"] else "unhealthy",
                "agent_status": status
            }
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
    
    return app


# Create the application instance
app = create_app()


def get_agent() -> CreditPolicyAgent:
    """Get the global agent instance."""
    global agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent 