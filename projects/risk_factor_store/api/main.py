"""
Main FastAPI application for the Credit Risk Factor Store.
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import structlog
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db, check_database_connection, check_redis_connection
from ..models.risk_factors import RiskFactor, RiskFactorValue, CreditEntity
from ..risk_engine.credit_scoring import CreditScoringEngine
from ..risk_engine.portfolio_analytics import PortfolioAnalyticsEngine
from .schemas import *
from .endpoints import risk_factors, credit_scoring, portfolio_analytics

logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A comprehensive credit risk factor database system for real-time risk assessment",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(risk_factors.router, prefix="/api/v1/risk-factors", tags=["Risk Factors"])
app.include_router(credit_scoring.router, prefix="/api/v1/credit-scoring", tags=["Credit Scoring"])
app.include_router(portfolio_analytics.router, prefix="/api/v1/portfolio-analytics", tags=["Portfolio Analytics"])


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Credit Risk Factor Store", version=settings.app_version)
    
    # Check database connection
    db_healthy = await check_database_connection()
    if not db_healthy:
        logger.error("Database connection failed")
        raise RuntimeError("Database connection failed")
    
    # Check Redis connection
    redis_healthy = await check_redis_connection()
    if not redis_healthy:
        logger.warning("Redis connection failed - caching disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down Credit Risk Factor Store")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Credit Risk Factor Store",
        "version": settings.app_version,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_healthy = await check_database_connection()
    redis_healthy = await check_redis_connection()
    
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version,
        "services": {
            "database": "healthy" if db_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "unhealthy"
        }
    }


@app.get("/api/v1/info")
async def app_info():
    """Application information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": "development" if settings.debug else "production",
        "features": [
            "Real-time risk factor ingestion",
            "Individual credit scoring",
            "Portfolio risk analytics",
            "Stress testing",
            "Data quality monitoring"
        ]
    }


@app.get("/api/v1/stats")
async def system_stats(db: Session = Depends(get_db)):
    """System statistics."""
    try:
        # Count risk factors
        risk_factor_count = db.query(RiskFactor).filter(
            RiskFactor.is_active == True
        ).count()
        
        # Count risk factor values (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_values_count = db.query(RiskFactorValue).filter(
            RiskFactorValue.date >= thirty_days_ago
        ).count()
        
        # Count credit entities
        entity_count = db.query(CreditEntity).filter(
            CreditEntity.is_active == True
        ).count()
        
        return {
            "risk_factors": {
                "total_active": risk_factor_count,
                "recent_data_points": recent_values_count
            },
            "entities": {
                "total_active": entity_count
            },
            "data_freshness": {
                "last_30_days": recent_values_count
            }
        }
        
    except Exception as e:
        logger.error("Failed to get system stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error("Internal server error", error=str(exc))
    return {
        "error": "Internal Server Error",
        "message": "An internal error occurred",
        "status_code": 500
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "projects.risk_factor_store.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    ) 