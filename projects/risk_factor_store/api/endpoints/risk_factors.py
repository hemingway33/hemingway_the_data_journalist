"""
Risk factors API endpoints.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_risk_factors():
    """List risk factors."""
    return {"message": "Risk factors endpoint"}

@router.get("/{factor_id}")
async def get_risk_factor(factor_id: str):
    """Get specific risk factor."""
    return {"factor_id": factor_id} 