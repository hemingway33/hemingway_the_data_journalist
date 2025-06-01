"""
Portfolio analytics API endpoints.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_portfolios():
    """List portfolios."""
    return {"message": "Portfolio analytics endpoint"}

@router.get("/{portfolio_id}/report")
async def get_portfolio_report(portfolio_id: str):
    """Get portfolio risk report."""
    return {"portfolio_id": portfolio_id} 