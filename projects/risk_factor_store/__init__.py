"""
Credit Risk Factor Store

A comprehensive system for ingesting, processing, and serving credit risk factors
for real-time credit risk assessment and prediction.
"""

__version__ = "1.0.0"
__author__ = "Hemingway Data Journalist"

from .core.config import settings
from .core.database import get_db
from .models.risk_factors import RiskFactor

__all__ = [
    "settings",
    "get_db", 
    "RiskFactor"
] 