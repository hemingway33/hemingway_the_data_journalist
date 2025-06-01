"""Custom exceptions for the Credit Policy Agent."""

from typing import Optional, Any, Dict


class CreditPolicyError(Exception):
    """Base exception for all Credit Policy Agent errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class PolicyValidationError(CreditPolicyError):
    """Raised when policy validation fails."""
    pass


class ModelError(CreditPolicyError):
    """Raised when model operations fail."""
    pass


class DataSourceError(CreditPolicyError):
    """Raised when data source operations fail."""
    pass


class OptimizationError(CreditPolicyError):
    """Raised when optimization operations fail."""
    pass


class ClientInteractionError(CreditPolicyError):
    """Raised when client interaction operations fail."""
    pass


class ConfigurationError(CreditPolicyError):
    """Raised when configuration is invalid."""
    pass


class AuthenticationError(CreditPolicyError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(CreditPolicyError):
    """Raised when authorization fails."""
    pass 