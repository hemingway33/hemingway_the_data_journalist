"""Policy validator for rule validation."""

from typing import Dict, Any, List
from .schemas import PolicySchema, RuleSchema


class PolicyValidator:
    """Validator for policy rules and configurations."""
    
    @staticmethod
    def validate_policy(policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete policy."""
        try:
            policy = PolicySchema(**policy_data)
            return {
                "valid": True,
                "errors": [],
                "warnings": []
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": []
            }
    
    @staticmethod
    def validate_rule(rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single rule."""
        try:
            rule = RuleSchema(**rule_data)
            return {
                "valid": True,
                "errors": [],
                "warnings": []
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": []
            } 