"""Secure explanation service for credit policy decisions."""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .config import Config


class ExplanationLevel(Enum):
    """Different levels of explanation detail."""
    CUSTOMER = "customer"           # Customer-facing, no sensitive info
    INTERNAL_STAFF = "internal"     # Internal staff, some details
    ADMIN = "admin"                 # Full access for administrators
    AUDIT = "audit"                 # Compliance and audit purposes


@dataclass
class ExplanationContext:
    """Context for generating explanations."""
    user_role: str
    explanation_level: ExplanationLevel
    regulatory_context: str = "FCRA"  # Fair Credit Reporting Act compliance
    

class SecureExplanationService:
    """
    Service for generating secure explanations that never expose model internals.
    
    This service ensures compliance with regulations like FCRA while protecting
    proprietary algorithms and preventing gaming of the system.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Protected fields that should NEVER be exposed to customers
        self._protected_fields = {
            "model_weights", "model_parameters", "feature_coefficients",
            "risk_score_raw", "confidence_threshold", "internal_scoring",
            "algorithm_details", "model_architecture", "training_data"
        }
        
        # Mapping of internal reasons to customer-friendly explanations
        self._reason_mapping = {
            "credit_score_below_threshold": {
                "customer": "Credit score does not meet our minimum requirements",
                "internal": "Credit score below configured threshold",
                "improvement": "Improve credit score by paying bills on time and reducing credit utilization"
            },
            "high_debt_to_income": {
                "customer": "Debt-to-income ratio exceeds our guidelines", 
                "internal": "DTI ratio above maximum allowed threshold",
                "improvement": "Pay down existing debts to reduce debt-to-income ratio"
            },
            "insufficient_income": {
                "customer": "Income level below our minimum requirements",
                "internal": "Annual income below minimum threshold",
                "improvement": "Document additional income sources or increase overall income"
            },
            "employment_status": {
                "customer": "Employment status does not meet our criteria",
                "internal": "Employment status not in approved categories",
                "improvement": "Ensure stable employment with proper documentation"
            },
            "high_overall_risk": {
                "customer": "Overall risk assessment indicates higher than acceptable risk",
                "internal": "Combined risk factors exceed acceptable thresholds",
                "improvement": "Improve overall financial profile through responsible credit management"
            }
        }
    
    def generate_explanation(
        self, 
        decision_data: Dict[str, Any], 
        context: ExplanationContext
    ) -> Dict[str, Any]:
        """
        Generate explanation based on user role and context.
        
        Args:
            decision_data: Raw decision data (may contain sensitive info)
            context: Explanation context with user role and level
            
        Returns:
            Sanitized explanation appropriate for the user role
        """
        self.logger.info(f"Generating {context.explanation_level.value} explanation")
        
        # Always start with sanitized data
        sanitized_data = self._sanitize_data(decision_data, context.explanation_level)
        
        if context.explanation_level == ExplanationLevel.CUSTOMER:
            return self._generate_customer_explanation(sanitized_data)
        elif context.explanation_level == ExplanationLevel.INTERNAL_STAFF:
            return self._generate_internal_explanation(sanitized_data)
        elif context.explanation_level == ExplanationLevel.ADMIN:
            return self._generate_admin_explanation(sanitized_data)
        elif context.explanation_level == ExplanationLevel.AUDIT:
            return self._generate_audit_explanation(sanitized_data)
        else:
            return self._generate_customer_explanation(sanitized_data)
    
    def _sanitize_data(
        self, 
        data: Dict[str, Any], 
        level: ExplanationLevel
    ) -> Dict[str, Any]:
        """Remove sensitive information based on explanation level."""
        sanitized = {}
        
        for key, value in data.items():
            # Always exclude protected fields from customer explanations
            if level == ExplanationLevel.CUSTOMER and key in self._protected_fields:
                continue
                
            # For internal staff, allow some additional details but still protect core model info
            if level == ExplanationLevel.INTERNAL_STAFF and key in ["model_weights", "model_parameters"]:
                continue
                
            # Only admin and audit can see certain fields
            if key in ["model_architecture", "training_data"] and level not in [ExplanationLevel.ADMIN, ExplanationLevel.AUDIT]:
                continue
                
            sanitized[key] = value
        
        return sanitized
    
    def _generate_customer_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate customer-facing explanation with no sensitive information."""
        approved = data.get("approved", False)
        
        explanation = {
            "decision": "approved" if approved else "declined",
            "application_id": data.get("application_id"),
            "timestamp": data.get("timestamp")
        }
        
        if approved:
            explanation.update({
                "message": "Congratulations! Your loan application has been approved.",
                "next_steps": [
                    "Review your loan terms and conditions",
                    "Sign the loan agreement",
                    "Funds will be disbursed according to your agreement"
                ]
            })
            
            # Only include approved loan details
            if data.get("interest_rate"):
                explanation["interest_rate"] = data["interest_rate"]
            if data.get("loan_term_months"):
                explanation["loan_term_months"] = data["loan_term_months"]
        else:
            # Generate adverse action notice compliant with FCRA
            adverse_actions = self._generate_adverse_actions(data)
            explanation.update({
                "message": "We are unable to approve your loan application at this time.",
                "primary_factors": adverse_actions.get("primary_factors", []),
                "improvement_suggestions": adverse_actions.get("improvements", []),
                "your_rights": self._get_fcra_rights_notice(),
                "reapplication": "You may reapply in the future once you've addressed the factors above."
            })
        
        return explanation
    
    def _generate_internal_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for internal staff with operational details."""
        explanation = self._generate_customer_explanation(data)
        
        # Add internal operational details (but not model secrets)
        explanation["internal_details"] = {
            "confidence_level": data.get("confidence", 0),
            "policy_version": data.get("policy_version", "unknown"),
            "processing_time_ms": data.get("processing_time", 0),
            "review_required": data.get("confidence", 1) < 0.8
        }
        
        return explanation
    
    def _generate_admin_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed explanation for administrators."""
        explanation = self._generate_internal_explanation(data)
        
        # Add administrative details
        explanation["admin_details"] = {
            "rule_evaluation_summary": data.get("rule_evaluation", {}),
            "data_sources_used": data.get("data_sources", []),
            "model_version": data.get("model_version", "unknown"),
            "audit_trail": data.get("audit_trail", [])
        }
        
        return explanation
    
    def _generate_audit_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for compliance and audit purposes."""
        explanation = self._generate_admin_explanation(data)
        
        # Add compliance-specific information
        explanation["compliance_details"] = {
            "regulatory_framework": "FCRA, ECOA, Fair Lending Guidelines",
            "bias_check_results": data.get("bias_check", "passed"),
            "data_retention_policy": "7 years as per regulatory requirements",
            "decision_appeal_process": "Available through customer service"
        }
        
        return explanation
    
    def _generate_adverse_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate FCRA-compliant adverse action reasons."""
        reasons = data.get("decision_reasons", [])
        primary_factors = []
        improvements = []
        
        for reason in reasons:
            if "Failed:" in reason:
                for key, mapping in self._reason_mapping.items():
                    if any(term in reason.lower() for term in key.split("_")):
                        primary_factors.append(mapping["customer"])
                        improvements.append(mapping["improvement"])
                        break
        
        # Ensure we have at least one primary factor (FCRA requirement)
        if not primary_factors:
            primary_factors.append("Application does not meet current underwriting criteria")
            improvements.append("Consider reapplying when your financial situation improves")
        
        return {
            "primary_factors": list(set(primary_factors)),  # Remove duplicates
            "improvements": list(set(improvements))
        }
    
    def _get_fcra_rights_notice(self) -> List[str]:
        """Get FCRA rights notice for adverse actions."""
        return [
            "You have the right to obtain a free copy of your credit report from the credit reporting agency",
            "You have the right to dispute any inaccurate or incomplete information in your credit report",
            "Identity theft victims and active duty military personnel have additional rights"
        ]
    
    def validate_explanation_security(self, explanation: Dict[str, Any]) -> bool:
        """Validate that explanation doesn't contain sensitive information."""
        explanation_str = str(explanation).lower()
        
        # Check for protected terms
        protected_terms = [
            "weight", "coefficient", "parameter", "threshold", "algorithm",
            "model_", "score_raw", "internal_", "training_", "feature_importance"
        ]
        
        for term in protected_terms:
            if term in explanation_str:
                self.logger.warning(f"Potential sensitive information detected: {term}")
                return False
        
        return True 