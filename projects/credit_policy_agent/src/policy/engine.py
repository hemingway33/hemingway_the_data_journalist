"""Policy Engine for credit policy management and evaluation."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.config import Config
from ..core.exceptions import PolicyValidationError


class RuleType(Enum):
    """Types of policy rules."""
    CREDIT_SCORE = "credit_score"
    INCOME = "income"
    DEBT_TO_INCOME = "debt_to_income"
    EMPLOYMENT = "employment"
    LOAN_AMOUNT = "loan_amount"
    CUSTOM = "custom"


class Operator(Enum):
    """Comparison operators for rules."""
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


@dataclass
class PolicyRule:
    """Individual policy rule definition."""
    rule_id: str
    rule_type: RuleType
    field: str
    operator: Operator
    value: Union[float, int, str, List[Any]]
    weight: float = 1.0
    description: str = ""
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate this rule against application data."""
        if not self.active:
            return True
        
        field_value = self._get_field_value(data, self.field)
        if field_value is None:
            return False
        
        return self._apply_operator(field_value, self.operator, self.value)
    
    def _get_field_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get field value from nested data structure."""
        keys = field.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return None
        
        return current
    
    def _apply_operator(self, field_value: Any, operator: Operator, rule_value: Any) -> bool:
        """Apply comparison operator."""
        try:
            if operator == Operator.GREATER_THAN:
                return float(field_value) > float(rule_value)
            elif operator == Operator.GREATER_EQUAL:
                return float(field_value) >= float(rule_value)
            elif operator == Operator.LESS_THAN:
                return float(field_value) < float(rule_value)
            elif operator == Operator.LESS_EQUAL:
                return float(field_value) <= float(rule_value)
            elif operator == Operator.EQUAL:
                return field_value == rule_value
            elif operator == Operator.NOT_EQUAL:
                return field_value != rule_value
            elif operator == Operator.IN:
                return field_value in rule_value
            elif operator == Operator.NOT_IN:
                return field_value not in rule_value
            elif operator == Operator.CONTAINS:
                return str(rule_value).lower() in str(field_value).lower()
            else:
                return False
        except (ValueError, TypeError):
            return False


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    approved: bool
    confidence: float
    interest_rate: Optional[float] = None
    loan_term_months: Optional[int] = None
    reasons: List[str] = None
    failed_rules: List[str] = None
    passed_rules: List[str] = None
    risk_tier: str = "medium"
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.failed_rules is None:
            self.failed_rules = []
        if self.passed_rules is None:
            self.passed_rules = []


@dataclass
class ValidationResult:
    """Result of policy validation."""
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class PolicyEngine:
    """
    Core policy engine for managing and evaluating credit policies.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Policy state
        self.current_rules: Dict[str, PolicyRule] = {}
        self.policy_version = "1.0.0"
        self.policy_history: List[Dict[str, Any]] = []
        
        # Default interest rate tiers
        self.interest_rate_tiers = {
            "low_risk": 0.03,      # 3%
            "medium_risk": 0.06,   # 6%
            "high_risk": 0.12,     # 12%
        }
        
        # Default loan terms (months)
        self.loan_terms = {
            "short": 12,
            "medium": 36,
            "long": 60
        }
    
    async def initialize(self) -> None:
        """Initialize the policy engine with default rules."""
        self.logger.info("Initializing Policy Engine")
        
        # Load default credit policy rules
        await self._load_default_rules()
        
        self.logger.info(f"Policy Engine initialized with {len(self.current_rules)} rules")
    
    async def shutdown(self) -> None:
        """Shutdown the policy engine."""
        self.logger.info("Shutting down Policy Engine")
        # Save current state if needed
        await self._save_policy_state()
    
    async def evaluate_application(
        self, 
        application_data: Dict[str, Any], 
        risk_assessment: Any
    ) -> PolicyResult:
        """
        Evaluate a loan application against current policy rules.
        
        Args:
            application_data: Enhanced application data
            risk_assessment: Risk assessment from models
            
        Returns:
            PolicyResult: Evaluation result with decision and reasoning
        """
        self.logger.debug("Evaluating application against policy rules")
        
        # Combine application and risk data
        evaluation_data = {
            **application_data,
            "risk_score": getattr(risk_assessment, 'risk_score', 0.5),
            "risk_tier": getattr(risk_assessment, 'risk_tier', 'medium')
        }
        
        passed_rules = []
        failed_rules = []
        reasons = []
        total_weight = 0
        passed_weight = 0
        
        # Evaluate each rule
        for rule_id, rule in self.current_rules.items():
            if not rule.active:
                continue
                
            total_weight += rule.weight
            
            if rule.evaluate(evaluation_data):
                passed_rules.append(rule_id)
                passed_weight += rule.weight
                reasons.append(f"Passed: {rule.description}")
            else:
                failed_rules.append(rule_id)
                reasons.append(f"Failed: {rule.description}")
        
        # Calculate confidence and approval
        confidence = passed_weight / total_weight if total_weight > 0 else 0
        approved = confidence >= 0.7  # 70% threshold for approval
        
        # Determine risk tier
        risk_score = evaluation_data.get("risk_score", 0.5)
        if risk_score < 0.3:
            risk_tier = "low_risk"
        elif risk_score < 0.7:
            risk_tier = "medium_risk"
        else:
            risk_tier = "high_risk"
        
        # Assign interest rate and loan term
        interest_rate = self.interest_rate_tiers.get(risk_tier, 0.08)
        loan_term = self._determine_loan_term(evaluation_data, risk_tier)
        
        return PolicyResult(
            approved=approved,
            confidence=confidence,
            interest_rate=interest_rate,
            loan_term_months=loan_term,
            reasons=reasons,
            failed_rules=failed_rules,
            passed_rules=passed_rules,
            risk_tier=risk_tier
        )
    
    async def update_rules(self, new_rules: Dict[str, Any]) -> None:
        """Update policy rules."""
        self.logger.info("Updating policy rules")
        
        # Save current version to history
        self.policy_history.append({
            "version": self.policy_version,
            "rules": {rule_id: asdict(rule) for rule_id, rule in self.current_rules.items()},
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update rules
        for rule_id, rule_data in new_rules.items():
            if rule_id in self.current_rules:
                # Update existing rule
                self._update_rule(rule_id, rule_data)
            else:
                # Add new rule
                self._add_rule(rule_id, rule_data)
        
        # Increment version
        version_parts = self.policy_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        self.policy_version = '.'.join(version_parts)
        
        self.logger.info(f"Policy updated to version {self.policy_version}")
    
    async def validate_rules(self, rules: Dict[str, Any]) -> ValidationResult:
        """Validate policy rules before applying."""
        errors = []
        warnings = []
        
        for rule_id, rule_data in rules.items():
            try:
                # Validate rule structure
                if not isinstance(rule_data, dict):
                    errors.append(f"Rule {rule_id}: Must be a dictionary")
                    continue
                
                required_fields = ['rule_type', 'field', 'operator', 'value']
                for field in required_fields:
                    if field not in rule_data:
                        errors.append(f"Rule {rule_id}: Missing required field '{field}'")
                
                # Validate rule type
                if 'rule_type' in rule_data:
                    try:
                        RuleType(rule_data['rule_type'])
                    except ValueError:
                        errors.append(f"Rule {rule_id}: Invalid rule_type '{rule_data['rule_type']}'")
                
                # Validate operator
                if 'operator' in rule_data:
                    try:
                        Operator(rule_data['operator'])
                    except ValueError:
                        errors.append(f"Rule {rule_id}: Invalid operator '{rule_data['operator']}'")
                
                # Validate weight
                if 'weight' in rule_data:
                    weight = rule_data['weight']
                    if not isinstance(weight, (int, float)) or weight < 0:
                        errors.append(f"Rule {rule_id}: Weight must be a non-negative number")
                
            except Exception as e:
                errors.append(f"Rule {rule_id}: Validation error - {str(e)}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    async def get_current_version(self) -> str:
        """Get current policy version."""
        return self.policy_version
    
    async def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of current rules."""
        return {
            "version": self.policy_version,
            "total_rules": len(self.current_rules),
            "active_rules": len([r for r in self.current_rules.values() if r.active]),
            "rules": {
                rule_id: {
                    "type": rule.rule_type.value,
                    "field": rule.field,
                    "operator": rule.operator.value,
                    "weight": rule.weight,
                    "active": rule.active,
                    "description": rule.description
                }
                for rule_id, rule in self.current_rules.items()
            }
        }
    
    def _determine_loan_term(self, evaluation_data: Dict[str, Any], risk_tier: str) -> int:
        """Determine appropriate loan term based on risk and amount."""
        application = evaluation_data.get("application")
        loan_amount = getattr(application, 'loan_amount', 10000) if application else 10000
        
        if risk_tier == "low_risk":
            if loan_amount > 50000:
                return self.loan_terms["long"]
            else:
                return self.loan_terms["medium"]
        elif risk_tier == "medium_risk":
            return self.loan_terms["medium"]
        else:
            return self.loan_terms["short"]
    
    def _update_rule(self, rule_id: str, rule_data: Dict[str, Any]) -> None:
        """Update an existing rule."""
        current_rule = self.current_rules[rule_id]
        
        # Update fields
        for field, value in rule_data.items():
            if field == 'rule_type':
                current_rule.rule_type = RuleType(value)
            elif field == 'operator':
                current_rule.operator = Operator(value)
            elif hasattr(current_rule, field):
                setattr(current_rule, field, value)
    
    def _add_rule(self, rule_id: str, rule_data: Dict[str, Any]) -> None:
        """Add a new rule."""
        rule = PolicyRule(
            rule_id=rule_id,
            rule_type=RuleType(rule_data['rule_type']),
            field=rule_data['field'],
            operator=Operator(rule_data['operator']),
            value=rule_data['value'],
            weight=rule_data.get('weight', 1.0),
            description=rule_data.get('description', ''),
            active=rule_data.get('active', True)
        )
        self.current_rules[rule_id] = rule
    
    async def _load_default_rules(self) -> None:
        """Load default credit policy rules."""
        default_rules = {
            "min_credit_score": PolicyRule(
                rule_id="min_credit_score",
                rule_type=RuleType.CREDIT_SCORE,
                field="application.credit_score",
                operator=Operator.GREATER_EQUAL,
                value=580,
                weight=2.0,
                description="Minimum credit score of 580 required"
            ),
            "max_debt_to_income": PolicyRule(
                rule_id="max_debt_to_income",
                rule_type=RuleType.DEBT_TO_INCOME,
                field="application.debt_to_income_ratio",
                operator=Operator.LESS_EQUAL,
                value=0.43,
                weight=1.5,
                description="Debt-to-income ratio must be 43% or less"
            ),
            "min_income": PolicyRule(
                rule_id="min_income",
                rule_type=RuleType.INCOME,
                field="application.income",
                operator=Operator.GREATER_EQUAL,
                value=25000,
                weight=1.0,
                description="Minimum annual income of $25,000"
            ),
            "employment_status": PolicyRule(
                rule_id="employment_status",
                rule_type=RuleType.EMPLOYMENT,
                field="application.employment_status",
                operator=Operator.IN,
                value=["employed", "self-employed"],
                weight=1.0,
                description="Must be employed or self-employed"
            ),
            "max_risk_score": PolicyRule(
                rule_id="max_risk_score",
                rule_type=RuleType.CUSTOM,
                field="risk_score",
                operator=Operator.LESS_EQUAL,
                value=0.8,
                weight=2.0,
                description="Risk score must be 80% or less"
            )
        }
        
        self.current_rules = default_rules
    
    async def _save_policy_state(self) -> None:
        """Save current policy state."""
        # In a real implementation, this would save to a database
        pass 