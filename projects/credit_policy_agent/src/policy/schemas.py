"""Policy schemas for validation and serialization."""

from typing import Dict, Any, List, Union, Optional
from pydantic import BaseModel, Field
from enum import Enum


class RuleTypeEnum(str, Enum):
    """Rule type enumeration."""
    CREDIT_SCORE = "credit_score"
    INCOME = "income"
    DEBT_TO_INCOME = "debt_to_income"
    EMPLOYMENT = "employment"
    LOAN_AMOUNT = "loan_amount"
    CUSTOM = "custom"


class OperatorEnum(str, Enum):
    """Operator enumeration."""
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class RuleSchema(BaseModel):
    """Schema for policy rule validation."""
    rule_id: str = Field(..., description="Unique rule identifier")
    rule_type: RuleTypeEnum = Field(..., description="Type of rule")
    field: str = Field(..., description="Field to evaluate")
    operator: OperatorEnum = Field(..., description="Comparison operator")
    value: Union[float, int, str, List[Any]] = Field(..., description="Value to compare against")
    weight: float = Field(1.0, ge=0, description="Rule weight")
    description: str = Field("", description="Rule description")
    active: bool = Field(True, description="Whether rule is active")


class PolicySchema(BaseModel):
    """Schema for policy validation."""
    version: str = Field(..., description="Policy version")
    rules: Dict[str, RuleSchema] = Field(..., description="Policy rules")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata") 