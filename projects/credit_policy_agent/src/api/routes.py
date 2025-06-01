"""API routes for the Credit Policy Agent."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..core.agent import LoanApplication, PolicyDecision
from .main import get_agent
from ..core.explanation import SecureExplanationService, ExplanationLevel, ExplanationContext


router = APIRouter()


class LoanApplicationRequest(BaseModel):
    """Request model for loan application."""
    application_id: str = Field(..., description="Unique application identifier")
    applicant_id: str = Field(..., description="Unique applicant identifier")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    credit_score: Optional[int] = Field(None, ge=300, le=850, description="Credit score")
    income: Optional[float] = Field(None, ge=0, description="Annual income")
    employment_status: Optional[str] = Field(None, description="Employment status")
    debt_to_income_ratio: Optional[float] = Field(None, ge=0, le=1, description="Debt-to-income ratio")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional application data")


class PolicyDecisionResponse(BaseModel):
    """Response model for policy decision."""
    application_id: str
    approved: bool
    confidence: float
    interest_rate: Optional[float] = None
    loan_term_months: Optional[int] = None
    decision_reasons: List[str] = []
    risk_score: Optional[float] = None
    required_information: List[str] = []
    interview_questions: List[str] = []
    timestamp: datetime


class PolicyUpdateRequest(BaseModel):
    """Request model for policy updates."""
    rules: Dict[str, Any] = Field(..., description="New or updated policy rules")
    reason: Optional[str] = Field(None, description="Reason for the update")


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    is_running: bool
    last_optimization: Optional[datetime] = None
    policy_version: str
    environment: str
    optimization_interval: int
    auto_retrain_enabled: bool


@router.post("/applications/evaluate", response_model=PolicyDecisionResponse)
async def evaluate_application(
    request: LoanApplicationRequest,
    agent = Depends(get_agent)
) -> PolicyDecisionResponse:
    """
    Evaluate a loan application and make a lending decision.
    """
    try:
        # Convert request to LoanApplication
        application = LoanApplication(
            application_id=request.application_id,
            applicant_id=request.applicant_id,
            loan_amount=request.loan_amount,
            loan_purpose=request.loan_purpose,
            credit_score=request.credit_score,
            income=request.income,
            employment_status=request.employment_status,
            debt_to_income_ratio=request.debt_to_income_ratio,
            additional_data=request.additional_data or {}
        )
        
        # Evaluate application
        decision = await agent.evaluate_application(application)
        
        # Convert to response model
        return PolicyDecisionResponse(
            application_id=decision.application_id,
            approved=decision.approved,
            confidence=decision.confidence,
            interest_rate=decision.interest_rate,
            loan_term_months=decision.loan_term_months,
            decision_reasons=decision.decision_reasons,
            risk_score=decision.risk_score,
            required_information=decision.required_information,
            interview_questions=decision.interview_questions,
            timestamp=decision.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Application evaluation failed: {str(e)}")


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status(agent = Depends(get_agent)) -> AgentStatusResponse:
    """
    Get current agent status and configuration.
    """
    try:
        status = await agent.get_agent_status()
        
        return AgentStatusResponse(
            is_running=status["is_running"],
            last_optimization=status.get("last_optimization"),
            policy_version=status["policy_version"],
            environment=status["config"]["environment"],
            optimization_interval=status["config"]["optimization_interval"],
            auto_retrain_enabled=status["config"]["auto_retrain_enabled"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@router.get("/policies/current")
async def get_current_policy(agent = Depends(get_agent)) -> Dict[str, Any]:
    """
    Get current policy rules and configuration.
    """
    try:
        return await agent.policy_engine.get_rule_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current policy: {str(e)}")


@router.put("/policies/rules")
async def update_policy_rules(
    request: PolicyUpdateRequest,
    agent = Depends(get_agent)
) -> Dict[str, Any]:
    """
    Update policy rules.
    """
    try:
        success = await agent.update_policy_rules(request.rules)
        
        if success:
            return {
                "success": True,
                "message": "Policy rules updated successfully",
                "new_version": await agent.policy_engine.get_current_version()
            }
        else:
            return {
                "success": False,
                "message": "Policy rules update rejected due to negative impact"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update policy rules: {str(e)}")


@router.post("/policies/validate")
async def validate_policy_rules(
    request: PolicyUpdateRequest,
    agent = Depends(get_agent)
) -> Dict[str, Any]:
    """
    Validate policy rules without applying them.
    """
    try:
        validation_result = await agent.policy_engine.validate_rules(request.rules)
        
        return {
            "valid": validation_result.valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate policy rules: {str(e)}")


@router.get("/metrics/summary")
async def get_metrics_summary(agent = Depends(get_agent)) -> Dict[str, Any]:
    """
    Get performance metrics summary.
    """
    try:
        # This would return actual metrics from the metrics collector
        return {
            "total_applications": 0,
            "approval_rate": 0.0,
            "average_risk_score": 0.0,
            "policy_updates": 0,
            "model_performance": {},
            "data_source_status": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/optimization/trigger")
async def trigger_optimization(agent = Depends(get_agent)) -> Dict[str, Any]:
    """
    Manually trigger a policy optimization cycle.
    """
    try:
        # This would trigger an immediate optimization
        return {
            "success": True,
            "message": "Optimization cycle triggered",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger optimization: {str(e)}")


@router.get("/models/performance")
async def get_model_performance(agent = Depends(get_agent)) -> Dict[str, Any]:
    """
    Get model performance metrics.
    """
    try:
        # This would return actual model performance data
        return {
            "models": {},
            "overall_accuracy": 0.0,
            "last_evaluation": None,
            "retraining_needed": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.get("/data-sources/status")
async def get_data_sources_status(agent = Depends(get_agent)) -> Dict[str, Any]:
    """
    Get data sources status and quality metrics.
    """
    try:
        # This would return actual data source information
        return {
            "sources": {},
            "total_sources": 0,
            "active_sources": 0,
            "average_quality": 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data sources status: {str(e)}")


@router.get("/applications/{application_id}/explanation")
async def get_customer_explanation(
    application_id: str,
    agent = Depends(get_agent)
) -> Dict[str, Any]:
    """
    Get customer-facing explanation for a loan decision.
    This endpoint never exposes model internals or sensitive information.
    """
    try:
        # In a real implementation, you would fetch the decision from database
        # For demo purposes, return a secure explanation format
        
        explanation_service = SecureExplanationService(agent.config)
        
        # Mock decision data (would come from database in real implementation)
        mock_decision_data = {
            "application_id": application_id,
            "approved": False,
            "decision_reasons": [
                "Failed: Credit score below minimum threshold",
                "Failed: Debt-to-income ratio exceeds guidelines"
            ],
            "timestamp": datetime.utcnow(),
            # These would be protected from customer view
            "confidence": 0.65,
            "risk_score": 0.78,
            "model_weights": {"feature_1": 0.45, "feature_2": -0.32},  # PROTECTED
            "internal_scoring": {"raw_score": 0.234}  # PROTECTED
        }
        
        context = ExplanationContext(
            user_role="customer",
            explanation_level=ExplanationLevel.CUSTOMER
        )
        
        explanation = explanation_service.generate_explanation(
            mock_decision_data, 
            context
        )
        
        # Validate that no sensitive info leaked
        is_secure = explanation_service.validate_explanation_security(explanation)
        if not is_secure:
            raise HTTPException(
                status_code=500, 
                detail="Security validation failed - explanation contains sensitive information"
            )
        
        return explanation
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate customer explanation: {str(e)}"
        )


@router.get("/applications/{application_id}/explanation/internal")
async def get_internal_explanation(
    application_id: str,
    user_role: str = "internal_staff",
    agent = Depends(get_agent)
) -> Dict[str, Any]:
    """
    Get internal explanation for staff use.
    Contains operational details but still protects core model secrets.
    Requires proper authentication in production.
    """
    try:
        # In production, validate user permissions here
        if user_role not in ["internal_staff", "admin", "audit"]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        explanation_service = SecureExplanationService(agent.config)
        
        # Mock decision data
        mock_decision_data = {
            "application_id": application_id,
            "approved": False,
            "confidence": 0.65,
            "policy_version": "1.0.0",
            "processing_time": 145,
            "decision_reasons": [
                "Failed: Credit score below minimum threshold",
                "Failed: Debt-to-income ratio exceeds guidelines"
            ],
            "rule_evaluation": {"total_rules": 5, "passed": 3, "failed": 2},
            "data_sources": ["bureau_1", "bureau_2", "alternative_data"],
            "model_version": "v2.1.0",
            "timestamp": datetime.utcnow(),
            # Still protected even for internal users
            "model_weights": {"feature_1": 0.45},  # PROTECTED
            "training_data": {"dataset_id": "train_2024"}  # PROTECTED
        }
        
        # Map role to explanation level
        level_mapping = {
            "internal_staff": ExplanationLevel.INTERNAL_STAFF,
            "admin": ExplanationLevel.ADMIN,
            "audit": ExplanationLevel.AUDIT
        }
        
        context = ExplanationContext(
            user_role=user_role,
            explanation_level=level_mapping.get(user_role, ExplanationLevel.INTERNAL_STAFF)
        )
        
        explanation = explanation_service.generate_explanation(
            mock_decision_data,
            context
        )
        
        return explanation
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate internal explanation: {str(e)}"
        )


@router.post("/explanations/validate-security")
async def validate_explanation_security(
    explanation_data: Dict[str, Any],
    agent = Depends(get_agent)
) -> Dict[str, Any]:
    """
    Validate that an explanation doesn't contain sensitive model information.
    Used for testing and quality assurance.
    """
    try:
        explanation_service = SecureExplanationService(agent.config)
        
        is_secure = explanation_service.validate_explanation_security(explanation_data)
        
        return {
            "is_secure": is_secure,
            "message": "Explanation is secure" if is_secure else "Explanation contains sensitive information",
            "validation_timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate explanation security: {str(e)}"
        )


@router.get("/policies/explanation-guidelines")
async def get_explanation_guidelines() -> Dict[str, Any]:
    """
    Get guidelines for explanation generation and security policies.
    """
    return {
        "explanation_levels": {
            "customer": "Customer-facing explanations with no sensitive information",
            "internal_staff": "Internal operational details without model secrets",
            "admin": "Administrative details for system management",
            "audit": "Full compliance and audit information"
        },
        "protected_information": [
            "Model weights and parameters",
            "Training data details",
            "Algorithm implementation specifics",
            "Internal scoring coefficients",
            "Feature importance values",
            "Threshold configurations"
        ],
        "compliance_frameworks": [
            "Fair Credit Reporting Act (FCRA)",
            "Equal Credit Opportunity Act (ECOA)",
            "Fair Lending Guidelines",
            "General Data Protection Regulation (GDPR)"
        ],
        "security_principles": [
            "Never expose model internals to customers",
            "Provide actionable feedback without revealing system vulnerabilities",
            "Maintain audit trails for all explanations",
            "Validate all explanations for sensitive information leakage"
        ]
    } 