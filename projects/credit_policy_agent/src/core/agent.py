"""Main Credit Policy Agent class."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .config import Config, get_config
from .exceptions import CreditPolicyError
from ..policy.engine import PolicyEngine
from ..optimization.optimizer import OptimizationEngine
from ..models.manager import ModelManager
from ..data_sources.manager import DataSourceManager
from ..client_interaction.manager import ClientInteractionManager
from ..monitoring.metrics import MetricsCollector


@dataclass
class LoanApplication:
    """Loan application data structure."""
    application_id: str
    applicant_id: str
    loan_amount: float
    loan_purpose: str
    credit_score: Optional[int] = None
    income: Optional[float] = None
    employment_status: Optional[str] = None
    debt_to_income_ratio: Optional[float] = None
    application_date: datetime = None
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.application_date is None:
            self.application_date = datetime.utcnow()
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class PolicyDecision:
    """Policy decision result."""
    application_id: str
    approved: bool
    confidence: float
    interest_rate: Optional[float] = None
    loan_term_months: Optional[int] = None
    decision_reasons: List[str] = None
    risk_score: Optional[float] = None
    required_information: List[str] = None
    interview_questions: List[str] = None
    timestamp: datetime = None
    
    # New explanation fields
    customer_explanation: str = ""
    adverse_action_reasons: List[str] = None
    improvement_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.decision_reasons is None:
            self.decision_reasons = []
        if self.required_information is None:
            self.required_information = []
        if self.interview_questions is None:
            self.interview_questions = []
        if self.adverse_action_reasons is None:
            self.adverse_action_reasons = []
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []
    
    def get_customer_explanation(self) -> str:
        """Get customer-friendly explanation without revealing model details."""
        if self.approved:
            return f"Congratulations! Your loan application has been approved. We found your application meets our lending criteria with a confidence level that allows us to offer you competitive terms."
        else:
            explanation = "We're unable to approve your loan application at this time. "
            if self.adverse_action_reasons:
                explanation += "The primary factors in this decision were: " + "; ".join(self.adverse_action_reasons) + ". "
            if self.improvement_suggestions:
                explanation += "To improve your chances of approval in the future, consider: " + "; ".join(self.improvement_suggestions) + "."
            return explanation
    
    def get_internal_explanation(self) -> Dict[str, Any]:
        """Get detailed explanation for internal use only (never expose to customers)."""
        return {
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "decision_reasons": self.decision_reasons,
            "policy_evaluation_details": "PROTECTED - Internal use only"
        }


class CreditPolicyAgent:
    """
    Main Credit Policy Agent that orchestrates all components to make intelligent
    lending decisions and continuously optimize credit policies.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.policy_engine = PolicyEngine(self.config)
        self.optimization_engine = OptimizationEngine(self.config)
        self.model_manager = ModelManager(self.config)
        self.data_source_manager = DataSourceManager(self.config)
        self.client_interaction_manager = ClientInteractionManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        
        # Agent state
        self._is_running = False
        self._last_optimization = None
        self._optimization_task = None
    
    async def start(self) -> None:
        """Start the credit policy agent."""
        self.logger.info("Starting Credit Policy Agent")
        
        try:
            # Initialize all components
            await self._initialize_components()
            
            # Start background tasks
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            
            self._is_running = True
            self.logger.info("Credit Policy Agent started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Credit Policy Agent: {e}")
            raise CreditPolicyError(f"Agent startup failed: {e}")
    
    async def stop(self) -> None:
        """Stop the credit policy agent."""
        self.logger.info("Stopping Credit Policy Agent")
        
        self._is_running = False
        
        # Cancel background tasks
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components
        await self._shutdown_components()
        
        self.logger.info("Credit Policy Agent stopped")
    
    async def evaluate_application(self, application: LoanApplication) -> PolicyDecision:
        """
        Evaluate a loan application and make a lending decision.
        
        Args:
            application: The loan application to evaluate
            
        Returns:
            PolicyDecision: The lending decision with reasoning
        """
        self.logger.info(f"Evaluating application {application.application_id}")
        
        try:
            # Collect additional data if needed
            enhanced_data = await self._collect_additional_data(application)
            
            # Get risk assessment from models
            risk_assessment = await self.model_manager.assess_risk(enhanced_data)
            
            # Apply current policy rules
            policy_result = await self.policy_engine.evaluate_application(
                enhanced_data, risk_assessment
            )
            
            # Generate interview questions if needed
            interview_questions = []
            required_info = []
            
            if not policy_result.approved and policy_result.confidence < 0.8:
                # Generate targeted questions to improve decision confidence
                interview_questions = await self.client_interaction_manager.generate_interview_questions(
                    application, risk_assessment
                )
                
                # Identify missing information
                required_info = await self._identify_required_information(
                    application, risk_assessment
                )
            
            # Create decision
            decision = PolicyDecision(
                application_id=application.application_id,
                approved=policy_result.approved,
                confidence=policy_result.confidence,
                interest_rate=policy_result.interest_rate,
                loan_term_months=policy_result.loan_term_months,
                decision_reasons=policy_result.reasons,
                risk_score=risk_assessment.risk_score,
                required_information=required_info,
                interview_questions=interview_questions
            )
            
            # Generate customer-safe explanation
            await self._generate_customer_explanation(decision, application)
            
            # Record metrics
            await self.metrics_collector.record_decision(decision, application)
            
            self.logger.info(
                f"Application {application.application_id} "
                f"{'approved' if decision.approved else 'rejected'} "
                f"with confidence {decision.confidence:.2f}"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error evaluating application {application.application_id}: {e}")
            raise CreditPolicyError(f"Application evaluation failed: {e}")
    
    async def update_policy_rules(self, new_rules: Dict[str, Any]) -> bool:
        """
        Update policy rules based on optimization results or manual input.
        
        Args:
            new_rules: New policy rules to apply
            
        Returns:
            bool: True if update was successful
        """
        self.logger.info("Updating policy rules")
        
        try:
            # Validate new rules
            validation_result = await self.policy_engine.validate_rules(new_rules)
            if not validation_result.valid:
                self.logger.warning(f"Policy validation failed: {validation_result.errors}")
                return False
            
            # Simulate impact of new rules
            impact_analysis = await self.optimization_engine.simulate_policy_impact(new_rules)
            
            # Apply rules if impact is positive
            if impact_analysis.is_beneficial():
                await self.policy_engine.update_rules(new_rules)
                await self.metrics_collector.record_policy_update(new_rules, impact_analysis)
                self.logger.info("Policy rules updated successfully")
                return True
            else:
                self.logger.warning("New rules would have negative impact, rejecting update")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating policy rules: {e}")
            raise CreditPolicyError(f"Policy update failed: {e}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and performance metrics."""
        return {
            "is_running": self._is_running,
            "last_optimization": self._last_optimization,
            "policy_version": await self.policy_engine.get_current_version(),
            "model_performance": await self.model_manager.get_performance_summary(),
            "data_source_status": await self.data_source_manager.get_status_summary(),
            "recent_metrics": await self.metrics_collector.get_recent_summary(),
            "config": {
                "environment": self.config.environment,
                "optimization_interval": self.config.optimization.optimization_interval_hours,
                "auto_retrain_enabled": self.config.model.auto_retrain_enabled
            }
        }
    
    async def _initialize_components(self) -> None:
        """Initialize all agent components."""
        await self.policy_engine.initialize()
        await self.model_manager.initialize()
        await self.data_source_manager.initialize()
        await self.client_interaction_manager.initialize()
        await self.metrics_collector.initialize()
        await self.optimization_engine.initialize()
    
    async def _shutdown_components(self) -> None:
        """Shutdown all agent components."""
        await self.optimization_engine.shutdown()
        await self.metrics_collector.shutdown()
        await self.client_interaction_manager.shutdown()
        await self.data_source_manager.shutdown()
        await self.model_manager.shutdown()
        await self.policy_engine.shutdown()
    
    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self._is_running:
            try:
                await self._run_optimization_cycle()
                
                # Sleep until next optimization
                interval = timedelta(hours=self.config.optimization.optimization_interval_hours)
                await asyncio.sleep(interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes
    
    async def _run_optimization_cycle(self) -> None:
        """Run a single optimization cycle."""
        self.logger.info("Starting optimization cycle")
        
        try:
            # Collect performance data
            performance_data = await self.metrics_collector.get_performance_data()
            
            # Check if optimization is needed
            if await self.optimization_engine.should_optimize(performance_data):
                # Run optimization
                optimization_result = await self.optimization_engine.optimize_policies(
                    performance_data
                )
                
                # Apply optimized policies if beneficial
                if optimization_result.is_beneficial():
                    await self.update_policy_rules(optimization_result.new_rules)
                
                # Check if models need retraining
                model_performance = await self.model_manager.evaluate_all_models()
                for model_id, performance in model_performance.items():
                    if performance.needs_retraining:
                        await self.model_manager.trigger_retraining(model_id)
                
                # Evaluate data sources
                await self.data_source_manager.evaluate_all_sources()
            
            self._last_optimization = datetime.utcnow()
            self.logger.info("Optimization cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in optimization cycle: {e}")
            raise
    
    async def _collect_additional_data(self, application: LoanApplication) -> Dict[str, Any]:
        """Collect additional data for application evaluation."""
        enhanced_data = {
            "application": application,
            "external_data": {},
            "alternative_data": {}
        }
        
        # Collect data from active sources
        active_sources = await self.data_source_manager.get_active_sources()
        for source in active_sources:
            try:
                source_data = await self.data_source_manager.collect_data(
                    source.source_id, application
                )
                enhanced_data["external_data"][source.source_id] = source_data
            except Exception as e:
                self.logger.warning(f"Failed to collect data from {source.source_id}: {e}")
        
        return enhanced_data
    
    async def _identify_required_information(
        self, 
        application: LoanApplication, 
        risk_assessment: Any
    ) -> List[str]:
        """Identify additional information needed for better decision making."""
        required_info = []
        
        # Check for missing critical information
        if application.credit_score is None:
            required_info.append("credit_score")
        
        if application.income is None:
            required_info.append("monthly_income")
        
        if application.employment_status is None:
            required_info.append("employment_status")
        
        # Add risk-based requirements
        if hasattr(risk_assessment, 'required_info'):
            required_info.extend(risk_assessment.required_info)
        
        return list(set(required_info))  # Remove duplicates
    
    async def _generate_customer_explanation(
        self, 
        decision: PolicyDecision, 
        application: LoanApplication
    ) -> None:
        """Generate customer-safe explanation without revealing model internals."""
        
        adverse_actions = []
        improvements = []
        
        # Analyze failed rules to generate customer-friendly explanations
        for reason in decision.decision_reasons:
            if "Failed:" in reason:
                if "credit score" in reason.lower():
                    adverse_actions.append("Credit score below our minimum requirement")
                    improvements.append("Work on improving your credit score by paying bills on time and reducing credit utilization")
                
                elif "debt-to-income" in reason.lower():
                    adverse_actions.append("Debt-to-income ratio exceeds our guidelines")
                    improvements.append("Consider paying down existing debts to improve your debt-to-income ratio")
                
                elif "income" in reason.lower():
                    adverse_actions.append("Income level below our minimum threshold")
                    improvements.append("Document additional income sources or wait until income increases")
                
                elif "employment" in reason.lower():
                    adverse_actions.append("Employment status does not meet our requirements")
                    improvements.append("Ensure you have stable employment history and documentation")
                
                elif "risk score" in reason.lower():
                    adverse_actions.append("Overall risk assessment indicates higher than acceptable risk")
                    improvements.append("Improve your overall credit profile through responsible financial management")
        
        # If no specific reasons, provide general guidance
        if not adverse_actions and not decision.approved:
            adverse_actions.append("Application does not meet our current lending criteria")
            improvements.append("Consider reapplying in the future with improved financial standing")
        
        # Update decision with customer-safe explanations
        decision.adverse_action_reasons = adverse_actions
        decision.improvement_suggestions = improvements
        decision.customer_explanation = decision.get_customer_explanation()
    
    async def _sanitize_decision_for_customer(self, decision: PolicyDecision) -> Dict[str, Any]:
        """Remove sensitive information from decision before customer exposure."""
        return {
            "application_id": decision.application_id,
            "approved": decision.approved,
            "interest_rate": decision.interest_rate if decision.approved else None,
            "loan_term_months": decision.loan_term_months if decision.approved else None,
            "explanation": decision.customer_explanation,
            "improvement_suggestions": decision.improvement_suggestions if not decision.approved else [],
            "required_information": decision.required_information,
            "interview_questions": decision.interview_questions,
            "timestamp": decision.timestamp
        } 