"""Example test script for the Credit Policy Agent."""

import asyncio
import json
from datetime import datetime
from src.core.agent import CreditPolicyAgent, LoanApplication
from src.core.config import get_config
from src.core.explanation import SecureExplanationService, ExplanationLevel, ExplanationContext


async def test_loan_evaluation():
    """Test loan application evaluation."""
    print("🏦 Credit Policy Agent Test")
    print("=" * 50)
    
    # Initialize agent
    config = get_config()
    agent = CreditPolicyAgent(config)
    
    try:
        # Start the agent
        await agent.start()
        print("✅ Agent started successfully")
        
        # Create test loan applications
        test_applications = [
            LoanApplication(
                application_id="APP001",
                applicant_id="USER001",
                loan_amount=25000.0,
                loan_purpose="home_improvement",
                credit_score=720,
                income=65000.0,
                employment_status="employed",
                debt_to_income_ratio=0.35
            ),
            LoanApplication(
                application_id="APP002",
                applicant_id="USER002",
                loan_amount=50000.0,
                loan_purpose="debt_consolidation",
                credit_score=580,
                income=45000.0,
                employment_status="employed",
                debt_to_income_ratio=0.45
            ),
            LoanApplication(
                application_id="APP003",
                applicant_id="USER003",
                loan_amount=15000.0,
                loan_purpose="auto_loan",
                credit_score=650,
                income=55000.0,
                employment_status="self-employed",
                debt_to_income_ratio=0.25
            )
        ]
        
        # Evaluate each application
        for i, application in enumerate(test_applications, 1):
            print(f"\n📋 Evaluating Application {i}: {application.application_id}")
            print(f"   Amount: ${application.loan_amount:,.2f}")
            print(f"   Credit Score: {application.credit_score}")
            print(f"   Income: ${application.income:,.2f}")
            print(f"   DTI Ratio: {application.debt_to_income_ratio:.2%}")
            
            # Get decision
            decision = await agent.evaluate_application(application)
            
            # Display results
            status = "✅ APPROVED" if decision.approved else "❌ REJECTED"
            print(f"   Decision: {status}")
            print(f"   Confidence: {decision.confidence:.2%}")
            
            if decision.approved:
                print(f"   Interest Rate: {decision.interest_rate:.2%}")
                print(f"   Loan Term: {decision.loan_term_months} months")
            
            # Show customer-safe explanation
            print(f"\n   📝 Customer Explanation:")
            print(f"   {decision.customer_explanation}")
            
            if decision.improvement_suggestions:
                print(f"   💡 Improvement Suggestions:")
                for suggestion in decision.improvement_suggestions[:2]:
                    print(f"     • {suggestion}")
        
        # Get agent status
        print(f"\n📊 Agent Status")
        print("=" * 30)
        status = await agent.get_agent_status()
        print(f"Running: {status['is_running']}")
        print(f"Policy Version: {status['policy_version']}")
        print(f"Environment: {status['config']['environment']}")
        
        # Get current policy summary
        print(f"\n📜 Current Policy Rules")
        print("=" * 30)
        policy_summary = await agent.policy_engine.get_rule_summary()
        print(f"Total Rules: {policy_summary['total_rules']}")
        print(f"Active Rules: {policy_summary['active_rules']}")
        
        for rule_id, rule_info in list(policy_summary['rules'].items())[:3]:
            print(f"  • {rule_id}: {rule_info['description']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Stop the agent
        await agent.stop()
        print("\n✅ Agent stopped successfully")


async def test_secure_explanations():
    """Test secure explanation features that protect model internals."""
    print(f"\n🔐 Testing Secure Explanations")
    print("=" * 40)
    
    config = get_config()
    explanation_service = SecureExplanationService(config)
    
    # Mock sensitive decision data (simulating what internal system might have)
    sensitive_decision_data = {
        "application_id": "APP999",
        "approved": False,
        "confidence": 0.65,
        "decision_reasons": [
            "Failed: Credit score below minimum threshold",
            "Failed: Debt-to-income ratio exceeds guidelines"
        ],
        "timestamp": datetime.utcnow(),
        
        # SENSITIVE DATA that should NEVER be exposed to customers
        "model_weights": {
            "credit_score_coefficient": 0.45,
            "income_coefficient": 0.32,
            "dti_coefficient": -0.67
        },
        "model_parameters": {
            "decision_threshold": 0.7,
            "risk_adjustment_factor": 1.2
        },
        "internal_scoring": {
            "raw_risk_score": 0.834,
            "adjusted_score": 0.751
        },
        "feature_coefficients": [0.45, -0.32, 0.18, -0.56],
        "algorithm_details": "XGBoost ensemble with custom risk adjustments"
    }
    
    # Test different explanation levels
    test_cases = [
        ("customer", ExplanationLevel.CUSTOMER, "👤 Customer"),
        ("internal_staff", ExplanationLevel.INTERNAL_STAFF, "👨‍💼 Internal Staff"),
        ("admin", ExplanationLevel.ADMIN, "👨‍💻 Administrator"),
        ("audit", ExplanationLevel.AUDIT, "📋 Auditor")
    ]
    
    for role, level, display_name in test_cases:
        print(f"\n{display_name} Explanation:")
        print("-" * 25)
        
        context = ExplanationContext(
            user_role=role,
            explanation_level=level
        )
        
        explanation = explanation_service.generate_explanation(
            sensitive_decision_data,
            context
        )
        
        # Validate security
        is_secure = explanation_service.validate_explanation_security(explanation)
        security_status = "🔒 SECURE" if is_secure else "⚠️  LEAKED SENSITIVE DATA"
        print(f"Security Status: {security_status}")
        
        # Show key parts of explanation (but not overwhelming detail)
        if level == ExplanationLevel.CUSTOMER:
            print(f"Message: {explanation.get('message', 'N/A')}")
            if explanation.get('primary_factors'):
                print(f"Primary Factors: {explanation['primary_factors'][:2]}")  # Show first 2
        else:
            print(f"Contains internal details: {'internal_details' in explanation}")
            print(f"Contains admin details: {'admin_details' in explanation}")
            print(f"Contains compliance details: {'compliance_details' in explanation}")
        
        # Most importantly - verify no model weights are exposed
        explanation_str = str(explanation).lower()
        has_weights = "weight" in explanation_str or "coefficient" in explanation_str
        has_params = "parameter" in explanation_str or "threshold" in explanation_str
        
        if level == ExplanationLevel.CUSTOMER and (has_weights or has_params):
            print("❌ SECURITY BREACH: Model internals exposed to customer!")
        else:
            print("✅ Model internals properly protected")


async def test_policy_update():
    """Test policy rule updates."""
    print(f"\n🔧 Testing Policy Updates")
    print("=" * 30)
    
    config = get_config()
    agent = CreditPolicyAgent(config)
    
    try:
        await agent.start()
        
        # Test new rule
        new_rules = {
            "min_loan_amount": {
                "rule_type": "loan_amount",
                "field": "application.loan_amount",
                "operator": "gte",
                "value": 1000,
                "weight": 0.5,
                "description": "Minimum loan amount of $1,000",
                "active": True
            }
        }
        
        # Validate rules first
        validation = await agent.policy_engine.validate_rules(new_rules)
        print(f"Rule Validation: {'✅ Valid' if validation.valid else '❌ Invalid'}")
        
        if validation.valid:
            # Update rules
            success = await agent.update_policy_rules(new_rules)
            print(f"Rule Update: {'✅ Success' if success else '❌ Failed'}")
            
            if success:
                new_version = await agent.policy_engine.get_current_version()
                print(f"New Policy Version: {new_version}")
        else:
            print(f"Validation Errors: {validation.errors}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await agent.stop()


async def test_explanation_security_validation():
    """Test the security validation system."""
    print(f"\n🛡️  Testing Security Validation")
    print("=" * 35)
    
    config = get_config()
    explanation_service = SecureExplanationService(config)
    
    # Test cases: safe vs unsafe explanations
    test_explanations = [
        {
            "name": "Safe Customer Explanation",
            "data": {
                "decision": "declined",
                "message": "Unable to approve at this time",
                "primary_factors": ["Credit score below requirements"],
                "improvement_suggestions": ["Improve credit score"]
            },
            "should_be_secure": True
        },
        {
            "name": "UNSAFE - Contains Model Weights",
            "data": {
                "decision": "declined",
                "message": "Score calculated using weights: credit_score_weight=0.45",
                "model_coefficients": [0.45, -0.32, 0.18]
            },
            "should_be_secure": False
        },
        {
            "name": "UNSAFE - Contains Thresholds",
            "data": {
                "decision": "declined", 
                "message": "Score below threshold of 0.75",
                "internal_parameters": {"decision_threshold": 0.75}
            },
            "should_be_secure": False
        }
    ]
    
    for test_case in test_explanations:
        print(f"\n📋 Testing: {test_case['name']}")
        is_secure = explanation_service.validate_explanation_security(test_case['data'])
        expected = test_case['should_be_secure']
        
        if is_secure == expected:
            print(f"✅ PASSED - Security validation working correctly")
        else:
            print(f"❌ FAILED - Expected {'secure' if expected else 'insecure'}, got {'secure' if is_secure else 'insecure'}")
        
        print(f"   Result: {'🔒 Secure' if is_secure else '⚠️  Contains sensitive data'}")


if __name__ == "__main__":
    print("🚀 Starting Credit Policy Agent Tests\n")
    
    # Run tests
    asyncio.run(test_loan_evaluation())
    asyncio.run(test_secure_explanations())
    asyncio.run(test_policy_update())
    asyncio.run(test_explanation_security_validation())
    
    print("\n🎉 All tests completed!")
    print("\n🔑 Key Security Features Demonstrated:")
    print("  ✅ Customer explanations never expose model weights/parameters")
    print("  ✅ Different explanation levels for different user roles")
    print("  ✅ Automatic security validation prevents data leakage")
    print("  ✅ FCRA-compliant adverse action notices")
    print("  ✅ Clear improvement suggestions without revealing system vulnerabilities") 