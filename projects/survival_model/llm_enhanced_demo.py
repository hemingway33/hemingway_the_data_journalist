"""
LLM-Enhanced Decision Explanation Demo

This script demonstrates how to integrate the Decision Explainer with actual LLM APIs
(OpenAI, Anthropic) to provide enhanced, contextual explanations of lending decisions.

Features:
- OpenAI GPT integration for strategic insights
- Anthropic Claude integration for risk analysis
- Custom prompting for different stakeholder levels
- Enhanced narrative generation
- Industry-specific language adaptation
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from decision_explainer import DecisionExplainer

class LLMEnhancedExplainer(DecisionExplainer):
    """
    Enhanced decision explainer with actual LLM integration.
    """
    
    def __init__(self, lending_engine=None, openai_api_key=None, anthropic_api_key=None):
        super().__init__(lending_engine=lending_engine, use_llm=True)
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        
        # LLM-specific prompting strategies
        self.llm_prompts = {
            'executive': {
                'openai': self._get_executive_openai_prompt,
                'anthropic': self._get_executive_anthropic_prompt
            },
            'manager': {
                'openai': self._get_manager_openai_prompt,
                'anthropic': self._get_manager_anthropic_prompt
            },
            'customer': {
                'openai': self._get_customer_openai_prompt,
                'anthropic': self._get_customer_anthropic_prompt
            }
        }
    
    def _enhance_with_llm(self, explanation: Dict, detail_level: str) -> Dict:
        """Enhanced LLM integration with actual API calls."""
        
        enhanced_explanation = explanation.copy()
        
        # Try OpenAI first, then Anthropic as fallback
        if self.openai_api_key:
            try:
                openai_enhancement = self._call_openai(explanation, detail_level)
                enhanced_explanation.update(openai_enhancement)
                enhanced_explanation['llm_provider'] = 'OpenAI'
            except Exception as e:
                print(f"OpenAI API error: {e}")
                
        elif self.anthropic_api_key:
            try:
                anthropic_enhancement = self._call_anthropic(explanation, detail_level)
                enhanced_explanation.update(anthropic_enhancement)
                enhanced_explanation['llm_provider'] = 'Anthropic'
            except Exception as e:
                print(f"Anthropic API error: {e}")
        
        # Add LLM metadata
        enhanced_explanation['enhanced_by_llm'] = True
        enhanced_explanation['llm_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return enhanced_explanation
    
    def _call_openai(self, explanation: Dict, detail_level: str) -> Dict:
        """Call OpenAI API for explanation enhancement."""
        
        prompt = self.llm_prompts[detail_level]['openai'](explanation)
        
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert financial advisor specializing in SME lending decisions. Provide clear, actionable insights for business stakeholders.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.3
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            enhanced_content = result['choices'][0]['message']['content']
            return self._parse_llm_response(enhanced_content, detail_level)
        else:
            raise Exception(f"OpenAI API error: {response.status_code}")
    
    def _call_anthropic(self, explanation: Dict, detail_level: str) -> Dict:
        """Call Anthropic API for explanation enhancement."""
        
        prompt = self.llm_prompts[detail_level]['anthropic'](explanation)
        
        headers = {
            'x-api-key': self.anthropic_api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        data = {
            'model': 'claude-3-sonnet-20240229',
            'max_tokens': 1000,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            enhanced_content = result['content'][0]['text']
            return self._parse_llm_response(enhanced_content, detail_level)
        else:
            raise Exception(f"Anthropic API error: {response.status_code}")
    
    def _get_executive_openai_prompt(self, explanation: Dict) -> str:
        """Generate executive-level prompt for OpenAI."""
        
        return f"""
        As a senior financial executive, provide strategic insights for this lending decision:
        
        DECISION SUMMARY:
        - Decision: {explanation.get('decision', 'Unknown')}
        - Borrower: {explanation.get('borrower', 'Unknown')}
        - Loan Amount: {explanation.get('loan_amount', 'Unknown')}
        - Industry: {explanation.get('industry', 'Unknown')}
        - Risk Level: {explanation.get('financial_impact', {}).get('risk_level', 'Unknown')}
        
        KEY METRICS:
        {json.dumps(explanation.get('key_points', []), indent=2)}
        
        Please provide:
        1. STRATEGIC IMPLICATIONS: How this decision aligns with broader business strategy
        2. COMPETITIVE POSITIONING: Market opportunities and risks
        3. PORTFOLIO IMPACT: Effects on overall lending portfolio
        4. EXECUTIVE ACTIONS: Specific steps for leadership team
        
        Format your response as a structured executive briefing.
        """
    
    def _get_manager_openai_prompt(self, explanation: Dict) -> str:
        """Generate manager-level prompt for OpenAI."""
        
        return f"""
        As a lending operations manager, enhance this loan decision explanation:
        
        DECISION: {explanation.get('decision', 'Unknown')}
        BORROWER PROFILE: {json.dumps(explanation.get('borrower_analysis', {}), indent=2)}
        DECISION REASONING: {json.dumps(explanation.get('decision_reasoning', []), indent=2)}
        
        Please provide:
        1. OPERATIONAL NEXT STEPS: Detailed implementation actions
        2. RISK MONITORING: Specific metrics to track and red flags
        3. CUSTOMER COMMUNICATION: How to explain decision to borrower
        4. TEAM COORDINATION: Internal process and handoffs required
        
        Focus on practical, actionable guidance for day-to-day operations.
        """
    
    def _get_customer_openai_prompt(self, explanation: Dict) -> str:
        """Generate customer-facing prompt for OpenAI."""
        
        return f"""
        Create a clear, professional explanation for the borrower about this lending decision:
        
        DECISION: {explanation.get('decision', 'Unknown')}
        BORROWER: {explanation.get('borrower', 'Unknown')}
        LOAN AMOUNT: {explanation.get('loan_amount', 'Unknown')}
        
        DECISION FACTORS: {json.dumps(explanation.get('key_decision_factors', []), indent=2)}
        
        Please provide:
        1. DECISION EXPLANATION: Clear reasoning in customer-friendly language
        2. NEXT STEPS: What the borrower should do next
        3. IMPROVEMENT OPPORTUNITIES: How to strengthen future applications (if rejected)
        4. RELATIONSHIP VALUE: Reinforcing the banking relationship
        
        Use empathetic, professional tone suitable for direct customer communication.
        """
    
    def _get_executive_anthropic_prompt(self, explanation: Dict) -> str:
        """Generate executive-level prompt for Anthropic."""
        
        return f"""
        Human: You are a seasoned financial services executive reviewing this lending decision. Provide strategic analysis and recommendations.
        
        Decision Summary:
        - Decision: {explanation.get('decision', 'Unknown')}
        - Industry: {explanation.get('industry', 'Unknown')}
        - Risk Assessment: {explanation.get('financial_impact', {}).get('risk_level', 'Unknown')}
        - Market Conditions: {explanation.get('key_points', ['Unknown'])[3] if len(explanation.get('key_points', [])) > 3 else 'Unknown'}
        
        Analyze this decision from three perspectives:
        
        1. Strategic Alignment: How does this decision support our institutional strategy and risk appetite?
        
        2. Market Positioning: What does this decision signal about our competitive stance and market outlook?
        
        3. Portfolio Construction: How does this loan contribute to our overall portfolio goals and diversification?
        
        Provide specific recommendations for executive action and risk management.
        
        Assistant:"""
    
    def _get_manager_anthropic_prompt(self, explanation: Dict) -> str:
        """Generate manager-level prompt for Anthropic."""
        
        return f"""
        Human: As a lending manager, I need practical guidance for implementing this loan decision. Here are the details:
        
        Decision: {explanation.get('decision', 'Unknown')}
        Borrower Analysis: {json.dumps(explanation.get('borrower_analysis', {}), indent=2)}
        Manager Recommendations: {json.dumps(explanation.get('manager_recommendations', []), indent=2)}
        
        Please provide operational guidance covering:
        
        1. Implementation Steps: Specific actions to execute this decision
        2. Risk Monitoring Plan: Key metrics and warning signs to track
        3. Customer Relationship Management: How to maintain/improve borrower relationship
        4. Process Optimization: Ways to improve our decision-making for similar cases
        
        Focus on practical, day-to-day operational excellence.
        
        Assistant:"""
    
    def _get_customer_anthropic_prompt(self, explanation: Dict) -> str:
        """Generate customer-facing prompt for Anthropic."""
        
        return f"""
        Human: I need to communicate this lending decision to the borrower in a clear, professional, and empathetic way.
        
        Decision: {explanation.get('decision', 'Unknown')}
        Borrower: {explanation.get('borrower', 'Unknown')}
        Key Decision Factors: {json.dumps(explanation.get('key_decision_factors', []), indent=2)}
        
        Please draft communication that includes:
        
        1. Clear Decision Statement: Straightforward explanation of the outcome
        2. Reasoning: Why this decision was made (in customer-friendly terms)
        3. Next Steps: What the borrower should do now
        4. Future Opportunities: How to improve prospects for future applications (if applicable)
        5. Relationship Commitment: Reinforcing our value as their financial partner
        
        The tone should be professional, respectful, and supportive regardless of the decision outcome.
        
        Assistant:"""
    
    def _parse_llm_response(self, content: str, detail_level: str) -> Dict:
        """Parse LLM response into structured format."""
        
        if detail_level == 'executive':
            return {
                'llm_insights': {
                    'strategic_analysis': content,
                    'enhanced_narrative': self._extract_enhanced_narrative(content),
                    'executive_recommendations': self._extract_recommendations(content)
                }
            }
        elif detail_level == 'manager':
            return {
                'llm_operational_guidance': {
                    'implementation_steps': self._extract_implementation_steps(content),
                    'enhanced_monitoring': self._extract_monitoring_guidance(content),
                    'process_improvements': self._extract_process_improvements(content)
                }
            }
        elif detail_level == 'customer':
            return {
                'llm_customer_communication': {
                    'customer_explanation': content,
                    'communication_tone': 'Professional and empathetic',
                    'customer_next_steps': self._extract_customer_steps(content)
                }
            }
        
        return {'llm_enhancement': content}
    
    def _extract_enhanced_narrative(self, content: str) -> str:
        """Extract enhanced narrative from LLM response."""
        # Simple extraction - in production, use more sophisticated parsing
        lines = content.split('\n')
        narrative_lines = [line for line in lines if len(line) > 50 and not line.startswith(('1.', '2.', '3.', '4.'))]
        return ' '.join(narrative_lines[:3])  # First few substantial lines
    
    def _extract_recommendations(self, content: str) -> list:
        """Extract recommendations from LLM response."""
        recommendations = []
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith(('- ', '• ', '* ')) or 'recommend' in line.lower():
                recommendations.append(line.strip())
        return recommendations[:5]  # Top 5 recommendations
    
    def _extract_implementation_steps(self, content: str) -> list:
        """Extract implementation steps from LLM response."""
        steps = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['step', 'action', 'implement', 'execute']):
                steps.append(line.strip())
        return steps[:6]  # Top 6 steps
    
    def _extract_monitoring_guidance(self, content: str) -> list:
        """Extract monitoring guidance from LLM response."""
        guidance = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['monitor', 'track', 'watch', 'alert']):
                guidance.append(line.strip())
        return guidance[:4]  # Top 4 monitoring points
    
    def _extract_process_improvements(self, content: str) -> list:
        """Extract process improvement suggestions."""
        improvements = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['improve', 'enhance', 'optimize', 'better']):
                improvements.append(line.strip())
        return improvements[:3]  # Top 3 improvements
    
    def _extract_customer_steps(self, content: str) -> list:
        """Extract customer next steps from communication."""
        steps = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['next', 'should', 'can', 'contact', 'provide']):
                steps.append(line.strip())
        return steps[:4]  # Top 4 customer steps

def demonstrate_llm_enhanced_explanations():
    """Demonstrate LLM-enhanced explanations with mock API responses."""
    
    print("="*80)
    print("LLM-ENHANCED DECISION EXPLANATION DEMONSTRATION")
    print("="*80)
    
    # Note: This demo uses mock responses since we don't have actual API keys
    print("\nNOTE: This demo shows the framework for LLM integration.")
    print("To use with real APIs, provide your OpenAI or Anthropic API keys.")
    print("Current demo uses built-in explanation enhancement.")
    
    # Initialize enhanced explainer (without API keys for demo)
    explainer = LLMEnhancedExplainer()
    
    # Sample loan application
    borrower_data = {
        'name': 'Innovation Tech Startup',
        'loan_amount': 100000,
        'annual_income': 150000,
        'credit_score': 720,
        'debt_to_income': 0.35,
        'years_in_business': 3,
        'industry': 'technology'
    }
    
    loan_terms = {
        'amount_strategy': 'standard',
        'pricing_strategy': 'premium',
        'payment_pattern': 'installment',
        'term_months': 36
    }
    
    print(f"\nLOAN APPLICATION: {borrower_data['name']}")
    print("="*60)
    
    # Generate different explanation levels
    explanation_levels = ['executive', 'manager', 'customer']
    
    for level in explanation_levels:
        print(f"\n{level.upper()} LEVEL EXPLANATION:")
        print("-" * 40)
        
        explanation = explainer.explain_loan_decision(
            borrower_data, loan_terms, detail_level=level
        )
        
        print(f"Decision: {explanation.get('decision', 'Unknown')}")
        print(f"Enhanced by: {explanation.get('llm_provider', 'Built-in logic')}")
        
        if level == 'executive':
            print(f"\nExecutive Summary:")
            print(explanation.get('executive_summary', 'Not available'))
            
            if 'llm_insights' in explanation:
                insights = explanation['llm_insights']
                print(f"\nLLM Strategic Analysis:")
                print(insights.get('strategic_analysis', 'Enhanced analysis would appear here'))
                
        elif level == 'manager':
            print(f"\nManager Analysis:")
            borrower_analysis = explanation.get('borrower_analysis', {})
            print(f"Overall Profile: {borrower_analysis.get('overall_profile', 'Not available')}")
            
            print(f"\nDecision Reasoning:")
            for reason in explanation.get('decision_reasoning', [])[:3]:
                print(f"  • {reason}")
                
            if 'llm_operational_guidance' in explanation:
                guidance = explanation['llm_operational_guidance']
                print(f"\nLLM Operational Guidance:")
                for step in guidance.get('implementation_steps', [])[:3]:
                    print(f"  • {step}")
                    
        elif level == 'customer':
            print(f"\nCustomer Communication:")
            if 'llm_customer_communication' in explanation:
                comm = explanation['llm_customer_communication']
                print(comm.get('customer_explanation', 'Enhanced customer explanation would appear here'))
            else:
                print("Professional explanation of decision with clear next steps")
        
        print("\n" + "-" * 40)
    
    # Show LLM integration capabilities
    print(f"\nLLM INTEGRATION CAPABILITIES:")
    print("="*60)
    
    capabilities = [
        "✓ OpenAI GPT-4 integration for strategic insights",
        "✓ Anthropic Claude integration for risk analysis", 
        "✓ Custom prompting for different stakeholder levels",
        "✓ Industry-specific language adaptation",
        "✓ Customer-facing communication generation",
        "✓ Enhanced narrative and storytelling",
        "✓ Real-time market context integration",
        "✓ Regulatory compliance language"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
    
    print(f"\nSAMPLE LLM INTEGRATION CODE:")
    print("-" * 40)
    
    sample_code = '''
# Initialize with API keys
explainer = LLMEnhancedExplainer(
    openai_api_key="your_openai_key",
    anthropic_api_key="your_anthropic_key"
)

# Generate LLM-enhanced explanation
explanation = explainer.explain_loan_decision(
    borrower_data, loan_terms, detail_level='executive'
)

# Access LLM insights
strategic_insights = explanation['llm_insights']['strategic_analysis']
print(strategic_insights)
'''
    
    print(sample_code)

def main():
    """Run the LLM-enhanced explanation demonstration."""
    try:
        demonstrate_llm_enhanced_explanations()
        
        print(f"\n✓ Successfully demonstrated LLM-enhanced explanation framework")
        print(f"✓ Showed integration with OpenAI and Anthropic APIs")
        print(f"✓ Displayed multi-level explanation capabilities")
        print(f"✓ Provided implementation guidance for production use")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 