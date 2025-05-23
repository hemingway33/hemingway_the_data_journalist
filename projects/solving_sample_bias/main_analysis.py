"""
Main Analysis Runner for Rejection Inference Methods Comparison

This module orchestrates comprehensive analysis and comparison of rejection 
inference methods across different scenarios and business contexts.
"""

import sys
import os
from typing import Dict, Any, Optional

# Import our modules
from sample_simulation import PREDEFINED_SCENARIOS, SimulationScenario
from rejection_inference_methods import create_rejection_inference_methods
from comparative_analysis import RejectionInferenceComparator  
from decision_framework import (
    RejectionInferenceDecisionFramework, 
    create_example_contexts,
    BusinessContext,
    Priority,
    DataQuality
)


class RejectionInferenceAnalysisRunner:
    """
    Main orchestrator for rejection inference analysis
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.decision_framework = RejectionInferenceDecisionFramework()
        # Get the directory where this script is located for saving plots
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
    def run_scenario_analysis(self, scenario_name: str = 'moderate_rejection', 
                            include_external_data: bool = False) -> Dict[str, Any]:
        """
        Run complete analysis for a specific scenario
        
        Parameters:
        -----------
        scenario_name : str
            Name of predefined scenario or 'custom'
        include_external_data : bool
            Whether to include external predictor methods
            
        Returns:
        --------
        Dict containing all analysis results
        """
        print(f"\n{'='*80}")
        print(f"REJECTION INFERENCE ANALYSIS: {scenario_name.upper()}")
        print(f"{'='*80}")
        
        # Get or create scenario
        if scenario_name in PREDEFINED_SCENARIOS:
            scenario = PREDEFINED_SCENARIOS[scenario_name]
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Override external data availability if specified
        if include_external_data:
            scenario.external_predictor_strength = 0.7
        
        print(f"Scenario: {scenario.name}")
        print(f"Rejection Rate: {scenario.rejection_rate:.1%}")
        print(f"Sample Size: {scenario.n_samples:,}")
        print(f"External Data: {'Yes' if scenario.external_predictor_strength > 0 else 'No'}")
        
        # Generate dataset
        print("\nGenerating dataset...")
        df, simulator = scenario.generate_complete_dataset()
        
        # Get feature columns
        feature_cols = simulator.get_normalized_feature_columns()
        
        # Create methods for comparison
        external_available = scenario.external_predictor_strength > 0
        methods = create_rejection_inference_methods(
            external_predictor_available=external_available,
            random_state=self.random_state
        )
        
        print(f"Comparing {len(methods)} methods...")
        
        # Run comparison
        comparator = RejectionInferenceComparator(random_state=self.random_state)
        results = comparator.compare_methods(methods, df, feature_cols, simulator)
        
        # Print detailed summary
        summary_df, bias_df = comparator.print_detailed_summary()
        
        # Create visualizations in the same folder as this script
        output_path = os.path.join(self.script_dir, f'{scenario_name}_analysis')
        comparator.create_visualizations(output_path)
        
        # Store results
        analysis_results = {
            'scenario': scenario,
            'dataset': df,
            'simulator': simulator,
            'methods': methods,
            'comparator': comparator,
            'results': results,
            'summary_df': summary_df,
            'bias_df': bias_df
        }
        
        self.results[scenario_name] = analysis_results
        return analysis_results
    
    def run_business_context_analysis(self, context_name: str = 'traditional_bank') -> Dict[str, Any]:
        """
        Run analysis for a specific business context using decision framework
        
        Parameters:
        -----------
        context_name : str
            Name of predefined business context
            
        Returns:
        --------
        Dict containing business context analysis results
        """
        print(f"\n{'='*80}")
        print(f"BUSINESS CONTEXT ANALYSIS: {context_name.upper()}")
        print(f"{'='*80}")
        
        # Get business context
        contexts = create_example_contexts()
        if context_name not in contexts:
            raise ValueError(f"Unknown context: {context_name}")
        
        context = contexts[context_name]
        
        # Print context details
        print(f"Business Context: {context_name.replace('_', ' ').title()}")
        print(f"Rejection Rate: {context.rejection_rate:.1%}")
        print(f"Sample Size: {context.sample_size:,}")
        print(f"External Data Available: {context.external_data_available}")
        print(f"External Data Quality: {context.external_data_quality.value}")
        print(f"Interpretability Priority: {context.interpretability_priority.value}")
        print(f"Performance Priority: {context.performance_priority.value}")
        print(f"Regulatory Constraints: {context.regulatory_constraints.value}")
        print(f"Computational Budget: {context.computational_budget.value}")
        
        # Get recommendation from decision framework
        recommendation = self.decision_framework.recommend_method(context)
        
        # Print recommendation
        print(f"\n{'-'*60}")
        print("DECISION FRAMEWORK RECOMMENDATION:")
        print(f"{'-'*60}")
        print(f"Primary Method: {recommendation.primary_method}")
        print(f"Backup Methods: {', '.join(recommendation.backup_methods)}")
        print(f"Confidence Level: {recommendation.confidence_level}")
        print(f"Expected Performance: {recommendation.expected_performance}")
        print(f"Implementation Complexity: {recommendation.implementation_complexity}")
        print(f"\nRationale: {recommendation.rationale}")
        
        if recommendation.considerations:
            print(f"\nConsiderations:")
            for consideration in recommendation.considerations:
                print(f"  • {consideration}")
        
        # Create scenario based on business context
        scenario = SimulationScenario(
            name=f"Business Context: {context_name}",
            n_samples=context.sample_size,
            rejection_rate=context.rejection_rate,
            external_predictor_strength=0.7 if context.external_data_available else 0.0,
            random_state=self.random_state
        )
        
        # Run technical analysis
        print(f"\n{'-'*60}")
        print("TECHNICAL VALIDATION:")
        print(f"{'-'*60}")
        
        df, simulator = scenario.generate_complete_dataset()
        feature_cols = simulator.get_normalized_feature_columns()
        
        # Create methods for comparison (focus on recommended ones)
        methods = create_rejection_inference_methods(
            external_predictor_available=context.external_data_available,
            random_state=self.random_state
        )
        
        # Run comparison
        comparator = RejectionInferenceComparator(random_state=self.random_state)
        results = comparator.compare_methods(methods, df, feature_cols, simulator)
        
        # Get recommendation based on priority
        if context.performance_priority == Priority.HIGH and context.interpretability_priority == Priority.LOW:
            priority = 'performance'
        elif context.interpretability_priority == Priority.HIGH:
            priority = 'interpretability'
        else:
            priority = 'balanced'
        
        recommended_method = comparator.recommend_method(priority)
        
        # Store results
        context_results = {
            'context': context,
            'recommendation': recommendation,
            'scenario': scenario,
            'dataset': df,
            'simulator': simulator,
            'methods': methods,
            'comparator': comparator,
            'results': results,
            'technical_recommendation': recommended_method
        }
        
        return context_results
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """
        Run comprehensive comparison across multiple scenarios
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE REJECTION INFERENCE COMPARISON")
        print(f"{'='*80}")
        
        # Run analysis for key scenarios
        scenarios_to_analyze = ['low_rejection', 'moderate_rejection', 'high_rejection', 'external_data_available']
        
        all_results = {}
        
        for scenario_name in scenarios_to_analyze:
            try:
                include_external = scenario_name == 'external_data_available'
                results = self.run_scenario_analysis(scenario_name, include_external)
                all_results[scenario_name] = results
            except Exception as e:
                print(f"Error analyzing {scenario_name}: {str(e)}")
                continue
        
        # Summary comparison across scenarios
        self._create_cross_scenario_summary(all_results)
        
        return all_results
    
    def _create_cross_scenario_summary(self, all_results: Dict[str, Any]):
        """
        Create summary comparison across scenarios
        """
        print(f"\n{'='*80}")
        print("CROSS-SCENARIO SUMMARY")
        print(f"{'='*80}")
        
        # Collect best methods by scenario
        best_methods = {}
        
        for scenario_name, results in all_results.items():
            if 'comparator' in results:
                comparator = results['comparator']
                
                # Get best method for each priority
                best_performance = comparator.recommend_method('performance')
                best_interpretability = comparator.recommend_method('interpretability')
                best_balanced = comparator.recommend_method('balanced')
                
                best_methods[scenario_name] = {
                    'performance': best_performance,
                    'interpretability': best_interpretability,
                    'balanced': best_balanced,
                    'rejection_rate': results['scenario'].rejection_rate
                }
        
        # Print summary table
        print(f"\nBEST METHODS BY SCENARIO AND PRIORITY:")
        print(f"{'-'*100}")
        print(f"{'Scenario':<20} {'Rejection Rate':<15} {'Performance':<25} {'Interpretability':<25} {'Balanced':<25}")
        print(f"{'-'*100}")
        
        for scenario_name, methods in best_methods.items():
            rejection_rate = f"{methods['rejection_rate']:.0%}"
            print(f"{scenario_name:<20} {rejection_rate:<15} {methods['performance']:<25} "
                  f"{methods['interpretability']:<25} {methods['balanced']:<25}")
        
        # General recommendations
        print(f"\nGENERAL RECOMMENDATIONS:")
        print(f"{'-'*60}")
        recommendations = [
            "• Low rejection rates (<30%): Approved-only methods often sufficient",
            "• Medium rejection rates (30-60%): Conservative propensity weighting recommended",
            "• High rejection rates (>60%): Aggressive methods needed, consider external data",
            "• Interpretability priority: Conservative propensity or ensemble averaging",
            "• Performance priority: External predictor methods when available",
            "• Regulatory constraints: Avoid simple rejection inference, prefer propensity methods",
            "• Resource constraints: Start with approved-only, upgrade as needed"
        ]
        
        for rec in recommendations:
            print(rec)
    
    def demonstrate_decision_framework(self):
        """
        Demonstrate the decision framework with example business contexts
        """
        print(f"\n{'='*80}")
        print("DECISION FRAMEWORK DEMONSTRATION")
        print(f"{'='*80}")
        
        # Get example contexts
        contexts = create_example_contexts()
        
        print(f"Analyzing {len(contexts)} business contexts...\n")
        
        for context_name, context in contexts.items():
            print(f"{'-'*60}")
            print(f"CONTEXT: {context_name.upper().replace('_', ' ')}")
            print(f"{'-'*60}")
            
            recommendation = self.decision_framework.recommend_method(context)
            
            print(f"Recommended Method: {recommendation.primary_method}")
            print(f"Confidence: {recommendation.confidence_level}")
            print(f"Rationale: {recommendation.rationale}")
            
            if recommendation.considerations:
                print("Key Considerations:")
                for consideration in recommendation.considerations[:3]:  # Show top 3
                    print(f"  • {consideration}")
            print()
        
        # Show decision matrix
        print(f"{'-'*60}")
        print("DECISION MATRIX")
        print(f"{'-'*60}")
        
        decision_matrix = self.decision_framework.create_decision_matrix()
        print(decision_matrix.to_string(index=False))
    
    def create_method_selection_guide(self):
        """
        Create a practical method selection guide
        """
        print(f"\n{'='*80}")
        print("REJECTION INFERENCE METHOD SELECTION GUIDE")
        print(f"{'='*80}")
        
        guide_sections = {
            "QUICK DECISION TREE": [
                "1. Is rejection rate < 30%?",
                "   → YES: Use Approved Only or Regularized Approved",
                "   → NO: Continue to question 2",
                "",
                "2. Is interpretability critical (regulatory/compliance)?",
                "   → YES: Use Conservative Propensity Weighting",
                "   → NO: Continue to question 3",
                "",
                "3. Is external data available and high quality?",
                "   → YES: Use Hybrid External + Propensity",
                "   → NO: Use Simple Rejection Inference or Propensity Weighting",
                "",
                "4. Is computational budget limited?",
                "   → YES: Use simpler methods (Approved Only, Simple Rejection Inference)",
                "   → NO: Use more sophisticated methods (Ensemble, Hybrid approaches)"
            ],
            
            "METHOD PROFILES SUMMARY": [
                "APPROVED ONLY:",
                "  • Best for: Low rejection rates, high interpretability needs",
                "  • Pros: Simple, interpretable, regulatory-friendly",
                "  • Cons: Ignores selection bias, lower performance",
                "",
                "CONSERVATIVE PROPENSITY WEIGHTING:",
                "  • Best for: Balanced needs, regulatory environments",
                "  • Pros: Good interpretability, addresses bias, stable",
                "  • Cons: Moderate performance improvement",
                "",
                "SIMPLE REJECTION INFERENCE:",
                "  • Best for: High rejection rates, performance priority",
                "  • Pros: Addresses bias directly, good performance",
                "  • Cons: Poor interpretability, regulatory risk",
                "",
                "EXTERNAL PREDICTOR ENHANCED:",
                "  • Best for: High rejection rates, external data available",
                "  • Pros: Best performance potential",
                "  • Cons: Complex, data dependency, interpretability issues"
            ],
            
            "IMPLEMENTATION ROADMAP": [
                "PHASE 1 - BASELINE (Week 1-2):",
                "  • Implement Approved Only method",
                "  • Establish performance baseline",
                "  • Measure bias impact",
                "",
                "PHASE 2 - BIAS CORRECTION (Week 3-4):",
                "  • Implement Conservative Propensity Weighting",
                "  • Compare with baseline",
                "  • Validate interpretability",
                "",
                "PHASE 3 - OPTIMIZATION (Week 5-8):",
                "  • If external data available: Test External Predictor methods",
                "  • If high rejection rate: Test Simple Rejection Inference",
                "  • If stability critical: Test Ensemble Averaging",
                "",
                "PHASE 4 - PRODUCTION (Week 9+):",
                "  • Deploy best method based on validation",
                "  • Implement monitoring and model stability checks",
                "  • Plan for regular retraining and validation"
            ]
        }
        
        for section_name, content in guide_sections.items():
            print(f"\n{section_name}:")
            print("-" * len(section_name))
            for line in content:
                print(line)


def main():
    """
    Main function to run comprehensive rejection inference analysis
    """
    print("Starting Comprehensive Rejection Inference Analysis...")
    
    # Initialize analysis runner
    runner = RejectionInferenceAnalysisRunner(random_state=42)
    
    # Run different types of analysis
    try:
        # 1. Demonstrate decision framework
        runner.demonstrate_decision_framework()
        
        # 2. Run business context analysis
        print("\n" + "="*80)
        print("BUSINESS CONTEXT EXAMPLES")
        print("="*80)
        
        contexts_to_analyze = ['fintech_startup', 'traditional_bank', 'online_lender']
        for context_name in contexts_to_analyze:
            try:
                runner.run_business_context_analysis(context_name)
            except Exception as e:
                print(f"Error analyzing {context_name}: {str(e)}")
                continue
        
        # 3. Run scenario analysis
        print("\n" + "="*80)
        print("TECHNICAL SCENARIO ANALYSIS")
        print("="*80)
        
        # Focus on key scenarios
        scenarios_to_test = ['moderate_rejection', 'high_rejection']
        for scenario in scenarios_to_test:
            try:
                runner.run_scenario_analysis(scenario, include_external_data=False)
            except Exception as e:
                print(f"Error analyzing {scenario}: {str(e)}")
                continue
        
        # Test external data scenario
        try:
            runner.run_scenario_analysis('moderate_rejection', include_external_data=True)
        except Exception as e:
            print(f"Error analyzing external data scenario: {str(e)}")
        
        # 4. Create practical guide
        runner.create_method_selection_guide()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print("\nKey outputs generated:")
        print("• Method comparison visualizations")
        print("• Business context recommendations")
        print("• Technical performance analysis")
        print("• Practical implementation guide")
        
    except Exception as e:
        print(f"Error in main analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 