"""
Demo Script: FICO Performance Inference Methods

This script demonstrates the exact FICO performance inference methods 
from the "Building Powerful Scorecards" whitepaper with clear explanations.
"""

import pandas as pd
import numpy as np
from sample_simulation import SimulationScenario
from fico_performance_inference_methods import (
    ExternalInformationMethod,
    DomainExpertiseParcelingMethod, 
    DualScoreInferenceMethod,
    FICOMethodValidator
)

def demo_external_information_method():
    """
    Demonstrate the FICO External Information Method
    
    This method uses credit bureau scores to infer reject performance.
    Formula: logOdds = B0 + B1*CB_SCORE
    """
    print("="*80)
    print("DEMO: FICO EXTERNAL INFORMATION METHOD")
    print("="*80)
    print("Method: Uses credit bureau scores to infer reject performance")
    print("Formula: logOdds = B0 + B1*CB_SCORE")
    print("         pG = 1/(1 + exp(-(B0 + B1*CB_SCORE)))")
    print()
    
    # Create scenario with external data
    scenario = SimulationScenario(
        name="External Info Demo",
        rejection_rate=0.6,
        n_samples=5000,
        external_predictor_strength=0.7,
        random_state=42
    )
    
    df, simulator = scenario.generate_complete_dataset()
    feature_cols = simulator.get_normalized_feature_columns()
    
    print(f"Dataset: {len(df)} samples, {(~df['approved']).mean():.1%} rejection rate")
    print(f"External predictor available: Yes")
    print()
    
    # Apply method
    method = ExternalInformationMethod(
        external_score_col='external_predictor',
        assignment_threshold=0.5
    )
    
    df_augmented = method.infer_reject_performance(df, feature_cols)
    
    # Show results
    print("RESULTS:")
    print(f"B0 (Intercept): {method.B0:.3f}")
    print(f"B1 (CB Coefficient): {method.B1:.3f}")
    print(f"Rejects assigned Good: {method.inference_stats['n_rejects_assigned_good']}")
    print(f"Rejects assigned Bad: {method.inference_stats['n_rejects_assigned_bad']}")
    print(f"Average pG for rejects: {method.inference_stats['avg_pG_rejects']:.6f}")
    print()
    
    # Show interpretation
    print("INTERPRETATION:")
    print(f"- For every 1-unit increase in CB score, odds of default change by factor {np.exp(method.B1):.3f}")
    print(f"- Most rejects get assigned as {'Good' if method.inference_stats['n_rejects_assigned_good'] > method.inference_stats['n_rejects_assigned_bad'] else 'Bad'}")
    print(f"- Method is {'conservative' if method.inference_stats['avg_pG_rejects'] < 0.5 else 'liberal'} in assignments")
    print()


def demo_domain_parceling_method():
    """
    Demonstrate the FICO Domain Expertise Parceling Method
    
    This method uses iterative parceling with viability testing.
    """
    print("="*80)
    print("DEMO: FICO DOMAIN EXPERTISE PARCELING METHOD")
    print("="*80)
    print("Method: Iterative parceling with viability testing")
    print("Process:")
    print("  1. Create KN_SCORE on known population")
    print("  2. Initial assignment: logOdds = C0 + C1*KN_SCORE")
    print("  3. Train model T on full TTD population") 
    print("  4. Test viability by comparing log(Odds) alignment")
    print("  5. Iterate until convergence")
    print()
    
    # Create scenario
    scenario = SimulationScenario(
        name="Parceling Demo", 
        rejection_rate=0.5,
        n_samples=5000,
        external_predictor_strength=0.6,
        random_state=42
    )
    
    df, simulator = scenario.generate_complete_dataset()
    feature_cols = simulator.get_normalized_feature_columns()
    
    print(f"Dataset: {len(df)} samples, {(~df['approved']).mean():.1%} rejection rate")
    print()
    
    # Apply method
    method = DomainExpertiseParcelingMethod(
        max_iterations=5,
        convergence_threshold=0.05
    )
    
    df_augmented = method.infer_reject_performance(df, feature_cols)
    
    # Show results
    print("RESULTS:")
    print(f"C0 (KN Intercept): {method.C0:.3f}")
    print(f"C1 (KN Coefficient): {method.C1:.3f}")
    print(f"Iterations completed: {method.inference_stats['n_iterations']}")
    print(f"Converged: {method.inference_stats['converged']}")
    print(f"Final alignment score: {method.inference_stats['final_alignment_score']:.3f}")
    print()
    
    # Show parceling history
    print("PARCELING HISTORY:")
    for i, step in enumerate(method.inference_stats['parceling_history'][:3]):  # Show first 3
        print(f"  Iteration {step['iteration']}:")
        print(f"    Known slope: {step['known_slope']:.4f}")
        print(f"    Unknown slope: {step['unknown_slope']:.4f}")
        print(f"    Alignment score: {step['alignment_score']:.4f}")
        print(f"    Viable: {step['is_viable']}")
    if len(method.inference_stats['parceling_history']) > 3:
        print("    ...")
    print()
    
    # Show interpretation
    print("INTERPRETATION:")
    print(f"- Method {'converged' if method.inference_stats['converged'] else 'did not converge'}")
    print(f"- Alignment threshold: 0.05, achieved: {method.inference_stats['final_alignment_score']:.3f}")
    if not method.inference_stats['converged']:
        print("- May need better KN_SCORE engineering or relaxed convergence criteria")
    print()


def demo_dual_score_method():
    """
    Demonstrate the FICO Dual Score Inference Method
    
    This method combines KN_SCORE (performance) and AR_SCORE (acceptance).
    """
    print("="*80)
    print("DEMO: FICO DUAL SCORE INFERENCE METHOD")
    print("="*80)
    print("Method: Combines KN_SCORE (performance) and AR_SCORE (acceptance)")
    print("Formula: Dual Score = ar_weight * AR_SCORE + kn_weight * KN_SCORE")
    print("Purpose: Addresses selection bias by modeling approval process")
    print()
    
    # Create scenario
    scenario = SimulationScenario(
        name="Dual Score Demo",
        rejection_rate=0.6,
        n_samples=5000,
        external_predictor_strength=0.5,
        random_state=42
    )
    
    df, simulator = scenario.generate_complete_dataset()
    feature_cols = simulator.get_normalized_feature_columns()
    
    print(f"Dataset: {len(df)} samples, {(~df['approved']).mean():.1%} rejection rate")
    print()
    
    # Apply method
    method = DualScoreInferenceMethod(
        ar_weight=0.3,  # 30% weight to acceptance score
        kn_weight=0.7   # 70% weight to performance score
    )
    
    df_augmented = method.infer_reject_performance(df, feature_cols)
    
    # Show results
    print("RESULTS:")
    print(f"AR Weight (Acceptance): {method.ar_weight}")
    print(f"KN Weight (Performance): {method.kn_weight}")
    print(f"Rejects assigned Good: {method.inference_stats['n_rejects_assigned_good']}")
    print(f"Rejects assigned Bad: {method.inference_stats['n_rejects_assigned_bad']}")
    print(f"Average dual score: {method.inference_stats['avg_dual_score_rejects']:.6f}")
    print(f"Average reject probability: {method.inference_stats['avg_reject_prob']:.6f}")
    print()
    
    # Show interpretation
    print("INTERPRETATION:")
    print(f"- AR_SCORE models the approval process (selection bias)")
    print(f"- KN_SCORE models performance risk on known population")
    print(f"- Weighting: {method.ar_weight*100:.0f}% approval, {method.kn_weight*100:.0f}% performance")
    good_pct = method.inference_stats['n_rejects_assigned_good'] / (method.inference_stats['n_rejects_assigned_good'] + method.inference_stats['n_rejects_assigned_bad'])
    print(f"- Assigns {good_pct:.1%} of rejects as Good performers")
    print()


def demo_method_comparison():
    """
    Compare all three FICO methods on the same dataset
    """
    print("="*80)
    print("DEMO: FICO METHODS COMPARISON")
    print("="*80)
    
    # Create comprehensive scenario
    scenario = SimulationScenario(
        name="Comparison Demo",
        rejection_rate=0.6,
        n_samples=8000,
        external_predictor_strength=0.6,
        random_state=42
    )
    
    df, simulator = scenario.generate_complete_dataset()
    feature_cols = simulator.get_normalized_feature_columns()
    
    print(f"Dataset: {len(df)} samples")
    print(f"Rejection rate: {(~df['approved']).mean():.1%}")
    print(f"True reject default rate: {df[~df['approved']]['actual_default'].mean():.1%}")
    print()
    
    # Test all methods
    methods = [
        ("External Information", ExternalInformationMethod()),
        ("Domain Parceling", DomainExpertiseParcelingMethod(max_iterations=3)),
        ("Dual Score", DualScoreInferenceMethod())
    ]
    
    validator = FICOMethodValidator()
    results = []
    
    for name, method in methods:
        print(f"Testing {name}...")
        result = validator.validate_method(method, df, feature_cols)
        if result:
            results.append({
                'Method': name,
                'AUC': result['auc'],
                'Inference_Accuracy': result.get('reject_inference_accuracy', 'N/A')
            })
    
    # Show comparison
    print("\nCOMPARISON RESULTS:")
    print("-" * 50)
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    print()
    
    # Best method
    if results:
        best_auc = max(results, key=lambda x: x['AUC'])
        print(f"Best AUC: {best_auc['Method']} ({best_auc['AUC']:.3f})")
        
        # Filter for methods with inference accuracy
        inference_results = [r for r in results if r['Inference_Accuracy'] != 'N/A']
        if inference_results:
            best_inference = max(inference_results, key=lambda x: x['Inference_Accuracy'])
            print(f"Best Inference: {best_inference['Method']} ({best_inference['Inference_Accuracy']:.3f})")
    print()


def run_full_demo():
    """
    Run the complete FICO methods demonstration
    """
    print("FICO PERFORMANCE INFERENCE METHODS - LIVE DEMO")
    print("Based on 'Building Powerful Scorecards' Whitepaper")
    print("=" * 80)
    print()
    
    # Run individual method demos
    demo_external_information_method()
    input("Press Enter to continue to Domain Parceling demo...")
    print()
    
    demo_domain_parceling_method() 
    input("Press Enter to continue to Dual Score demo...")
    print()
    
    demo_dual_score_method()
    input("Press Enter to continue to methods comparison...")
    print()
    
    demo_method_comparison()
    
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("Key Takeaways:")
    print("✓ External Information: Simple, effective with external data")
    print("✓ Domain Parceling: Sophisticated but may require tuning")
    print("✓ Dual Score: Excellent inference accuracy, models selection bias")
    print("✓ All methods: Implement exact FICO whitepaper specifications")
    print()
    print("Files: Check fico_performance_inference_methods.py for full implementation")


if __name__ == "__main__":
    run_full_demo() 