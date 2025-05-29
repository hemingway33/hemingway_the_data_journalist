"""
Comprehensive Comparison: FICO Whitepaper Methods vs Existing Methods

This script compares the exact FICO performance inference methods from the whitepaper
against the existing rejection inference methods in the codebase.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sample_simulation import SimulationScenario
from fico_performance_inference_methods import create_fico_methods, FICOMethodValidator
from rejection_inference_methods import create_rejection_inference_methods
from comparative_analysis import RejectionInferenceComparator

def run_comprehensive_comparison():
    """
    Run comprehensive comparison between FICO and existing methods
    """
    print("="*80)
    print("COMPREHENSIVE COMPARISON: FICO WHITEPAPER vs EXISTING METHODS")
    print("="*80)
    
    # Test multiple scenarios
    scenarios = {
        'moderate_rejection_external': SimulationScenario(
            name="Moderate Rejection + External Data",
            rejection_rate=0.5,
            n_samples=12000,
            external_predictor_strength=0.6,
            random_state=42
        ),
        'high_rejection_external': SimulationScenario(
            name="High Rejection + External Data", 
            rejection_rate=0.7,
            n_samples=12000,
            external_predictor_strength=0.6,
            random_state=42
        ),
        'moderate_rejection_no_external': SimulationScenario(
            name="Moderate Rejection + No External Data",
            rejection_rate=0.5,
            n_samples=12000,
            external_predictor_strength=0.0,
            random_state=42
        )
    }
    
    all_results = []
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario.name}")
        print(f"{'='*80}")
        
        # Generate data
        df, simulator = scenario.generate_complete_dataset()
        feature_cols = simulator.get_normalized_feature_columns()
        
        print(f"Dataset: {len(df)} samples, {(~df['approved']).mean():.1%} rejection rate")
        print(f"External data available: {scenario.external_predictor_strength > 0}")
        
        # Test FICO methods
        print(f"\n{'-'*50}")
        print("FICO WHITEPAPER METHODS")
        print(f"{'-'*50}")
        
        fico_validator = FICOMethodValidator(random_state=42)
        
        if scenario.external_predictor_strength > 0:
            # All FICO methods available
            fico_methods = create_fico_methods(random_state=42)
        else:
            # Only non-external FICO methods
            fico_methods = [m for m in create_fico_methods(random_state=42) 
                           if 'External Info' not in m.name]
        
        for method in fico_methods:
            try:
                result = fico_validator.validate_method(method, df, feature_cols)
                if result:
                    result['method_category'] = 'FICO Whitepaper'
                    result['scenario'] = scenario_name
                    all_results.append(result)
            except Exception as e:
                print(f"❌ FICO method {method.name} failed: {str(e)}")
        
        # Test existing methods
        print(f"\n{'-'*50}")
        print("EXISTING REJECTION INFERENCE METHODS")
        print(f"{'-'*50}")
        
        existing_methods = create_rejection_inference_methods(
            external_predictor_available=(scenario.external_predictor_strength > 0),
            random_state=42
        )
        
        comparator = RejectionInferenceComparator()
        existing_results = comparator.compare_methods(
            existing_methods, df, feature_cols, simulator
        )
        
        # Add to combined results
        for method_name, result in existing_results.items():
            if result is not None:
                result_dict = {
                    'method_name': result['method'].name,
                    'auc': result['auc'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'coefficient_mae': result['coefficient_mae'],
                    'interpretability_score': result['interpretability_score'],
                    'method_category': 'Existing Methods',
                    'scenario': scenario_name
                }
                all_results.append(result_dict)
    
    # Create comprehensive comparison DataFrame
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        
        # Save detailed results
        comparison_df.to_csv('fico_vs_existing_comprehensive_comparison.csv', index=False)
        
        # Create summary analysis
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Performance comparison by category
        print("\n1. AVERAGE PERFORMANCE BY METHOD CATEGORY:")
        print("-" * 50)
        
        agg_dict = {
            'auc': ['mean', 'std', 'count']
        }
        
        # Only add reject_inference_accuracy if it exists in the data
        if 'reject_inference_accuracy' in comparison_df.columns:
            agg_dict['reject_inference_accuracy'] = ['mean', 'std']
        
        perf_summary = comparison_df.groupby('method_category').agg(agg_dict).round(3)
        
        print(perf_summary)
        
        # Best performers overall
        print("\n2. TOP 10 METHODS OVERALL (by AUC):")
        print("-" * 50)
        
        display_columns = ['method_name', 'method_category', 'scenario', 'auc']
        if 'reject_inference_accuracy' in comparison_df.columns:
            display_columns.append('reject_inference_accuracy')
            
        top_methods = comparison_df.nlargest(10, 'auc')[display_columns]
        print(top_methods.to_string(index=False))
        
        # Scenario-specific analysis
        print("\n3. BEST METHOD PER SCENARIO:")
        print("-" * 50)
        
        for scenario in comparison_df['scenario'].unique():
            scenario_best = comparison_df[comparison_df['scenario'] == scenario].nlargest(1, 'auc')
            print(f"\n{scenario}:")
            print(f"  Method: {scenario_best['method_name'].iloc[0]}")
            print(f"  Category: {scenario_best['method_category'].iloc[0]}")
            print(f"  AUC: {scenario_best['auc'].iloc[0]:.3f}")
            if 'reject_inference_accuracy' in scenario_best.columns:
                print(f"  Inference Accuracy: {scenario_best['reject_inference_accuracy'].iloc[0]:.3f}")
        
        # Create visualizations
        create_comparison_plots(comparison_df)
        
        return comparison_df
    
    else:
        print("No results generated")
        return pd.DataFrame()


def create_comparison_plots(comparison_df):
    """Create visualization plots for the comparison"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FICO Whitepaper vs Existing Methods Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: AUC Distribution by Category
    ax1 = axes[0, 0]
    sns.boxplot(data=comparison_df, x='method_category', y='auc', ax=ax1)
    ax1.set_title('AUC Distribution by Method Category')
    ax1.set_ylabel('AUC Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Reject Inference Accuracy by Category
    ax2 = axes[0, 1]
    if 'reject_inference_accuracy' in comparison_df.columns:
        sns.boxplot(data=comparison_df, x='method_category', y='reject_inference_accuracy', ax=ax2)
        ax2.set_title('Reject Inference Accuracy by Category')
        ax2.set_ylabel('Inference Accuracy')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'Reject Inference\nAccuracy\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Performance by Scenario
    ax3 = axes[1, 0]
    scenario_perf = comparison_df.groupby(['scenario', 'method_category'])['auc'].mean().unstack()
    scenario_perf.plot(kind='bar', ax=ax3)
    ax3.set_title('Average AUC by Scenario and Method Category')
    ax3.set_ylabel('Average AUC')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Method Category')
    
    # Plot 4: Top Methods Scatter
    ax4 = axes[1, 1]
    for category in comparison_df['method_category'].unique():
        cat_data = comparison_df[comparison_df['method_category'] == category]
        ax4.scatter(cat_data['auc'], cat_data.get('reject_inference_accuracy', 0.5), 
                   label=category, alpha=0.7, s=60)
    
    ax4.set_xlabel('AUC Score')
    ax4.set_ylabel('Reject Inference Accuracy')
    ax4.set_title('AUC vs Reject Inference Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fico_vs_existing_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'fico_vs_existing_comparison.png'")
    plt.close()


def analyze_fico_method_details():
    """
    Detailed analysis of FICO method performance and characteristics
    """
    print(f"\n{'='*80}")
    print("DETAILED FICO METHOD ANALYSIS")
    print(f"{'='*80}")
    
    # Create scenario optimized for FICO methods
    scenario = SimulationScenario(
        name="FICO Method Analysis",
        rejection_rate=0.6,
        n_samples=15000,
        external_predictor_strength=0.7,
        random_state=42
    )
    
    df, simulator = scenario.generate_complete_dataset()
    feature_cols = simulator.get_normalized_feature_columns()
    
    print(f"Analysis dataset: {len(df)} samples")
    print(f"Rejection rate: {(~df['approved']).mean():.1%}")
    print(f"True reject default rate: {df[~df['approved']]['actual_default'].mean():.1%}")
    
    # Test each FICO method individually with detailed analysis
    fico_methods = create_fico_methods(random_state=42)
    validator = FICOMethodValidator(random_state=42)
    
    detailed_results = []
    
    for method in fico_methods:
        print(f"\n{'-'*60}")
        print(f"DETAILED ANALYSIS: {method.name}")
        print(f"{'-'*60}")
        
        try:
            result = validator.validate_method(method, df, feature_cols)
            if result:
                # Add method-specific insights
                print(f"\nMethod-specific statistics:")
                for key, value in result['inference_stats'].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"  {key}: {len(value)} entries")
                    elif isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                
                detailed_results.append({
                    'method': method.name,
                    'auc': result['auc'],
                    'inference_accuracy': result.get('reject_inference_accuracy', 'N/A'),
                    'stats': result['inference_stats']
                })
                
        except Exception as e:
            print(f"Failed: {str(e)}")
    
    # Save detailed analysis
    detailed_df = pd.DataFrame(detailed_results)
    if not detailed_df.empty:
        detailed_df.to_csv('fico_methods_detailed_analysis.csv', index=False)
        print(f"\nDetailed analysis saved to 'fico_methods_detailed_analysis.csv'")
    
    return detailed_df


if __name__ == "__main__":
    # Run comprehensive comparison
    comparison_results = run_comprehensive_comparison()
    
    # Run detailed FICO analysis
    fico_details = analyze_fico_method_details()
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print("Files generated:")
    print("1. fico_vs_existing_comprehensive_comparison.csv")
    print("2. fico_vs_existing_comparison.png")
    print("3. fico_methods_detailed_analysis.csv")
    print("4. fico_whitepaper_validation_results.csv") 