"""
Comparative Analysis Module for Rejection Inference Methods

This module provides tools for comparing different rejection inference methods
across multiple dimensions including predictive performance, coefficient
interpretability, and business impact metrics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class RejectionInferenceComparator:
    """
    Main class for comparing rejection inference methods
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.test_data = None
        self.feature_cols = None
        self.true_coefficients = None
        self.simulator = None  # Store simulator reference
        
    def compare_methods(self, methods, df, feature_cols, simulator, test_size=0.3):
        """
        Compare multiple rejection inference methods
        
        Parameters:
        -----------
        methods : dict
            Dictionary of method_name -> method_instance
        df : DataFrame
            Complete dataset with all samples
        feature_cols : list
            List of feature column names
        simulator : LoanDataSimulator
            Simulator instance for ground truth comparison
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        Dict with comparison results for each method
        """
        self.simulator = simulator  # Store simulator reference
        self.feature_cols = feature_cols
        self.true_coefficients = simulator.get_true_coefficients_array()
        
        # Split data
        train_idx, test_idx = train_test_split(
            range(len(df)), 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=df['actual_default']
        )
        
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        self.test_data = df_test
        
        print(f"Training on {len(df_train):,} samples, testing on {len(df_test):,} samples")
        print(f"Test set approval rate: {df_test['approved'].mean():.1%}")
        print(f"Test set default rate: {df_test['actual_default'].mean():.1%}")
        
        # Fit and evaluate each method
        for method_name, method in methods.items():
            print(f"\nEvaluating {method.name}...")
            
            try:
                # Fit method
                method.fit(df_train, feature_cols, simulator)
                
                # Evaluate on test set
                X_test = df_test[feature_cols]
                y_test = df_test['actual_default']
                
                # Get predictions
                pred_proba = method.predict_proba(X_test)
                pred_binary = (pred_proba > 0.5).astype(int)
                
                # Calculate performance metrics
                auc = roc_auc_score(y_test, pred_proba)
                accuracy = accuracy_score(y_test, pred_binary)
                precision = precision_score(y_test, pred_binary, zero_division=0)
                recall = recall_score(y_test, pred_binary, zero_division=0)
                
                # Calculate coefficient metrics
                fitted_coefs = method.get_coefficients()
                coef_bias = np.array(fitted_coefs) - np.array(self.true_coefficients)
                coef_mae = np.mean(np.abs(coef_bias))
                coef_mse = np.mean(coef_bias**2)
                max_coef_bias = np.max(np.abs(coef_bias))
                
                # Calculate interpretability score (0-100, higher is better)
                max_possible_mae = 10.0
                interpretability_score = max(0, 100 * (1 - coef_mae / max_possible_mae))
                
                # Store results
                self.results[method_name] = {
                    'method': method,
                    'auc': auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'coefficients': fitted_coefs,
                    'coefficient_bias': coef_bias,
                    'coefficient_mae': coef_mae,
                    'coefficient_mse': coef_mse,
                    'max_coefficient_bias': max_coef_bias,
                    'interpretability_score': interpretability_score,
                    'predictions': pred_proba
                }
                
                print(f"  AUC: {auc:.3f}")
                print(f"  Coefficient MAE: {coef_mae:.3f}")
                print(f"  Interpretability Score: {interpretability_score:.1f}/100")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                self.results[method_name] = None
        
        return self.results
    
    def create_performance_summary(self):
        """
        Create a summary DataFrame of performance metrics
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_methods first.")
        
        summary_data = []
        for method_name, result in self.results.items():
            if result is not None:
                summary_data.append({
                    'Method': result['method'].name,
                    'AUC': result['auc'],
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'Coefficient_MAE': result['coefficient_mae'],
                    'Max_Coef_Bias': result['max_coefficient_bias'],
                    'Interpretability_Score': result['interpretability_score']
                })
        
        return pd.DataFrame(summary_data)
    
    def create_coefficient_comparison_table(self, simulator):
        """
        Create detailed coefficient comparison table
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_methods first.")
        
        feature_names = simulator.get_feature_names()
        
        # Create coefficient comparison DataFrame
        coef_df = pd.DataFrame()
        coef_df['True_Coefficients'] = self.true_coefficients
        coef_df.index = feature_names
        
        for method_name, result in self.results.items():
            if result is not None:
                coef_df[result['method'].name] = result['coefficients']
        
        return coef_df
    
    def create_bias_analysis(self):
        """
        Create bias analysis showing coefficient deviations from truth
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_methods first.")
        
        bias_data = []
        for method_name, result in self.results.items():
            if result is not None:
                bias_data.append({
                    'Method': result['method'].name,
                    'MSE': result['coefficient_mse'],
                    'MAE': result['coefficient_mae'],
                    'Max_Abs_Bias': result['max_coefficient_bias'],
                    'Interpretability_Score': result['interpretability_score']
                })
        
        return pd.DataFrame(bias_data)
    
    def analyze_label_assignment_quality(self, df):
        """
        Analyze quality of label assignment for rejection inference methods
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_methods first.")
        
        rejected_mask = ~df['approved']
        rejected_true_labels = df[rejected_mask]['actual_default'].values
        
        analysis_results = {}
        
        for method_name, result in self.results.items():
            if result is not None and 'rejection' in method_name.lower():
                # For rejection inference methods, we need to simulate their label assignment
                # This is a simplified analysis - in practice you'd want to access the actual assignments
                method_name_clean = result['method'].name
                
                if 'Simple' in method_name_clean:
                    # Simple rejection inference based on approval score
                    rejected_data = df[rejected_mask]
                    threshold = np.percentile(rejected_data['approval_score'], 30)
                    assigned_labels = (rejected_data['approval_score'] < threshold).astype(int)
                    
                    accuracy = (assigned_labels == rejected_true_labels).mean()
                    precision = ((assigned_labels == 1) & (rejected_true_labels == 1)).sum() / max((assigned_labels == 1).sum(), 1)
                    recall = ((assigned_labels == 1) & (rejected_true_labels == 1)).sum() / max((rejected_true_labels == 1).sum(), 1)
                    
                    analysis_results[method_name_clean] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'assigned_default_rate': assigned_labels.mean(),
                        'true_default_rate': rejected_true_labels.mean()
                    }
        
        return analysis_results
    
    def create_visualizations(self, output_prefix='rejection_inference_comparison'):
        """
        Create comprehensive visualizations comparing methods
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_methods first.")
        
        # Filter out None results
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            raise ValueError("No valid results to visualize.")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Rejection Inference Methods Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: AUC Comparison
        method_names = [result['method'].name for result in valid_results.values()]
        aucs = [result['auc'] for result in valid_results.values()]
        
        bars1 = axes[0, 0].bar(range(len(method_names)), aucs, alpha=0.7)
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].set_title('Model Performance (AUC)')
        axes[0, 0].set_xticks(range(len(method_names)))
        axes[0, 0].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, auc in zip(bars1, aucs):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Coefficient MAE Comparison
        coef_maes = [result['coefficient_mae'] for result in valid_results.values()]
        
        bars2 = axes[0, 1].bar(range(len(method_names)), coef_maes, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Coefficient MAE')
        axes[0, 1].set_title('Coefficient Accuracy (Lower is Better)')
        axes[0, 1].set_xticks(range(len(method_names)))
        axes[0, 1].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mae in zip(bars2, coef_maes):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(coef_maes)*0.02,
                           f'{mae:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Interpretability Score
        interp_scores = [result['interpretability_score'] for result in valid_results.values()]
        
        bars3 = axes[0, 2].bar(range(len(method_names)), interp_scores, alpha=0.7, color='green')
        axes[0, 2].set_xlabel('Method')
        axes[0, 2].set_ylabel('Interpretability Score')
        axes[0, 2].set_title('Interpretability Score (Higher is Better)')
        axes[0, 2].set_xticks(range(len(method_names)))
        axes[0, 2].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 100)
        
        # Add value labels
        for bar, score in zip(bars3, interp_scores):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{score:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Performance vs Interpretability Scatter
        axes[1, 0].scatter(coef_maes, aucs, s=100, alpha=0.7)
        for i, name in enumerate(method_names):
            axes[1, 0].annotate(name, (coef_maes[i], aucs[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Coefficient MAE (Lower is Better)')
        axes[1, 0].set_ylabel('AUC (Higher is Better)')
        axes[1, 0].set_title('Performance vs Interpretability Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Coefficient Bias Heatmap
        if self.simulator is not None:
            try:
                coef_df = self.create_coefficient_comparison_table(self.simulator)
                # Use the actual coefficient comparison data
                sns.heatmap(coef_df.iloc[:, 1:], annot=True, cmap='RdBu_r', center=0, 
                           cbar_kws={'label': 'Coefficient Bias'}, ax=axes[1, 1], fmt='.2f')
                axes[1, 1].set_title('Coefficient Bias Heatmap')
                axes[1, 1].set_xlabel('Feature')
                axes[1, 1].set_ylabel('Method')
            except Exception as e:
                print(f"Warning: Could not create coefficient comparison table: {e}")
                # Fallback to simplified version
                self._create_simplified_bias_heatmap(axes[1, 1], valid_results)
        else:
            # Fallback to simplified version
            self._create_simplified_bias_heatmap(axes[1, 1], valid_results)
        
        # Plot 6: Overall Score (Weighted combination)
        # Combine AUC (weight 0.4) and Interpretability (weight 0.6) for balanced score
        combined_scores = []
        for result in valid_results.values():
            # Normalize AUC to 0-100 scale (assuming AUC ranges from 0.5 to 1.0)
            normalized_auc = ((result['auc'] - 0.5) / 0.5) * 100
            # Weight: 40% performance, 60% interpretability
            combined_score = 0.4 * normalized_auc + 0.6 * result['interpretability_score']
            combined_scores.append(combined_score)
        
        bars6 = axes[1, 2].bar(range(len(method_names)), combined_scores, alpha=0.7, color='purple')
        axes[1, 2].set_xlabel('Method')
        axes[1, 2].set_ylabel('Combined Score')
        axes[1, 2].set_title('Overall Score (40% Performance + 60% Interpretability)')
        axes[1, 2].set_xticks(range(len(method_names)))
        axes[1, 2].set_xticklabels(method_names, rotation=45, ha='right')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 100)
        
        # Add value labels
        for bar, score in zip(bars6, combined_scores):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{score:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Ensure the output filename has .png extension
        if not output_prefix.endswith('.png'):
            output_filename = f'{output_prefix}_comprehensive.png'
        else:
            output_filename = output_prefix
            
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as '{output_filename}'")
        
        return fig
    
    def print_detailed_summary(self):
        """
        Print comprehensive summary of comparison results
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_methods first.")
        
        print("\n" + "="*80)
        print("COMPREHENSIVE REJECTION INFERENCE METHODS COMPARISON")
        print("="*80)
        
        # Performance Summary
        summary_df = self.create_performance_summary()
        print("\nPERFORMACE SUMMARY:")
        print("-" * 60)
        print(summary_df.round(3).to_string(index=False))
        
        # Ranking by different criteria
        print("\nRANKINGS:")
        print("-" * 60)
        
        # Best AUC
        best_auc = summary_df.loc[summary_df['AUC'].idxmax()]
        print(f"Best AUC: {best_auc['Method']} ({best_auc['AUC']:.3f})")
        
        # Best Interpretability
        best_interp = summary_df.loc[summary_df['Interpretability_Score'].idxmax()]
        print(f"Best Interpretability: {best_interp['Method']} ({best_interp['Interpretability_Score']:.1f}/100)")
        
        # Best Coefficient Accuracy
        best_coef = summary_df.loc[summary_df['Coefficient_MAE'].idxmin()]
        print(f"Best Coefficient Accuracy: {best_coef['Method']} (MAE: {best_coef['Coefficient_MAE']:.3f})")
        
        # Balanced Score
        summary_df['Balanced_Score'] = (
            0.4 * ((summary_df['AUC'] - 0.5) / 0.5 * 100) + 
            0.6 * summary_df['Interpretability_Score']
        )
        best_balanced = summary_df.loc[summary_df['Balanced_Score'].idxmax()]
        print(f"Best Balanced Score: {best_balanced['Method']} ({best_balanced['Balanced_Score']:.1f}/100)")
        
        # Bias Analysis
        bias_df = self.create_bias_analysis()
        print(f"\nCOEFFICIENT BIAS ANALYSIS:")
        print("-" * 60)
        print(bias_df.round(4).to_string(index=False))
        
        return summary_df, bias_df
    
    def recommend_method(self, priority='balanced'):
        """
        Recommend best method based on specified priority
        
        Parameters:
        -----------
        priority : str
            'performance' - prioritize AUC
            'interpretability' - prioritize coefficient accuracy
            'balanced' - balanced combination
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_methods first.")
        
        summary_df = self.create_performance_summary()
        
        if priority == 'performance':
            best_idx = summary_df['AUC'].idxmax()
            criterion = f"AUC: {summary_df.loc[best_idx, 'AUC']:.3f}"
        elif priority == 'interpretability':
            best_idx = summary_df['Interpretability_Score'].idxmax()
            criterion = f"Interpretability: {summary_df.loc[best_idx, 'Interpretability_Score']:.1f}/100"
        elif priority == 'balanced':
            # 40% performance, 60% interpretability
            summary_df['Balanced_Score'] = (
                0.4 * ((summary_df['AUC'] - 0.5) / 0.5 * 100) + 
                0.6 * summary_df['Interpretability_Score']
            )
            best_idx = summary_df['Balanced_Score'].idxmax()
            criterion = f"Balanced Score: {summary_df.loc[best_idx, 'Balanced_Score']:.1f}/100"
        else:
            raise ValueError("Priority must be 'performance', 'interpretability', or 'balanced'")
        
        recommended_method = summary_df.loc[best_idx, 'Method']
        
        print(f"\nRECOMMENDED METHOD ({priority.upper()} PRIORITY):")
        print("-" * 60)
        print(f"Method: {recommended_method}")
        print(f"Criterion: {criterion}")
        print(f"AUC: {summary_df.loc[best_idx, 'AUC']:.3f}")
        print(f"Coefficient MAE: {summary_df.loc[best_idx, 'Coefficient_MAE']:.3f}")
        print(f"Interpretability Score: {summary_df.loc[best_idx, 'Interpretability_Score']:.1f}/100")
        
        return recommended_method
    
    def _create_simplified_bias_heatmap(self, ax, valid_results):
        """
        Create a simplified bias heatmap when full coefficient comparison is not available
        """
        # Create simplified bias heatmap
        bias_matrix = []
        feature_names = ['Intercept', 'Credit Score', 'Income', 'DTI', 'Employment', 'Loan Amount', 'Credit Lines']
        
        for result in valid_results.values():
            bias_matrix.append(result['coefficient_bias'])
        
        bias_df = pd.DataFrame(bias_matrix, 
                              index=[result['method'].name for result in valid_results.values()],
                              columns=feature_names)
        
        sns.heatmap(bias_df, annot=True, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Coefficient Bias'}, ax=ax, fmt='.2f')
        ax.set_title('Coefficient Bias Heatmap (Simplified)')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Method') 