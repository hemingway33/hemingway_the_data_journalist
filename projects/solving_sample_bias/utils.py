"""
Utility functions for the Rejection Inference Framework

This module contains common helper functions used across multiple modules
to reduce code duplication and improve maintainability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import warnings

def calculate_performance_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                y_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
    y_pred : array-like, optional
        Predicted binary labels (will be computed if not provided)
        
    Returns:
    --------
    Dict with performance metrics
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0)
        }
    
    return metrics

def calculate_coefficient_bias(estimated_coefs: np.ndarray, 
                             true_coefs: np.ndarray) -> Dict[str, float]:
    """
    Calculate coefficient bias metrics
    
    Parameters:
    -----------
    estimated_coefs : array-like
        Estimated coefficients
    true_coefs : array-like
        True coefficients
        
    Returns:
    --------
    Dict with bias metrics
    """
    bias = estimated_coefs - true_coefs
    
    return {
        'mean_absolute_error': np.mean(np.abs(bias)),
        'mean_squared_error': np.mean(bias ** 2),
        'max_absolute_error': np.max(np.abs(bias)),
        'relative_bias': np.mean(np.abs(bias) / (np.abs(true_coefs) + 1e-8))
    }

def create_comparison_plot(results_df: pd.DataFrame, 
                          metric: str,
                          title: str = None,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a comparison bar plot for a specific metric
    
    Parameters:
    -----------
    results_df : DataFrame
        Results dataframe with methods as index
    metric : str
        Column name to plot
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric value
    sorted_df = results_df.sort_values(metric, ascending=False)
    
    # Create bar plot
    bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
    
    # Customize plot
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df.index, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'{metric.replace("_", " ").title()} Comparison')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def format_method_name(method_name: str) -> str:
    """
    Format method names for display
    
    Parameters:
    -----------
    method_name : str
        Raw method name
        
    Returns:
    --------
    Formatted method name
    """
    # Remove 'Method' suffix if present
    if method_name.endswith('Method'):
        method_name = method_name[:-6]
    
    # Add spaces before capital letters
    formatted = ''
    for i, char in enumerate(method_name):
        if i > 0 and char.isupper():
            formatted += ' '
        formatted += char
    
    return formatted

def validate_data_quality(df: pd.DataFrame, 
                         required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate data quality and return summary
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    required_columns : list
        List of required column names
        
    Returns:
    --------
    Dict with data quality metrics
    """
    quality_report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_columns': [col for col in required_columns if col not in df.columns],
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Calculate missing value percentages
    quality_report['missing_percentages'] = {
        col: (count / len(df)) * 100 
        for col, count in quality_report['missing_values'].items()
        if count > 0
    }
    
    return quality_report

def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Parameters:
    -----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float
        Default value if denominator is zero
        
    Returns:
    --------
    Division result or default
    """
    return numerator / denominator if denominator != 0 else default

def create_method_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary table of method results
    
    Parameters:
    -----------
    results : dict
        Results dictionary from comparative analysis
        
    Returns:
    --------
    DataFrame with method summary
    """
    summary_data = []
    
    for method_name, method_results in results.items():
        if isinstance(method_results, dict) and 'performance' in method_results:
            perf = method_results['performance']
            bias = method_results.get('bias', {})
            
            summary_data.append({
                'Method': format_method_name(method_name),
                'AUC': perf.get('auc', 0),
                'Accuracy': perf.get('accuracy', 0),
                'Precision': perf.get('precision', 0),
                'Recall': perf.get('recall', 0),
                'Coef_MAE': bias.get('mean_absolute_error', 0),
                'Coef_MSE': bias.get('mean_squared_error', 0)
            })
    
    return pd.DataFrame(summary_data).set_index('Method')

def print_section_header(title: str, width: int = 80, char: str = '='):
    """
    Print a formatted section header
    
    Parameters:
    -----------
    title : str
        Section title
    width : int
        Total width of header
    char : str
        Character to use for border
    """
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")

def save_results_to_file(results: Dict[str, Any], 
                        filename: str,
                        format: str = 'json'):
    """
    Save results to file in specified format
    
    Parameters:
    -----------
    results : dict
        Results to save
    filename : str
        Output filename
    format : str
        Output format ('json', 'pickle', 'csv')
    """
    import json
    import pickle
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_numpy_to_lists(results)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    elif format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'csv' and isinstance(results, pd.DataFrame):
        results.to_csv(filename)
    else:
        raise ValueError(f"Unsupported format: {format}")

def convert_numpy_to_lists(obj):
    """
    Recursively convert numpy arrays to lists for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    else:
        return obj 