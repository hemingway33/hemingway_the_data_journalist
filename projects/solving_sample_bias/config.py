"""
Configuration settings for the Rejection Inference Framework

This module centralizes all configuration parameters to make the framework
more maintainable and configurable.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""
    random_state: int = 42
    test_size: float = 0.3
    cv_folds: int = 5
    max_iter: int = 1000
    regularization_strength: float = 0.1
    propensity_weight_cap: float = 10.0
    ensemble_n_estimators: int = 100

@dataclass
class SimulationConfig:
    """Configuration for data simulation"""
    default_n_samples: int = 10000
    default_rejection_rate: float = 0.5
    feature_noise_std: float = 0.1
    correlation_strength: float = 0.3
    external_predictor_noise: float = 0.2

@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization"""
    figure_size: tuple = (15, 10)
    dpi: int = 300
    save_plots: bool = True
    plot_format: str = 'png'
    results_dir: str = 'results'
    verbose: bool = True

@dataclass
class BusinessConfig:
    """Configuration for business context analysis"""
    performance_weight: float = 0.4
    interpretability_weight: float = 0.3
    regulatory_weight: float = 0.2
    complexity_weight: float = 0.1
    confidence_threshold: float = 0.7

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.model = ModelConfig()
        self.simulation = SimulationConfig()
        self.analysis = AnalysisConfig()
        self.business = BusinessConfig()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON or YAML file"""
        # Implementation for loading from file
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'simulation': self.simulation.__dict__,
            'analysis': self.analysis.__dict__,
            'business': self.business.__dict__
        }

# Global configuration instance
config = Config() 