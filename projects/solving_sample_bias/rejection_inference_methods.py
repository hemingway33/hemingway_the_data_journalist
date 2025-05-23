"""
Rejection Inference Methods Module

This module implements various approaches to handle sample selection bias
in consumer loan data, from conservative methods to advanced external
predictor utilization.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class RejectionInferenceMethod(ABC):
    """
    Abstract base class for rejection inference methods
    """
    
    def __init__(self, name, random_state=42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, df, feature_cols, simulator):
        """
        Fit the rejection inference method to the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset with outcomes and approval status
        feature_cols : list
            List of feature column names to use
        simulator : LoanDataSimulator
            Simulator object for accessing ground truth
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict probabilities using the fitted model
        """
        pass
    
    def get_coefficients(self):
        """
        Extract model coefficients for comparison
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted first")
        
        return [self.model.intercept_[0]] + list(self.model.coef_[0])


class ApprovedOnlyMethod(RejectionInferenceMethod):
    """
    Traditional approach: Train only on approved samples with observed outcomes
    """
    
    def __init__(self, regularization_strength=1.0, random_state=42):
        super().__init__("Approved Only", random_state)
        self.regularization_strength = regularization_strength
        
    def fit(self, df, feature_cols, simulator):
        # Use only approved samples
        approved_data = df[df['approved']].copy()
        
        X = approved_data[feature_cols]
        y = approved_data['observed_default']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=self.regularization_strength
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class RegularizedApprovedMethod(ApprovedOnlyMethod):
    """
    Regularized version of approved-only method to reduce overfitting
    """
    
    def __init__(self, regularization_strength=0.1, random_state=42):
        super().__init__(regularization_strength, random_state)
        self.name = f"Regularized Approved (C={regularization_strength})"


class SimpleRejectionInferenceMethod(RejectionInferenceMethod):
    """
    Simple rejection inference: Assign labels to rejected samples based on approval score
    """
    
    def __init__(self, default_assignment_percentile=30, random_state=42):
        super().__init__("Simple Rejection Inference", random_state)
        self.default_assignment_percentile = default_assignment_percentile
        
    def fit(self, df, feature_cols, simulator):
        # Create augmented dataset
        df_augmented = df.copy()
        rejected_data = df_augmented[~df_augmented['approved']]
        
        # Assign labels to rejected samples based on approval score
        rejection_threshold = np.percentile(
            rejected_data['approval_score'], 
            self.default_assignment_percentile
        )
        
        # Create augmented labels
        augmented_defaults = df_augmented['observed_default'].copy()
        rejected_mask = ~df_augmented['approved']
        
        # Assign high default probability to worst rejected applicants
        worst_rejected = (rejected_mask & 
                         (df_augmented['approval_score'] < rejection_threshold))
        augmented_defaults[worst_rejected] = 1
        
        # Assign low default probability to better rejected applicants
        better_rejected = (rejected_mask & 
                          (df_augmented['approval_score'] >= rejection_threshold))
        augmented_defaults[better_rejected] = 0
        
        # Train on full augmented dataset
        X = df_augmented[feature_cols]
        y = augmented_defaults
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class PropensityWeightingMethod(RejectionInferenceMethod):
    """
    Propensity score weighting: Weight approved samples by inverse probability of approval
    """
    
    def __init__(self, weight_cap_percentile=95, random_state=42):
        super().__init__("Propensity Weighting", random_state)
        self.weight_cap_percentile = weight_cap_percentile
        
    def fit(self, df, feature_cols, simulator):
        # Train approval model for propensity scores
        approval_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Scale features for approval model
        approval_scaler = StandardScaler()
        X_full = approval_scaler.fit_transform(df[feature_cols])
        approval_model.fit(X_full, df['approved'])
        
        # Calculate propensity scores
        propensity_scores = approval_model.predict_proba(X_full)[:, 1]
        
        # Use only approved samples but weight them
        approved_data = df[df['approved']].copy()
        
        # Get propensity scores for approved samples using boolean indexing
        approved_propensity_scores = propensity_scores[df['approved']]
        
        # Calculate weights for approved samples
        weights = 1 / approved_propensity_scores
        
        # Cap extreme weights
        if self.weight_cap_percentile < 100:
            weight_cap = np.percentile(weights, self.weight_cap_percentile)
            weights = np.minimum(weights, weight_cap)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Train weighted model
        X = approved_data[feature_cols]
        y = approved_data['observed_default']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with weights
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class ConservativePropensityMethod(PropensityWeightingMethod):
    """
    Conservative propensity weighting with stricter weight capping
    """
    
    def __init__(self, weight_cap_percentile=90, regularization_strength=0.5, random_state=42):
        super().__init__(weight_cap_percentile, random_state)
        self.name = "Conservative Propensity (Capped)"
        self.regularization_strength = regularization_strength
        
    def fit(self, df, feature_cols, simulator):
        # Call parent method but with regularization
        super().fit(df, feature_cols, simulator)
        
        # Refit with regularization
        approved_data = df[df['approved']].copy()
        X = approved_data[feature_cols]
        y = approved_data['observed_default']
        X_scaled = self.scaler.transform(X)
        
        # Calculate weights (same as parent)
        approval_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        approval_scaler = StandardScaler()
        X_full = approval_scaler.fit_transform(df[feature_cols])
        approval_model.fit(X_full, df['approved'])
        propensity_scores = approval_model.predict_proba(X_full)[:, 1]
        
        # Get propensity scores for approved samples using boolean indexing
        approved_propensity_scores = propensity_scores[df['approved']]
        weights = 1 / approved_propensity_scores
        weight_cap = np.percentile(weights, self.weight_cap_percentile)
        weights = np.minimum(weights, weight_cap)
        weights = weights / weights.mean()
        
        # Train regularized model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=self.regularization_strength
        )
        self.model.fit(X_scaled, y, sample_weight=weights)


class EnsembleAveragingMethod(RejectionInferenceMethod):
    """
    Ensemble averaging: Train multiple models on subsamples and average coefficients
    """
    
    def __init__(self, n_models=10, subsample_ratio=0.8, random_state=42):
        super().__init__("Ensemble Averaging", random_state)
        self.n_models = n_models
        self.subsample_ratio = subsample_ratio
        self.models = []
        
    def fit(self, df, feature_cols, simulator):
        approved_data = df[df['approved']].copy()
        approved_indices = approved_data.index.tolist()
        
        X_full = approved_data[feature_cols]
        y_full = approved_data['observed_default']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled_full = self.scaler.fit_transform(X_full)
        
        # Train multiple models on subsamples
        self.models = []
        for i in range(self.n_models):
            # Random subsample
            np.random.seed(self.random_state + i)
            subsample_size = int(self.subsample_ratio * len(approved_indices))
            subsample_idx = np.random.choice(
                len(approved_indices), 
                size=subsample_size, 
                replace=False
            )
            
            X_sub = X_scaled_full[subsample_idx]
            y_sub = y_full.iloc[subsample_idx]
            
            # Train model
            model = LogisticRegression(
                random_state=self.random_state + i,
                max_iter=1000,
                C=0.5
            )
            model.fit(X_sub, y_sub)
            self.models.append(model)
        
        # Create averaged model for prediction
        avg_intercept = np.mean([m.intercept_[0] for m in self.models])
        avg_coefs = np.mean([m.coef_[0] for m in self.models], axis=0)
        
        # Create final model with averaged coefficients
        self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.model.fit(X_scaled_full, y_full)  # Fit on full data for shape
        self.model.intercept_ = np.array([avg_intercept])
        self.model.coef_ = np.array([avg_coefs])
        
        self.is_fitted = True
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class ExternalPredictorMethod(RejectionInferenceMethod):
    """
    External predictor enhanced rejection inference
    """
    
    def __init__(self, high_risk_percentile=85, low_risk_percentile=50, random_state=42):
        super().__init__("External Predictor Enhanced", random_state)
        self.high_risk_percentile = high_risk_percentile
        self.low_risk_percentile = low_risk_percentile
        
    def fit(self, df, feature_cols, simulator):
        if 'external_predictor' not in df.columns:
            raise ValueError("External predictor column not found in dataset")
        
        # Create augmented dataset using external predictor
        df_augmented = df.copy()
        rejected_mask = ~df_augmented['approved']
        rejected_data = df_augmented[rejected_mask]
        
        # Set thresholds based on external predictor
        external_threshold_high = np.percentile(
            rejected_data['external_predictor'], 
            self.high_risk_percentile
        )
        external_threshold_low = np.percentile(
            rejected_data['external_predictor'], 
            self.low_risk_percentile
        )
        
        # Assign labels based on external predictor
        augmented_defaults = df_augmented['observed_default'].copy()
        
        # High risk rejected samples
        high_risk_rejected = (rejected_mask & 
                             (df_augmented['external_predictor'] > external_threshold_high))
        # Low risk rejected samples  
        low_risk_rejected = (rejected_mask & 
                            (df_augmented['external_predictor'] <= external_threshold_low))
        # Medium risk rejected samples
        medium_risk_rejected = (rejected_mask & 
                               (df_augmented['external_predictor'] > external_threshold_low) & 
                               (df_augmented['external_predictor'] <= external_threshold_high))
        
        augmented_defaults[high_risk_rejected] = 1
        augmented_defaults[low_risk_rejected] = 0
        
        # For medium risk, use probabilistic assignment
        if medium_risk_rejected.sum() > 0:
            medium_scores = df_augmented[medium_risk_rejected]['external_predictor']
            medium_probs = ((medium_scores - external_threshold_low) / 
                           (external_threshold_high - external_threshold_low))
            np.random.seed(self.random_state)
            medium_assignments = np.random.binomial(1, medium_probs)
            augmented_defaults[medium_risk_rejected] = medium_assignments
        
        # Train on augmented dataset
        X = df_augmented[feature_cols]
        y = augmented_defaults
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class HybridExternalPropensityMethod(RejectionInferenceMethod):
    """
    Hybrid approach: External predictor for label assignment + propensity weighting
    """
    
    def __init__(self, high_risk_percentile=85, low_risk_percentile=50, 
                 weight_cap_percentile=95, rejected_weight=0.5, random_state=42):
        super().__init__("Hybrid (External + Propensity)", random_state)
        self.high_risk_percentile = high_risk_percentile
        self.low_risk_percentile = low_risk_percentile
        self.weight_cap_percentile = weight_cap_percentile
        self.rejected_weight = rejected_weight
        
    def fit(self, df, feature_cols, simulator):
        if 'external_predictor' not in df.columns:
            raise ValueError("External predictor column not found in dataset")
        
        # First, use external predictor method for label assignment
        external_method = ExternalPredictorMethod(
            self.high_risk_percentile, 
            self.low_risk_percentile, 
            self.random_state
        )
        external_method.fit(df, feature_cols, simulator)
        
        # Get augmented labels
        rejected_mask = ~df['approved']
        rejected_data = df[rejected_mask]
        external_threshold_high = np.percentile(
            rejected_data['external_predictor'], 
            self.high_risk_percentile
        )
        external_threshold_low = np.percentile(
            rejected_data['external_predictor'], 
            self.low_risk_percentile
        )
        
        augmented_defaults = df['observed_default'].copy()
        high_risk_rejected = (rejected_mask & 
                             (df['external_predictor'] > external_threshold_high))
        low_risk_rejected = (rejected_mask & 
                            (df['external_predictor'] <= external_threshold_low))
        medium_risk_rejected = (rejected_mask & 
                               (df['external_predictor'] > external_threshold_low) & 
                               (df['external_predictor'] <= external_threshold_high))
        
        augmented_defaults[high_risk_rejected] = 1
        augmented_defaults[low_risk_rejected] = 0
        
        if medium_risk_rejected.sum() > 0:
            medium_scores = df[medium_risk_rejected]['external_predictor']
            medium_probs = ((medium_scores - external_threshold_low) / 
                           (external_threshold_high - external_threshold_low))
            np.random.seed(self.random_state)
            medium_assignments = np.random.binomial(1, medium_probs)
            augmented_defaults[medium_risk_rejected] = medium_assignments
        
        # Calculate propensity weights
        approval_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        approval_scaler = StandardScaler()
        X_full = approval_scaler.fit_transform(df[feature_cols])
        approval_model.fit(X_full, df['approved'])
        propensity_scores = approval_model.predict_proba(X_full)[:, 1]
        
        # Create sample weights
        all_weights = np.ones(len(df))
        approved_mask = df['approved']
        
        # Weight approved samples by inverse propensity
        approved_weights = 1 / propensity_scores[approved_mask]
        weight_cap = np.percentile(approved_weights, self.weight_cap_percentile)
        approved_weights = np.minimum(approved_weights, weight_cap)
        approved_weights = approved_weights / approved_weights.mean()
        all_weights[approved_mask] = approved_weights
        
        # Lower weight for artificially labeled rejected samples
        all_weights[rejected_mask] = self.rejected_weight
        
        # Train model
        X = df[feature_cols]
        y = augmented_defaults
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train weighted model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.model.fit(X_scaled, y, sample_weight=all_weights)
        self.is_fitted = True
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class OracleMethod(RejectionInferenceMethod):
    """
    Oracle method: Train on full population with true outcomes (for comparison)
    """
    
    def __init__(self, random_state=42):
        super().__init__("Oracle (Full Data)", random_state)
        
    def fit(self, df, feature_cols, simulator):
        # Use full population with true outcomes
        X = df[feature_cols]
        y = df['actual_default']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


# Method factory for easy instantiation
def create_rejection_inference_methods(external_predictor_available=False, random_state=42):
    """
    Create a standard set of rejection inference methods for comparison
    
    Parameters:
    -----------
    external_predictor_available : bool
        Whether external predictor data is available
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    dict : Dictionary of method_name -> method_instance
    """
    methods = {
        'approved_only': ApprovedOnlyMethod(random_state=random_state),
        'regularized_approved': RegularizedApprovedMethod(random_state=random_state),
        'simple_rejection_inference': SimpleRejectionInferenceMethod(random_state=random_state),
        'propensity_weighting': PropensityWeightingMethod(random_state=random_state),
        'conservative_propensity': ConservativePropensityMethod(random_state=random_state),
        'ensemble_averaging': EnsembleAveragingMethod(random_state=random_state),
        'oracle': OracleMethod(random_state=random_state)
    }
    
    if external_predictor_available:
        methods.update({
            'external_predictor': ExternalPredictorMethod(random_state=random_state),
            'hybrid_external_propensity': HybridExternalPropensityMethod(random_state=random_state)
        })
    
    return methods 