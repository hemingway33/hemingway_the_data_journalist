"""
FICO Performance Inference Methods for Reject Samples

This module implements the exact FICO methods for inferring performance
of rejected samples as described in the "Building Powerful Scorecards" whitepaper.

Methods implemented based on FICO documentation:
1. External Information Method - using credit bureau scores  
2. Domain Expertise Parceling Method - iterative parceling with viability testing
3. Dual Score Inference - combining KN_SCORE and AR_SCORE

References:
- FICO "Building Powerful Scorecards" whitepaper (2014)
- Performance Inference section and methodology
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from abc import ABC, abstractmethod
import warnings
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


class FICOPerformanceInferenceMethod(ABC):
    """
    Abstract base class for FICO performance inference methods
    """
    
    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.inference_stats = {}
        
    @abstractmethod
    def infer_reject_performance(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Infer performance for rejected samples"""
        pass
    
    @abstractmethod
    def fit_scorecard(self, df_augmented: pd.DataFrame, feature_cols: List[str]) -> None:
        """Fit scorecard model on augmented dataset"""
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class ExternalInformationMethod(FICOPerformanceInferenceMethod):
    """
    FICO External Information Method
    
    Uses credit bureau (CB) score to infer reject performance following FICO methodology:
    1. Fit logistic model: logOdds = B0 + B1*CB_SCORE on known population
    2. Compute probability pG for unknowns: pG = 1/(1 + exp(-(B0 + B1*CB_SCORE)))
    3. Use pG to assign credible performance to rejected applicants
    
    This matches the exact approach described in the FICO whitepaper.
    """
    
    def __init__(self, external_score_col: str = 'external_predictor', 
                 assignment_threshold: float = 0.5, random_state: int = 42):
        super().__init__("FICO External Information", random_state)
        self.external_score_col = external_score_col
        self.assignment_threshold = assignment_threshold
        self.cb_model = None
        self.B0 = None  # Intercept coefficient
        self.B1 = None  # CB_SCORE coefficient
        
    def infer_reject_performance(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Apply FICO External Information method
        """
        if self.external_score_col not in df.columns:
            raise ValueError(f"External score column '{self.external_score_col}' not found in dataset")
            
        df_augmented = df.copy()
        
        # Step 1: Fit logistic model on known population (approved samples)
        # logOdds = B0 + B1*CB_SCORE
        approved_data = df_augmented[df_augmented['approved']].copy()
        
        if len(approved_data) == 0:
            raise ValueError("No approved samples available for training")
            
        X_cb = approved_data[[self.external_score_col]]
        y_cb = approved_data['observed_default']
        
        # Fit the CB score model
        self.cb_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.cb_model.fit(X_cb, y_cb)
        
        # Extract coefficients B0 and B1
        self.B0 = self.cb_model.intercept_[0]
        self.B1 = self.cb_model.coef_[0][0]
        
        # Step 2: Compute pG for rejected samples
        # pG = 1 / (1 + exp(-(B0 + B1*CB_SCORE)))
        rejected_mask = ~df_augmented['approved']
        rejected_cb_scores = df_augmented.loc[rejected_mask, self.external_score_col]
        
        # Calculate logOdds for rejects
        reject_logodds = self.B0 + self.B1 * rejected_cb_scores
        
        # Calculate pG (probability of Good performance)
        pG = 1 / (1 + np.exp(-reject_logodds))
        
        # Step 3: Assign performance based on pG
        df_augmented['inferred_default'] = df_augmented['observed_default'].copy()
        
        # Assign performance: if pG > threshold, assign Good (0), else Bad (1)
        reject_indices = df_augmented.index[rejected_mask]
        good_rejects = pG > self.assignment_threshold
        bad_rejects = pG <= self.assignment_threshold
        
        df_augmented.loc[reject_indices[good_rejects], 'inferred_default'] = 0  # Good
        df_augmented.loc[reject_indices[bad_rejects], 'inferred_default'] = 1   # Bad
        
        # Store inference statistics
        self.inference_stats = {
            'B0_intercept': self.B0,
            'B1_cb_coefficient': self.B1,
            'n_rejects_assigned_good': good_rejects.sum(),
            'n_rejects_assigned_bad': bad_rejects.sum(),
            'avg_pG_rejects': pG.mean(),
            'external_score_col': self.external_score_col
        }
        
        return df_augmented
    
    def fit_scorecard(self, df_augmented: pd.DataFrame, feature_cols: List[str]) -> None:
        """Fit final scorecard model on augmented TTD population"""
        X = df_augmented[feature_cols]
        y = df_augmented['inferred_default']
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.model.fit(X_scaled, y)
        self.is_fitted = True


class DomainExpertiseParcelingMethod(FICOPerformanceInferenceMethod):
    """
    FICO Domain Expertise Parceling Method
    
    Implements the iterative parceling process described in FICO whitepaper:
    1. Craft credible KN_SCORE on known population using domain knowledge
    2. Use KN_SCORE to assign initial performance: logOdds = C0 + C1*KN_SCORE
    3. Train new scoring model T on full TTD population
    4. Test viability by comparing log(Odds) across known/unknown populations
    5. If not aligned, estimate new slope/intercept and iterate
    6. Continue until odds-to-score fits converge
    
    This is the core FICO parceling methodology.
    """
    
    def __init__(self, max_iterations: int = 10, convergence_threshold: float = 0.05,
                 random_state: int = 42):
        super().__init__("FICO Domain Expertise Parceling", random_state)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.kn_score_model = None
        self.C0 = None  # KN_SCORE intercept
        self.C1 = None  # KN_SCORE coefficient
        self.parceling_history = []
        self.final_T_model = None
        
    def _create_kn_score(self, df_known: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Create KN_SCORE using domain expertise and feature engineering
        This is the "parcel score" that drives initial assignment
        """
        # For demonstration, we'll create a sophisticated KN_SCORE
        # In practice, this would use domain knowledge and careful engineering
        
        # Build initial model on known data with regularization for stability
        X_known = df_known[feature_cols]
        y_known = df_known['observed_default']
        
        scaler = StandardScaler()
        X_known_scaled = scaler.fit_transform(X_known)
        
        # Use regularized logistic regression for stability
        kn_model = LogisticRegression(
            random_state=self.random_state, 
            max_iter=1000,
            C=1.0,  # Moderate regularization
            penalty='l2'
        )
        kn_model.fit(X_known_scaled, y_known)
        
        # Generate KN_SCORE for entire dataset
        self.kn_score_model = (kn_model, scaler)
        
        return kn_model, scaler
    
    def _calculate_kn_scores(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Calculate KN_SCORE for all samples"""
        kn_model, scaler = self.kn_score_model
        X_scaled = scaler.transform(df[feature_cols])
        
        # Use log-odds as the KN_SCORE (this is the score used for parceling)
        log_odds = kn_model.decision_function(X_scaled)
        return log_odds
    
    def _test_viability(self, df_augmented: pd.DataFrame, feature_cols: List[str], 
                       iteration: int) -> Tuple[bool, float, Dict]:
        """
        Test viability of current inference by comparing log(Odds) alignment
        between known and unknown populations
        """
        # Train new scoring model T on full TTD population
        X_ttd = df_augmented[feature_cols]
        y_ttd = df_augmented['inferred_default']
        
        scaler_T = StandardScaler()
        X_ttd_scaled = scaler_T.fit_transform(X_ttd)
        
        T_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        T_model.fit(X_ttd_scaled, y_ttd)
        
        # Calculate log(Odds) for known and unknown populations separately
        known_mask = df_augmented['approved']
        unknown_mask = ~df_augmented['approved']
        
        # Get T scores (log-odds from model T)
        T_scores_known = T_model.decision_function(X_ttd_scaled[known_mask])
        T_scores_unknown = T_model.decision_function(X_ttd_scaled[unknown_mask])
        
        # Get actual outcomes
        y_known = df_augmented.loc[known_mask, 'inferred_default']
        y_unknown = df_augmented.loc[unknown_mask, 'inferred_default']
        
        # Fit separate linear models to test alignment
        # For known population
        if len(np.unique(y_known)) > 1:
            known_lr = LogisticRegression(random_state=self.random_state)
            known_lr.fit(T_scores_known.reshape(-1, 1), y_known)
            known_slope = known_lr.coef_[0][0]
            known_intercept = known_lr.intercept_[0]
        else:
            known_slope, known_intercept = 0, 0
            
        # For unknown population  
        if len(np.unique(y_unknown)) > 1:
            unknown_lr = LogisticRegression(random_state=self.random_state)
            unknown_lr.fit(T_scores_unknown.reshape(-1, 1), y_unknown)
            unknown_slope = unknown_lr.coef_[0][0]
            unknown_intercept = unknown_lr.intercept_[0]
        else:
            unknown_slope, unknown_intercept = 0, 0
        
        # Calculate alignment metrics
        slope_diff = abs(known_slope - unknown_slope)
        intercept_diff = abs(known_intercept - unknown_intercept)
        alignment_score = slope_diff + intercept_diff
        
        # Test for convergence
        is_viable = alignment_score < self.convergence_threshold
        
        viability_stats = {
            'iteration': iteration,
            'known_slope': known_slope,
            'known_intercept': known_intercept,
            'unknown_slope': unknown_slope,
            'unknown_intercept': unknown_intercept,
            'slope_difference': slope_diff,
            'intercept_difference': intercept_diff,
            'alignment_score': alignment_score,
            'is_viable': is_viable
        }
        
        # Store the final T model
        self.final_T_model = (T_model, scaler_T)
        
        return is_viable, alignment_score, viability_stats
    
    def infer_reject_performance(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Apply FICO Domain Expertise Parceling method
        """
        df_augmented = df.copy()
        
        # Step 1: Create KN_SCORE on known population
        known_data = df_augmented[df_augmented['approved']].copy()
        if len(known_data) == 0:
            raise ValueError("No known samples available")
            
        kn_model, kn_scaler = self._create_kn_score(known_data, feature_cols)
        self.kn_score_model = (kn_model, kn_scaler)
        
        # Step 2: Calculate KN_SCORE for all samples
        kn_scores = self._calculate_kn_scores(df_augmented, feature_cols)
        
        # Step 3: Initial assignment using KN_SCORE
        # logOdds = C0 + C1*KN_SCORE, pG' = 1/(1 + exp(-(C0 + C1*KN_SCORE)))
        
        # Fit model to relate KN_SCORE to outcomes on known data
        known_kn_scores = kn_scores[df_augmented['approved']]
        known_outcomes = known_data['observed_default']
        
        kn_assignment_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        kn_assignment_model.fit(known_kn_scores.reshape(-1, 1), known_outcomes)
        
        self.C0 = kn_assignment_model.intercept_[0]
        self.C1 = kn_assignment_model.coef_[0][0]
        
        # Initial assignment for rejects
        rejected_mask = ~df_augmented['approved']
        rejected_kn_scores = kn_scores[rejected_mask]
        
        # Calculate pG' for rejects
        reject_logodds = self.C0 + self.C1 * rejected_kn_scores
        pG_prime = 1 / (1 + np.exp(-reject_logodds))
        
        # Initial assignment
        df_augmented['inferred_default'] = df_augmented['observed_default'].copy()
        reject_indices = df_augmented.index[rejected_mask]
        
        # Assign based on pG' > 0.5 threshold
        good_rejects = pG_prime > 0.5
        bad_rejects = pG_prime <= 0.5
        
        df_augmented.loc[reject_indices[good_rejects], 'inferred_default'] = 0
        df_augmented.loc[reject_indices[bad_rejects], 'inferred_default'] = 1
        
        # Step 4: Iterative parceling with viability testing
        for iteration in range(self.max_iterations):
            is_viable, alignment_score, viability_stats = self._test_viability(
                df_augmented, feature_cols, iteration + 1
            )
            
            self.parceling_history.append(viability_stats)
            
            if is_viable:
                print(f"Convergence achieved at iteration {iteration + 1}")
                break
                
            # If not viable, update assignments and continue
            # Use the T model to update reject assignments
            if self.final_T_model is not None:
                T_model, T_scaler = self.final_T_model
                X_rejects = df_augmented.loc[rejected_mask, feature_cols]
                X_rejects_scaled = T_scaler.transform(X_rejects)
                
                # Get updated probabilities
                updated_probs = T_model.predict_proba(X_rejects_scaled)[:, 1]
                
                # Update assignments
                df_augmented.loc[reject_indices[updated_probs > 0.5], 'inferred_default'] = 1
                df_augmented.loc[reject_indices[updated_probs <= 0.5], 'inferred_default'] = 0
        
        # Store final statistics
        self.inference_stats = {
            'C0_intercept': self.C0,
            'C1_kn_coefficient': self.C1,
            'n_iterations': len(self.parceling_history),
            'final_alignment_score': self.parceling_history[-1]['alignment_score'] if self.parceling_history else None,
            'converged': is_viable if 'is_viable' in locals() else False,
            'parceling_history': self.parceling_history
        }
        
        return df_augmented
    
    def fit_scorecard(self, df_augmented: pd.DataFrame, feature_cols: List[str]) -> None:
        """Use the final T model from parceling process"""
        if self.final_T_model is not None:
            self.model, self.scaler = self.final_T_model
            self.is_fitted = True
        else:
            # Fallback: fit new model
            X = df_augmented[feature_cols]
            y = df_augmented['inferred_default']
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            self.model.fit(X_scaled, y)
            self.is_fitted = True


class DualScoreInferenceMethod(FICOPerformanceInferenceMethod):
    """
    FICO Dual Score Inference Method
    
    Uses combination of KN_SCORE and AR_SCORE as described in FICO whitepaper:
    - KN_SCORE: Performance score developed on known population
    - AR_SCORE: Accept/reject score that embodies the approval policies
    - Uses linear combination to estimate initial pG
    
    This addresses selection bias by explicitly modeling the acceptance process.
    """
    
    def __init__(self, ar_weight: float = 0.3, kn_weight: float = 0.7, 
                 random_state: int = 42):
        super().__init__("FICO Dual Score Inference", random_state)
        self.ar_weight = ar_weight  # Weight for AR_SCORE
        self.kn_weight = kn_weight  # Weight for KN_SCORE
        self.ar_model = None
        self.kn_model = None
        
    def _create_ar_score(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Create AR_SCORE that embodies the accept/reject policies
        This models the approval process itself
        """
        X = df[feature_cols]
        y_approval = df['approved'].astype(int)
        
        scaler_ar = StandardScaler()
        X_scaled = scaler_ar.fit_transform(X)
        
        # Model the approval process
        ar_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        ar_model.fit(X_scaled, y_approval)
        
        self.ar_model = (ar_model, scaler_ar)
        
        # Return approval probabilities as AR_SCORE
        ar_scores = ar_model.predict_proba(X_scaled)[:, 1]  # Probability of approval
        return ar_scores
    
    def _create_kn_score(self, df_known: pd.DataFrame, feature_cols: List[str]) -> Tuple:
        """
        Create KN_SCORE on known population for performance prediction
        """
        X_known = df_known[feature_cols]
        y_known = df_known['observed_default']
        
        scaler_kn = StandardScaler()
        X_known_scaled = scaler_kn.fit_transform(X_known)
        
        kn_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        kn_model.fit(X_known_scaled, y_known)
        
        self.kn_model = (kn_model, scaler_kn)
        
        return kn_model, scaler_kn
    
    def infer_reject_performance(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Apply FICO Dual Score Inference method
        """
        df_augmented = df.copy()
        
        # Step 1: Create AR_SCORE (models approval process)
        ar_scores = self._create_ar_score(df_augmented, feature_cols)
        
        # Step 2: Create KN_SCORE (models performance on known population)
        known_data = df_augmented[df_augmented['approved']].copy()
        if len(known_data) == 0:
            raise ValueError("No known samples available")
            
        kn_model, kn_scaler = self._create_kn_score(known_data, feature_cols)
        
        # Calculate KN_SCORE for all samples
        X_all = df_augmented[feature_cols]
        X_all_scaled = kn_scaler.transform(X_all)
        kn_scores = kn_model.predict_proba(X_all_scaled)[:, 1]  # Probability of default
        
        # Step 3: Combine AR_SCORE and KN_SCORE using linear combination
        # Dual score = ar_weight * AR_SCORE + kn_weight * KN_SCORE
        dual_scores = self.ar_weight * ar_scores + self.kn_weight * kn_scores
        
        # Step 4: Use dual scores to estimate pG for rejects
        rejected_mask = ~df_augmented['approved']
        reject_dual_scores = dual_scores[rejected_mask]
        
        # Convert dual scores to probabilities (calibrate on known data)
        known_dual_scores = dual_scores[df_augmented['approved']]
        known_outcomes = known_data['observed_default']
        
        # Fit calibration model
        calibration_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        calibration_model.fit(known_dual_scores.reshape(-1, 1), known_outcomes)
        
        # Apply to rejects
        reject_probs = calibration_model.predict_proba(reject_dual_scores.reshape(-1, 1))[:, 1]
        
        # Step 5: Assign performance based on calibrated probabilities
        df_augmented['inferred_default'] = df_augmented['observed_default'].copy()
        reject_indices = df_augmented.index[rejected_mask]
        
        # Assign based on probability > 0.5
        bad_rejects = reject_probs > 0.5
        good_rejects = reject_probs <= 0.5
        
        df_augmented.loc[reject_indices[bad_rejects], 'inferred_default'] = 1
        df_augmented.loc[reject_indices[good_rejects], 'inferred_default'] = 0
        
        # Store inference statistics
        self.inference_stats = {
            'ar_weight': self.ar_weight,
            'kn_weight': self.kn_weight,
            'n_rejects_assigned_good': good_rejects.sum(),
            'n_rejects_assigned_bad': bad_rejects.sum(),
            'avg_dual_score_rejects': reject_dual_scores.mean(),
            'avg_reject_prob': reject_probs.mean()
        }
        
        return df_augmented
    
    def fit_scorecard(self, df_augmented: pd.DataFrame, feature_cols: List[str]) -> None:
        """Fit final scorecard model on augmented TTD population"""
        X = df_augmented[feature_cols]
        y = df_augmented['inferred_default']
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.model.fit(X_scaled, y)
        self.is_fitted = True


def create_fico_methods(random_state: int = 42) -> List[FICOPerformanceInferenceMethod]:
    """
    Factory function to create FICO performance inference methods exactly as 
    described in the whitepaper
    """
    methods = [
        ExternalInformationMethod(
            external_score_col='external_predictor',
            assignment_threshold=0.5,
            random_state=random_state
        ),
        ExternalInformationMethod(
            external_score_col='external_predictor', 
            assignment_threshold=0.6,
            random_state=random_state
        ),
        DomainExpertiseParcelingMethod(
            max_iterations=5,
            convergence_threshold=0.05,
            random_state=random_state
        ),
        DomainExpertiseParcelingMethod(
            max_iterations=10,
            convergence_threshold=0.03,
            random_state=random_state
        ),
        DualScoreInferenceMethod(
            ar_weight=0.3,
            kn_weight=0.7,
            random_state=random_state
        ),
        DualScoreInferenceMethod(
            ar_weight=0.5,
            kn_weight=0.5,
            random_state=random_state
        )
    ]
    
    # Update names for clarity
    methods[0].name = "External Info (50% threshold)"
    methods[1].name = "External Info (60% threshold)"  
    methods[2].name = "Domain Parceling (5 iter, 0.05 conv)"
    methods[3].name = "Domain Parceling (10 iter, 0.03 conv)"
    methods[4].name = "Dual Score (AR:30%, KN:70%)"
    methods[5].name = "Dual Score (AR:50%, KN:50%)"
    
    return methods


class FICOMethodValidator:
    """
    Validation framework specifically for FICO methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def validate_method(self, method: FICOPerformanceInferenceMethod, 
                       df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Validate a single FICO method"""
        print(f"\nValidating {method.name}...")
        
        try:
            # Step 1: Infer reject performance
            df_augmented = method.infer_reject_performance(df, feature_cols)
            
            # Step 2: Fit scorecard
            method.fit_scorecard(df_augmented, feature_cols)
            
            # Step 3: Evaluate performance
            results = self._evaluate_performance(method, df, df_augmented, feature_cols)
            results['inference_stats'] = method.inference_stats
            
            print(f"✓ {method.name} validation complete")
            print(f"  - AUC: {results['auc']:.3f}")
            print(f"  - Samples augmented: {results['n_training_samples'] - results['n_original_samples']}")
            
            if 'reject_inference_accuracy' in results:
                print(f"  - Reject inference accuracy: {results['reject_inference_accuracy']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to validate {method.name}: {str(e)}")
            return None
    
    def _evaluate_performance(self, method: FICOPerformanceInferenceMethod,
                            df_original: pd.DataFrame, df_augmented: pd.DataFrame,
                            feature_cols: List[str]) -> Dict:
        """Evaluate method performance"""
        
        # Test on approved samples (known outcomes)
        approved_data = df_original[df_original['approved']].copy()
        X_test = approved_data[feature_cols]
        y_true = approved_data['observed_default']
        
        y_pred_proba = method.predict_proba(X_test)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        results = {
            'auc': auc,
            'n_training_samples': len(df_augmented),
            'n_original_samples': len(df_original),
            'augmentation_ratio': len(df_augmented) / len(df_original)
        }
        
        # Evaluate reject inference quality if ground truth available
        if 'actual_default' in df_original.columns:
            rejected_mask = ~df_original['approved']
            if rejected_mask.sum() > 0:
                true_reject_outcomes = df_original.loc[rejected_mask, 'actual_default']
                inferred_outcomes = df_augmented.loc[rejected_mask, 'inferred_default']
                
                # Calculate inference accuracy
                inference_accuracy = (true_reject_outcomes == inferred_outcomes).mean()
                results['reject_inference_accuracy'] = inference_accuracy
        
        return results
    
    def validate_all_methods(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Validate all FICO methods"""
        methods = create_fico_methods(self.random_state)
        results_list = []
        
        print("="*80)
        print("FICO PERFORMANCE INFERENCE VALIDATION")
        print("Implementing exact methods from FICO whitepaper")
        print("="*80)
        
        for method in methods:
            result = self.validate_method(method, df, feature_cols)
            if result is not None:
                result['method_name'] = method.name
                results_list.append(result)
        
        # Create comparison DataFrame
        if results_list:
            comparison_df = pd.DataFrame(results_list)
            comparison_df = comparison_df.sort_values('auc', ascending=False)
            
            print("\n" + "="*80)
            print("VALIDATION RESULTS SUMMARY")
            print("="*80)
            
            display_cols = ['method_name', 'auc', 'augmentation_ratio']
            if 'reject_inference_accuracy' in comparison_df.columns:
                display_cols.append('reject_inference_accuracy')
                
            print(comparison_df[display_cols].to_string(index=False))
            
            return comparison_df
        else:
            print("No methods validated successfully")
            return pd.DataFrame()


def run_fico_simulation():
    """
    Run simulation with exact FICO methods from whitepaper
    """
    try:
        from sample_simulation import SimulationScenario
        
        print("FICO Performance Inference Methods - Exact Implementation")
        print("Based on 'Building Powerful Scorecards' whitepaper")
        print("="*80)
        
        # Create scenario with external data (needed for External Information method)
        scenario = SimulationScenario(
            name="FICO Method Validation",
            rejection_rate=0.6,  # Higher rejection rate to test methods
            n_samples=15000,
            external_predictor_strength=0.6,  # Enable external predictor
            random_state=42
        )
        
        # Generate data
        df, simulator = scenario.generate_complete_dataset()
        feature_cols = simulator.get_normalized_feature_columns()
        
        print(f"Generated dataset: {len(df)} samples")
        print(f"Rejection rate: {(~df['approved']).mean():.1%}")
        print(f"Approved default rate: {df[df['approved']]['observed_default'].mean():.1%}")
        print(f"External predictor available: {'external_predictor' in df.columns}")
        
        # Validate FICO methods
        validator = FICOMethodValidator(random_state=42)
        results = validator.validate_all_methods(df, feature_cols)
        
        # Save results
        if not results.empty:
            results.to_csv('fico_whitepaper_validation_results.csv', index=False)
            print(f"\nResults saved to fico_whitepaper_validation_results.csv")
        
        return results
        
    except ImportError as e:
        print(f"Could not import simulation framework: {e}")
        return None


if __name__ == "__main__":
    # Run the FICO whitepaper method simulation
    results = run_fico_simulation()
