import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


class ConformalLogisticRegression:
    """
    Conformal Prediction wrapper for Logistic Regression.
    
    This class implements inductive conformal prediction for classification,
    providing prediction sets with guaranteed marginal coverage.
    """
    
    def __init__(self, alpha: float = 0.1, random_state: int = 42):
        """
        Initialize the conformal predictor.
        
        Args:
            alpha: Miscoverage level (1-alpha is the target coverage)
            random_state: Random state for reproducibility
        """
        self.alpha = alpha
        self.random_state = random_state
        self.model = LogisticRegression(random_state=random_state)
        self.calibration_scores = None
        self.classes_ = None
        self.quantile = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, calibration_size: float = 0.3):
        """
        Fit the conformal predictor.
        
        Args:
            X: Feature matrix
            y: Target labels
            calibration_size: Fraction of data to use for calibration
        """
        # Split data into training and calibration sets
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=calibration_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Fit the underlying model
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Compute nonconformity scores on calibration set
        self.calibration_scores = self._compute_nonconformity_scores(X_cal, y_cal)
        
        # Compute the quantile for prediction sets
        n_cal = len(self.calibration_scores)
        self.quantile = np.quantile(
            self.calibration_scores, 
            np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal,
            method='higher'
        )
        
        return self
    
    def _compute_nonconformity_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity scores using the softmax (probability) approach.
        
        The nonconformity score for a sample is 1 - p(true_class | x)
        """
        probabilities = self.model.predict_proba(X)
        scores = []
        
        for i, true_label in enumerate(y):
            # Find the index of the true label
            true_class_idx = np.where(self.classes_ == true_label)[0][0]
            # Nonconformity score is 1 - probability of true class
            score = 1 - probabilities[i, true_class_idx]
            scores.append(score)
            
        return np.array(scores)
    
    def predict_sets(self, X: np.ndarray) -> List[List]:
        """
        Predict conformal prediction sets.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            List of prediction sets (one set per sample)
        """
        if self.quantile is None:
            raise ValueError("Model must be fitted before making predictions")
            
        probabilities = self.model.predict_proba(X)
        prediction_sets = []
        
        for prob_row in probabilities:
            # Include all classes where 1 - p(class) <= quantile
            # Equivalently, p(class) >= 1 - quantile
            prediction_set = []
            for i, prob in enumerate(prob_row):
                if prob >= 1 - self.quantile:
                    prediction_set.append(self.classes_[i])
            
            # Ensure we always have at least one prediction
            if len(prediction_set) == 0:
                # Include the class with highest probability
                best_class_idx = np.argmax(prob_row)
                prediction_set.append(self.classes_[best_class_idx])
                
            prediction_sets.append(prediction_set)
            
        return prediction_sets
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Standard point predictions (most likely class).
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities.
        """
        return self.model.predict_proba(X)
    
    def evaluate_coverage(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the coverage and average set size.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Tuple of (coverage, average_set_size)
        """
        prediction_sets = self.predict_sets(X)
        
        # Calculate coverage
        covered = 0
        total_set_size = 0
        
        for i, (pred_set, true_label) in enumerate(zip(prediction_sets, y)):
            if true_label in pred_set:
                covered += 1
            total_set_size += len(pred_set)
        
        coverage = covered / len(y)
        avg_set_size = total_set_size / len(y)
        
        return coverage, avg_set_size


def demo_conformal_prediction():
    """
    Demonstrate conformal prediction on a synthetic dataset.
    """
    print("=== Conformal Prediction for Logistic Regression Demo ===\n")
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=8,
        n_redundant=2, 
        n_classes=3, 
        random_state=42
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test different alpha values
    alphas = [0.05, 0.1, 0.2]
    results = []
    
    for alpha in alphas:
        print(f"Testing with alpha = {alpha} (target coverage = {1-alpha:.1%})")
        
        # Fit conformal predictor
        cp_model = ConformalLogisticRegression(alpha=alpha, random_state=42)
        cp_model.fit(X_train, y_train)
        
        # Evaluate on test set
        coverage, avg_set_size = cp_model.evaluate_coverage(X_test, y_test)
        
        # Standard accuracy for comparison
        point_predictions = cp_model.predict(X_test)
        accuracy = accuracy_score(y_test, point_predictions)
        
        results.append({
            'alpha': alpha,
            'target_coverage': 1 - alpha,
            'actual_coverage': coverage,
            'avg_set_size': avg_set_size,
            'point_accuracy': accuracy
        })
        
        print(f"  Target Coverage: {1-alpha:.1%}")
        print(f"  Actual Coverage: {coverage:.1%}")
        print(f"  Average Set Size: {avg_set_size:.2f}")
        print(f"  Point Accuracy: {accuracy:.1%}")
        print()
    
    # Show some example predictions
    print("=== Example Predictions ===")
    cp_model = ConformalLogisticRegression(alpha=0.1, random_state=42)
    cp_model.fit(X_train, y_train)
    
    # Get predictions for first 10 test samples
    example_sets = cp_model.predict_sets(X_test[:10])
    example_probs = cp_model.predict_proba(X_test[:10])
    
    for i in range(10):
        true_label = y_test[i]
        pred_set = example_sets[i]
        probs = example_probs[i]
        
        print(f"Sample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Prediction set: {pred_set}")
        print(f"  Contains true label: {'✓' if true_label in pred_set else '✗'}")
        print(f"  Class probabilities: {dict(zip(cp_model.classes_, probs))}")
        print()
    
    # Create a summary table
    df_results = pd.DataFrame(results)
    print("=== Summary Results ===")
    print(df_results.round(3))
    
    return df_results


if __name__ == "__main__":
    # Run the demonstration
    results = demo_conformal_prediction()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage plot
    ax1.plot(results['alpha'], results['actual_coverage'], 'bo-', label='Actual Coverage')
    ax1.plot(results['alpha'], results['target_coverage'], 'r--', label='Target Coverage')
    ax1.set_xlabel('Alpha (Miscoverage Level)')
    ax1.set_ylabel('Coverage')
    ax1.set_title('Conformal Prediction Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set size plot
    ax2.plot(results['alpha'], results['avg_set_size'], 'go-')
    ax2.set_xlabel('Alpha (Miscoverage Level)')
    ax2.set_ylabel('Average Set Size')
    ax2.set_title('Average Prediction Set Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
