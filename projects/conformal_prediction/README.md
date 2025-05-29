# Conformal Prediction for Logistic Regression

This implementation provides a simple conformal prediction method for logistic regression that generates prediction sets with guaranteed marginal coverage.

## What is Conformal Prediction?

Conformal prediction is a framework for uncertainty quantification that provides prediction sets with guaranteed coverage probability. Unlike traditional confidence intervals that rely on distributional assumptions, conformal prediction is distribution-free and provides finite-sample guarantees.

For a classification problem with miscoverage level α, conformal prediction guarantees that the prediction set will contain the true label with probability at least 1-α.

## Features

- **Distribution-free guarantees**: No assumptions about data distribution
- **Finite-sample validity**: Coverage guarantees hold for any sample size
- **Flexible miscoverage levels**: Easy to adjust confidence levels
- **Efficient implementation**: Uses probability-based nonconformity scores

## How It Works

1. **Data Split**: The training data is split into proper training and calibration sets
2. **Model Training**: A logistic regression model is trained on the training set
3. **Calibration**: Nonconformity scores are computed on the calibration set using `score = 1 - P(true_class | x)`
4. **Quantile Computation**: The (1-α)-quantile of calibration scores determines the threshold
5. **Prediction Sets**: For new data, include all classes where `P(class | x) ≥ 1 - quantile`

## Usage

### Basic Usage

```python
from conformal_prediction_for_LR import ConformalLogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit conformal predictor
cp_model = ConformalLogisticRegression(alpha=0.1)  # 90% coverage
cp_model.fit(X_train, y_train)

# Get prediction sets
prediction_sets = cp_model.predict_sets(X_test)
print(f"First prediction set: {prediction_sets[0]}")

# Evaluate coverage
coverage, avg_set_size = cp_model.evaluate_coverage(X_test, y_test)
print(f"Coverage: {coverage:.1%}, Average set size: {avg_set_size:.2f}")
```

### Advanced Usage

```python
# Different miscoverage levels
for alpha in [0.05, 0.1, 0.2]:
    cp_model = ConformalLogisticRegression(alpha=alpha)
    cp_model.fit(X_train, y_train, calibration_size=0.3)
    coverage, size = cp_model.evaluate_coverage(X_test, y_test)
    print(f"α={alpha}: Coverage={coverage:.1%}, Size={size:.2f}")

# Get both point predictions and sets
point_preds = cp_model.predict(X_test)
pred_sets = cp_model.predict_sets(X_test)
probabilities = cp_model.predict_proba(X_test)
```

## Parameters

### ConformalLogisticRegression

- `alpha` (float, default=0.1): Miscoverage level. Target coverage is 1-α
- `random_state` (int, default=42): Random state for reproducibility

### Methods

- `fit(X, y, calibration_size=0.3)`: Fit the conformal predictor
- `predict_sets(X)`: Return prediction sets for new data
- `predict(X)`: Standard point predictions
- `predict_proba(X)`: Prediction probabilities
- `evaluate_coverage(X, y)`: Evaluate coverage and average set size

## Key Properties

1. **Marginal Coverage**: P(Y ∈ C(X)) ≥ 1-α for any distribution
2. **Exchangeability**: Requires that (X₁,Y₁),...,(Xₙ,Yₙ) are exchangeable
3. **Efficiency**: Smaller prediction sets while maintaining coverage
4. **Adaptivity**: Set sizes adapt to local uncertainty

## Running the Demo

```bash
python conformal_prediction_for_LR.py
```

This will run a comprehensive demonstration showing:
- Coverage analysis for different α values
- Example predictions with interpretation
- Visualization of coverage vs set size trade-offs

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Example Output

The demo shows how conformal prediction provides reliable uncertainty quantification:

```
Testing with alpha = 0.1 (target coverage = 90.0%)
  Target Coverage: 90.0%
  Actual Coverage: 91.7%  # Achieves or exceeds target
  Average Set Size: 1.62   # Efficient prediction sets
  Point Accuracy: 72.7%

Sample 1:
  True label: 1
  Prediction set: [1, 2]    # Uncertain between classes 1 and 2
  Contains true label: ✓    # Correctly includes true label
  Class probabilities: {0: 0.047, 1: 0.766, 2: 0.188}
```

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*
- Angelopoulos, A. N., & Bates, S. (2021). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification* 