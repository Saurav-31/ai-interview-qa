# Cross-Validation Techniques

## Question
What is cross-validation and why is it important? Explain different cross-validation strategies.

## Answer

### Overview
Cross-validation is a resampling technique used to evaluate machine learning models on limited data.

### Why Cross-Validation?

**Problems with Simple Train-Test Split:**
- Performance depends on specific split
- High variance in performance estimates

**Benefits:**
- More robust performance estimates
- Better use of limited data
- Helps detect overfitting

## Main Techniques

### 1. K-Fold Cross-Validation

Split data into k equal folds, train on k-1, validate on 1.

```python
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"Mean: {scores.mean():.4f}")
```

### 2. Stratified K-Fold

Maintains class distribution in each fold - **essential for imbalanced datasets**.

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold)
```

### 3. Time Series Cross-Validation

Respects temporal order in sequential data.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

## Comparison Table

| Method | Iterations | Use Case |
|--------|-----------|----------|
| **K-Fold** | k (5-10) | General purpose |
| **Stratified K-Fold** | k | Imbalanced data |
| **Time Series** | k | Sequential data |

## Nested Cross-Validation

For unbiased evaluation with hyperparameter tuning:
- **Outer loop**: Model evaluation
- **Inner loop**: Hyperparameter tuning

## Tags
#MachineLearning #CrossValidation #ModelEvaluation #Overfitting

## Difficulty
Medium

## Related Questions
- What is the bias-variance tradeoff?
- How to prevent data leakage?
