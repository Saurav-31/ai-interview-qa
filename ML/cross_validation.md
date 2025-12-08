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

### 1. What is the bias-variance tradeoff?

**Initial Answer:** The bias-variance tradeoff describes the fundamental tension between a model's ability to fit training data (low bias) and its ability to generalize to new, unseen data (low variance). High bias leads to underfitting (model too simple), while high variance leads to overfitting (model too complex).

**Analogy:** Like throwing darts - high bias means consistently missing the bullseye in the same direction (systematic error); high variance means hitting all over the board randomly (inconsistent predictions).

**Follow-up: How do you diagnose whether your model has high bias or high variance?**

Look at training vs validation performance:
- **High Bias**: Both training and validation scores are low and similar (underfitting)
- **High Variance**: Training score is high, but validation score is much lower (overfitting)
- **Good Fit**: Both scores are high and close to each other

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()

# High bias: both curves plateau at low values
# High variance: large gap between curves
```

### 2. How to prevent data leakage?

**Initial Answer:** Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates. This happens when training data inadvertently contains information about the target that won't be available at prediction time.

**Prevention strategies:**
- Apply preprocessing (scaling, encoding) only after splitting data
- Never use test data during training or validation
- Be careful with time-based features in temporal data
- Avoid using future information to predict the past

**Follow-up: Can you show an example of data leakage and how to fix it?**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ❌ WRONG - Data Leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
# Test data statistics leaked into training!

# ✅ CORRECT - No Leakage
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training
X_test_scaled = scaler.transform(X_test)  # Only transform test

# With cross-validation pipelines (best practice)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
# Pipeline ensures scaler fits separately in each CV fold
scores = cross_val_score(pipeline, X, y, cv=5)
```

### 3. What's the difference between validation set and test set?

**Initial Answer:** The validation set is used during the training process to tune hyperparameters and make decisions about model selection. The test set is used only once at the very end for final, unbiased evaluation. Both are held out from training, but they serve different purposes.

**Follow-up: Why can't we use the same set for both validation and final testing?**

Because every time you make a decision based on validation performance (choosing features, adjusting hyperparameters, selecting models), you're implicitly "learning" from that validation set. This creates **optimization bias** - your model becomes optimized for that specific validation set.

```python
# Three-way split approach
from sklearn.model_selection import train_test_split

# First split: separate test set (final evaluation only)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)
# Final split: 60% train, 20% validation, 20% test

# Use validation for tuning
for param in param_grid:
    model.set_params(**param)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    # Select best based on val_score

# Use test ONLY ONCE at the end
final_score = best_model.score(X_test, y_test)
```

### 4. When should you use Leave-One-Out Cross-Validation (LOOCV)?

**Initial Answer:** LOOCV uses each single sample as a validation set (n iterations for n samples). It's most appropriate for very small datasets (<100 samples) where every data point is precious and you need maximum training data. However, it's computationally expensive for large datasets and can have high variance.

**Follow-up: What are the specific advantages and disadvantages compared to K-Fold?**

**Advantages:**
- Maximum training data (n-1 samples)
- Deterministic (no randomness)
- Low bias in performance estimate

**Disadvantages:**
- High computational cost (n model fits)
- High variance in performance estimate
- No stratification possible
- Can be unstable for small n

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
import time

loo = LeaveOneOut()

# Small dataset example
X_small = X[:50]  # Only 50 samples
y_small = y[:50]

# LOOCV - good for small datasets
start = time.time()
loo_scores = cross_val_score(model, X_small, y_small, cv=loo)
print(f"LOOCV: {loo_scores.mean():.4f} (±{loo_scores.std():.4f})")
print(f"Time: {time.time() - start:.2f}s, Iterations: {len(loo_scores)}")

# Compare with 5-Fold
start = time.time()
kfold_scores = cross_val_score(model, X_small, y_small, cv=5)
print(f"5-Fold: {kfold_scores.mean():.4f} (±{kfold_scores.std():.4f})")
print(f"Time: {time.time() - start:.2f}s, Iterations: {len(kfold_scores)}")

# For large datasets, LOOCV becomes impractical
# X_large with 10,000 samples would need 10,000 model fits!
```

### 5. How does cross-validation help detect overfitting?

**Initial Answer:** A large gap between training performance and cross-validation performance indicates overfitting. If your model achieves 95% accuracy on training data but only 70% on cross-validation, it means the model has memorized the training data rather than learning generalizable patterns.

**Analogy:** Like a student who only studies past exam questions - they'll ace those specific questions but fail when presented with new questions testing the same concepts.

**Follow-up: How can you quantify overfitting and what threshold indicates a problem?**

```python
from sklearn.model_selection import cross_validate
import numpy as np

# Get both training and validation scores
cv_results = cross_validate(
    model, X, y, cv=5,
    return_train_score=True,
    scoring='accuracy'
)

train_scores = cv_results['train_score']
val_scores = cv_results['test_score']

train_mean = train_scores.mean()
val_mean = val_scores.mean()
gap = train_mean - val_mean

print(f"Training Score: {train_mean:.4f} (±{train_scores.std():.4f})")
print(f"Validation Score: {val_mean:.4f} (±{val_scores.std():.4f})")
print(f"Overfitting Gap: {gap:.4f}")

# Rule of thumb interpretation:
if gap < 0.05:
    print("✓ Good fit - minimal overfitting")
elif gap < 0.10:
    print("⚠ Slight overfitting - monitor")
elif gap < 0.20:
    print("⚠⚠ Moderate overfitting - consider regularization")
else:
    print("❌ Severe overfitting - reduce model complexity")

# Visualize overfitting across complexity
from sklearn.tree import DecisionTreeClassifier

depths = range(1, 20)
train_scores_list, val_scores_list = [], []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    cv_res = cross_validate(tree, X, y, cv=5, return_train_score=True)
    train_scores_list.append(cv_res['train_score'].mean())
    val_scores_list.append(cv_res['test_score'].mean())

plt.plot(depths, train_scores_list, label='Training')
plt.plot(depths, val_scores_list, label='Validation')
plt.xlabel('Tree Depth (Model Complexity)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Overfitting Detection via Cross-Validation')
plt.show()
# Optimal depth is where validation peaks before gap widens
```

### 6. What is stratified sampling and why is it important?

**Initial Answer:** Stratified sampling maintains the same class proportion in each fold as in the original dataset. It's critical for imbalanced datasets to ensure each fold is representative of the true class distribution, preventing folds with too few minority class samples.

**Follow-up: What happens if you don't use stratified sampling on imbalanced data?**

```python
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import make_classification
import numpy as np

# Create imbalanced dataset: 95% class 0, 5% class 1
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.95, 0.05],
                           n_features=20, random_state=42)

print(f"Original class distribution: {np.bincount(y)}")
print(f"Class 1 percentage: {(y==1).sum()/len(y)*100:.1f}%\n")

# ❌ Regular K-Fold - unbalanced folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
print("Regular K-Fold class distribution per fold:")
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    val_class1_pct = (y[val_idx]==1).sum() / len(val_idx) * 100
    print(f"  Fold {fold_idx+1}: {val_class1_pct:.1f}% class 1 "
          f"({(y[val_idx]==1).sum()} samples)")

print()

# ✅ Stratified K-Fold - balanced folds
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Stratified K-Fold class distribution per fold:")
for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
    val_class1_pct = (y[val_idx]==1).sum() / len(val_idx) * 100
    print(f"  Fold {fold_idx+1}: {val_class1_pct:.1f}% class 1 "
          f"({(y[val_idx]==1).sum()} samples)")

# Stratified ensures each fold has exactly 5% class 1
# Regular K-Fold might have 2% in one fold, 8% in another
```

### 7. How many folds should you choose in K-Fold cross-validation?

**Initial Answer:** Common choices are 5 or 10 folds. More folds means less bias (more training data per fold) but more variance and longer computation. 5-fold is a good default; 10-fold for larger datasets. Avoid too few (<3, high bias) or too many (approaching LOOCV, high variance and computation).

**Follow-up: How do you decide between 5-fold and 10-fold for your specific problem?**

**Decision factors:**
- **Dataset size**: <1000 samples → 5-fold; >10,000 samples → 10-fold
- **Computation budget**: Limited resources → 5-fold; ample resources → 10-fold
- **Variance tolerance**: Need stable estimates → 10-fold; can tolerate variance → 5-fold

```python
from sklearn.model_selection import cross_val_score
import time

# Compare different fold numbers
fold_options = [3, 5, 10, 20]
results = {}

for n_folds in fold_options:
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=n_folds, scoring='accuracy')
    duration = time.time() - start_time
    
    results[n_folds] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'time': duration
    }
    
    print(f"{n_folds}-Fold CV:")
    print(f"  Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    print(f"  Time: {duration:.2f}s")
    print(f"  Training size per fold: {len(X) * (n_folds-1) / n_folds:.0f} samples")
    print()

# Typical pattern:
# 3-fold:  Fast but high variance, less training data (66%)
# 5-fold:  Good balance (80% training)
# 10-fold: Lower variance, more training data (90%), slower
# 20-fold: Minimal bias (95% training), high variance, very slow
```

### 8. What is Monte Carlo Cross-Validation?

**Initial Answer:** Monte Carlo Cross-Validation (also called Repeated Random Sub-sampling or Shuffle-Split) repeatedly creates random train-test splits, unlike K-Fold which partitions data into non-overlapping folds. Each iteration uses a different random subset, providing more evaluation samples, but splits can overlap.

**Follow-up: When would you choose Monte Carlo CV over K-Fold?**

**Advantages:**
- More flexible - can choose any train/test ratio
- Can run as many iterations as needed
- Independent splits can reveal stability

**Disadvantages:**
- Some samples may never be tested
- Some samples may be tested multiple times
- Less efficient use of data than K-Fold

```python
from sklearn.model_selection import ShuffleSplit, cross_val_score

# Monte Carlo Cross-Validation
mc_cv = ShuffleSplit(
    n_splits=10,           # Number of random splits
    test_size=0.2,         # 20% for validation
    random_state=42
)

mc_scores = cross_val_score(model, X, y, cv=mc_cv)
print(f"Monte Carlo CV (10 splits):")
print(f"  Mean: {mc_scores.mean():.4f} (±{mc_scores.std():.4f})")

# Compare with K-Fold
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(model, X, y, cv=kfold)
print(f"\n5-Fold CV:")
print(f"  Mean: {kf_scores.mean():.4f} (±{kf_scores.std():.4f})")

# Check sample usage
sample_test_count = np.zeros(len(X))
for train_idx, test_idx in mc_cv.split(X):
    sample_test_count[test_idx] += 1

print(f"\nSample testing frequency in Monte Carlo:")
print(f"  Never tested: {(sample_test_count == 0).sum()} samples")
print(f"  Tested once: {(sample_test_count == 1).sum()} samples")
print(f"  Tested multiple times: {(sample_test_count > 1).sum()} samples")
print(f"  Max tests for one sample: {sample_test_count.max():.0f}")

# Use case: When you need custom split ratios (e.g., 70/30 instead of 80/20)
# or want to assess model stability across many random splits
```

### 9. How do you handle cross-validation with time series data?

**Initial Answer:** For time series data, you must respect temporal order - never train on future data to predict the past. Use TimeSeriesSplit or walk-forward validation where the training window expands or slides forward chronologically.

**Analogy:** Like weather forecasting - you can't use tomorrow's data to predict today's weather.

**Follow-up: What's the difference between expanding window and sliding window approaches?**

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

# Create time series data
n_samples = 100
dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
X = np.arange(n_samples).reshape(-1, 1)
y = np.sin(X.ravel() / 10) + np.random.randn(n_samples) * 0.1

# ✅ TimeSeriesSplit - Expanding Window
tscv = TimeSeriesSplit(n_splits=5)

print("TimeSeriesSplit (Expanding Window):")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}:")
    print(f"  Train: samples {train_idx[0]} to {train_idx[-1]} "
          f"({len(train_idx)} samples)")
    print(f"  Test:  samples {test_idx[0]} to {test_idx[-1]} "
          f"({len(test_idx)} samples)")

# Visualize expanding window
fig, axes = plt.subplots(5, 1, figsize=(10, 8))
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    axes[fold].scatter(train_idx, [fold]*len(train_idx), 
                       c='blue', label='Train', alpha=0.5)
    axes[fold].scatter(test_idx, [fold]*len(test_idx), 
                       c='red', label='Test', alpha=0.8)
    axes[fold].set_ylabel(f'Fold {fold+1}')
    if fold == 0:
        axes[fold].legend()
plt.xlabel('Time Index')
plt.tight_layout()
plt.show()

# Custom Sliding Window (fixed training size)
def sliding_window_cv(X, y, train_size=50, test_size=10, step=5):
    """Sliding window with fixed training window size"""
    scores = []
    n = len(X)
    
    for start in range(0, n - train_size - test_size + 1, step):
        train_end = start + train_size
        test_end = train_end + test_size
        
        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        
        print(f"Window: train [{start}:{train_end}], test [{train_end}:{test_end}], "
              f"score: {score:.4f}")
    
    return np.array(scores)

# Use sliding window for non-stationary data or when recent data is more relevant
scores = sliding_window_cv(X, y, train_size=50, test_size=10, step=10)
print(f"\nSliding Window Mean Score: {scores.mean():.4f}")

# ❌ WRONG - Regular K-Fold on time series
# This violates temporal ordering!
# kfold = KFold(n_splits=5, shuffle=True)  # DON'T DO THIS
```

### 10. What is nested cross-validation and when to use it?

**Initial Answer:** Nested cross-validation uses two levels: an outer loop for model performance evaluation and an inner loop for hyperparameter tuning. Use it when you need both hyperparameter selection and an unbiased estimate of generalization error, as tuning on the same folds used for evaluation creates optimistic bias.

**Follow-up: Can you show the difference between nested CV and regular GridSearchCV?**

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# ❌ BIASED - Using GridSearchCV alone
# This gives optimistic estimate because test folds see hyperparameter optimization
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)
print(f"GridSearchCV best score: {grid_search.best_score_:.4f}")
print(f"Best params: {grid_search.best_params_}")
print("⚠️  This score is biased (overly optimistic)!\n")

# ✅ UNBIASED - Nested Cross-Validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []

print("Nested Cross-Validation:")
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    # Split data for this outer fold
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner loop: hyperparameter tuning on training data only
    grid_search = GridSearchCV(
        SVC(), param_grid, cv=inner_cv, scoring='accuracy'
    )
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Outer loop: evaluate best model on held-out test fold
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_outer, y_test_outer)
    nested_scores.append(test_score)
    
    print(f"  Fold {fold+1}: {test_score:.4f} "
          f"(best params: {grid_search.best_params_})")

print(f"\nNested CV mean score: {np.mean(nested_scores):.4f} "
      f"(±{np.std(nested_scores):.4f})")
print("✓ This is an unbiased estimate of generalization performance")

# Alternative: use cross_val_score with GridSearchCV as estimator
nested_scores_alt = cross_val_score(
    GridSearchCV(SVC(), param_grid, cv=inner_cv),
    X, y, cv=outer_cv, scoring='accuracy'
)
print(f"\nAlternative nested CV: {nested_scores_alt.mean():.4f} "
      f"(±{nested_scores_alt.std():.4f})")

# Typical result: nested CV score is lower than GridSearchCV best_score
# because GridSearchCV score includes the folds used for tuning
```

### 11. Can cross-validation replace a separate test set?

**Initial Answer:** No, not for final model evaluation. Cross-validation is excellent for model selection and hyperparameter tuning during development, but you should always reserve a completely untouched test set for the final unbiased performance assessment before deployment.

**Follow-up: What if I don't have enough data for a separate test set?**

```python
# Scenario 1: Medium dataset (1000+ samples) - Use both
from sklearn.model_selection import train_test_split, GridSearchCV

# Reserve test set first (never touch until the very end)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use CV on development set for model selection
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_dev, y_dev)

print(f"Development CV score: {grid_search.best_score_:.4f}")
print(f"Best parameters: {grid_search.best_params_}")

# Only NOW evaluate on test set (once!)
final_score = grid_search.score(X_test, y_test)
print(f"Final test score: {final_score:.4f}")
print("This is the score you report as model performance")

# Scenario 2: Small dataset (<500 samples) - Nested CV might be necessary
print("\n--- Small Dataset Approach ---")
# When data is too limited for separate test set:
# 1. Use nested CV to get unbiased estimate
# 2. Retrain on ALL data with best hyperparameters for deployment

from sklearn.model_selection import cross_val_score

# Nested CV for unbiased evaluation
outer_scores = cross_val_score(
    GridSearchCV(SVC(), param_grid, cv=3),  # inner CV
    X, y, cv=5, scoring='f1'  # outer CV
)
print(f"Nested CV score (unbiased): {outer_scores.mean():.4f} "
      f"(±{outer_scores.std():.4f})")

# Find best hyperparameters on full dataset
final_grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
final_grid.fit(X, y)

# Deploy this model, report nested CV score as expected performance
print(f"Deploy model with: {final_grid.best_params_}")
print(f"Expected performance: {outer_scores.mean():.4f}")
```

### 12. How does shuffle impact cross-validation?

**Initial Answer:** Shuffling randomizes data order before splitting into folds. It's important for non-IID (non-independent, identically distributed) data, such as when samples are sorted by class or collected in batches. However, shuffling is forbidden for time series data. Use `shuffle=True` with fixed `random_state` for reproducibility.

**Follow-up: What problems can occur if you don't shuffle when you should?**

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# Create dataset where samples are sorted by class
X, y = make_classification(n_samples=1000, n_classes=3, n_informative=10,
                           n_clusters_per_class=1, random_state=42)

# Sort by class (simulating non-random data collection)
sorted_indices = np.argsort(y)
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]

print("Class distribution in sorted data:")
print(f"First 333 samples: {np.bincount(y_sorted[:333])}")
print(f"Middle 334 samples: {np.bincount(y_sorted[333:667])}")
print(f"Last 333 samples: {np.bincount(y_sorted[667:])}\n")

# ❌ No shuffle - PROBLEM!
kfold_no_shuffle = KFold(n_splits=3, shuffle=False)
scores_no_shuffle = cross_val_score(
    LogisticRegression(max_iter=1000), 
    X_sorted, y_sorted, cv=kfold_no_shuffle
)

print("Without shuffle:")
print(f"  Scores: {scores_no_shuffle}")
print(f"  Mean: {scores_no_shuffle.mean():.4f}")
print("  Problem: Each fold has mostly one class!\n")

# ✅ With shuffle - CORRECT
kfold_shuffle = KFold(n_splits=3, shuffle=True, random_state=42)
scores_shuffle = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_sorted, y_sorted, cv=kfold_shuffle
)

print("With shuffle:")
print(f"  Scores: {scores_shuffle}")
print(f"  Mean: {scores_shuffle.mean():.4f}")
print("  ✓ Each fold has balanced classes\n")

# Even better: Use StratifiedKFold
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores_stratified = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_sorted, y_sorted, cv=skfold
)

print("With stratified shuffle:")
print(f"  Scores: {scores_stratified}")
print(f"  Mean: {scores_stratified.mean():.4f}")
print("  ✓ Each fold has exact class proportions")

# Reproducibility with random_state
print("\nReproducibility test:")
scores1 = cross_val_score(model, X, y, 
                          cv=KFold(n_splits=5, shuffle=True, random_state=42))
scores2 = cross_val_score(model, X, y,
                          cv=KFold(n_splits=5, shuffle=True, random_state=42))
print(f"Same random_state: {np.array_equal(scores1, scores2)}")  # True

scores3 = cross_val_score(model, X, y,
                          cv=KFold(n_splits=5, shuffle=True, random_state=99))
print(f"Different random_state: {np.array_equal(scores1, scores3)}")  # False
```

### 13. What is Group K-Fold cross-validation?

**Initial Answer:** Group K-Fold ensures that samples from the same group (patient, user, session) stay together in the same fold. This prevents data leakage when multiple samples from one entity are correlated - if some samples from a patient are in training and others in validation, the model can "cheat" by learning patient-specific patterns.

**Example:** In medical data where one patient has multiple scans, all scans from that patient must be in the same fold.

**Follow-up: How is this different from regular K-Fold, and when exactly do you need it?**

```python
from sklearn.model_selection import GroupKFold, KFold
import numpy as np
import pandas as pd

# Example: Medical dataset with multiple samples per patient
n_patients = 20
n_samples_per_patient = 5
n_total = n_patients * n_samples_per_patient

# Create data with patient groups
patients = np.repeat(range(n_patients), n_samples_per_patient)
X = np.random.randn(n_total, 10)
y = np.random.randint(0, 2, n_total)

# Add correlation within patient groups (realistic scenario)
for patient in range(n_patients):
    patient_mask = patients == patient
    patient_bias = np.random.randn() * 2  # Patient-specific pattern
    X[patient_mask] += patient_bias

print(f"Dataset: {n_total} samples from {n_patients} patients")
print(f"Samples per patient: {n_samples_per_patient}\n")

# ❌ Regular K-Fold - LEAKAGE!
print("Regular K-Fold (WRONG for grouped data):")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    train_patients = set(patients[train_idx])
    val_patients = set(patients[val_idx])
    overlap = train_patients.intersection(val_patients)
    
    print(f"Fold {fold+1}: {len(overlap)} patients appear in BOTH train and validation!")
    if fold == 0 and len(overlap) > 0:
        print(f"  Example: Patient {list(overlap)[0]} has samples in both sets")
        print("  ⚠️  Model can learn patient-specific patterns and 'cheat'!\n")

# ✅ Group K-Fold - NO LEAKAGE
print("\nGroup K-Fold (CORRECT):")
gkfold = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkfold.split(X, y, groups=patients)):
    train_patients = set(patients[train_idx])
    val_patients = set(patients[val_idx])
    overlap = train_patients.intersection(val_patients)
    
    print(f"Fold {fold+1}:")
    print(f"  Train: {len(train_patients)} patients, {len(train_idx)} samples")
    print(f"  Val: {len(val_patients)} patients, {len(val_idx)} samples")
    print(f"  Overlap: {len(overlap)} patients ✓")

# Practical example with real structure
print("\n--- Real-world Example: Customer Behavior ---")

# E-commerce: predicting purchase from session data
df = pd.DataFrame({
    'user_id': [1,1,1, 2,2, 3,3,3,3, 4,4],
    'session_id': [101,102,103, 201,202, 301,302,303,304, 401,402],
    'clicks': [5,3,7, 2,4, 6,8,3,5, 4,2],
    'purchased': [0,0,1, 0,1, 0,0,1,1, 0,0]
})

X = df[['clicks']].values
y = df['purchased'].values
groups = df['user_id'].values

# Wrong: regular CV would split same user across train/test
# Correct: Group K-Fold keeps all sessions from a user together
gkf = GroupKFold(n_splits=3)
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    print(f"Fold {fold+1}:")
    print(f"  Training users: {set(groups[train_idx])}")
    print(f"  Testing users: {set(groups[test_idx])}")
    # No user appears in both sets!
```

### 14. How do you choose the scoring metric for cross-validation?

**Initial Answer:** The scoring metric used in cross-validation should match your business goal and the metric you'll optimize in production. Use accuracy for balanced classification, F1-score or AUROC for imbalanced data, RMSE/MAE for regression. The wrong metric can lead to selecting models that perform poorly on what actually matters.

**Follow-up: Can you show examples of choosing the wrong metric and its consequences?**

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Create highly imbalanced dataset (95% class 0, 5% class 1)
X, y = make_classification(
    n_samples=1000, n_classes=2, weights=[0.95, 0.05],
    n_features=20, random_state=42
)

print(f"Class distribution: {np.bincount(y)} (95% negative, 5% positive)\n")

# Naive baseline: always predict majority class
dummy = DummyClassifier(strategy='most_frequent')
real_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# ❌ WRONG METRIC: Accuracy on imbalanced data
print("Using Accuracy (WRONG for imbalanced data):")
dummy_acc = cross_val_score(dummy, X, y, cv=5, scoring='accuracy').mean()
model_acc = cross_val_score(real_model, X, y, cv=5, scoring='accuracy').mean()

print(f"  Dummy (always predict 0): {dummy_acc:.4f}")
print(f"  Real model: {model_acc:.4f}")
print(f"  Difference: {model_acc - dummy_acc:.4f}")
print("  ⚠️  Looks good but dummy achieves 95% by doing nothing!\n")

# ✅ CORRECT METRICS for imbalanced data
print("Using F1-Score (better for imbalanced):")
dummy_f1 = cross_val_score(dummy, X, y, cv=5, scoring='f1').mean()
model_f1 = cross_val_score(real_model, X, y, cv=5, scoring='f1').mean()

print(f"  Dummy: {dummy_f1:.4f}")
print(f"  Real model: {model_f1:.4f}")
print(f"  Difference: {model_f1 - dummy_f1:.4f} ✓ Clear improvement\n")

print("Using ROC-AUC (best for imbalanced binary):")
dummy_auc = cross_val_score(dummy, X, y, cv=5, scoring='roc_auc').mean()
model_auc = cross_val_score(real_model, X, y, cv=5, scoring='roc_auc').mean()

print(f"  Dummy: {dummy_auc:.4f}")
print(f"  Real model: {model_auc:.4f}\n")

# Multiple metrics comparison
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'roc_auc': 'roc_auc'
}

print("Multi-metric evaluation:")
results = cross_validate(real_model, X, y, cv=5, scoring=scoring)

for metric in scoring.keys():
    score = results[f'test_{metric}'].mean()
    print(f"  {metric:10s}: {score:.4f}")

# Custom scorer for business metrics
def cost_scorer(y_true, y_pred):
    """Custom metric: False Negative costs $100, False Positive costs $10"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = fn * 100 + fp * 10
    return -cost  # Negative because sklearn maximizes scores

custom_scorer = make_scorer(cost_scorer)

print("\nUsing custom business metric (minimize cost):")
cost_scores = cross_val_score(real_model, X, y, cv=5, scoring=custom_scorer)
print(f"  Average cost: ${-cost_scores.mean():.2f} per 1000 predictions")

# Regression metrics example
print("\n--- Regression Metrics ---")
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)

reg_scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

for metric in reg_scoring:
    scores = cross_val_score(Ridge(), X_reg, y_reg, cv=5, scoring=metric)
    print(f"  {metric:30s}: {scores.mean():.4f}")

print("\nKey takeaway: Choose metrics that align with your actual business objective!")
```

### 15. What are the computational costs of cross-validation?

**Initial Answer:** K-fold cross-validation requires training K models. When combined with hyperparameter search, the cost multiplies (K folds × number of parameter combinations). This creates a trade-off between thorough evaluation and computation time. For large datasets, consider fewer folds or randomized search instead of exhaustive grid search.

**Analogy:** Like tasting every dish at a buffet versus sampling a few representative ones - more thorough evaluation but takes much longer.

**Follow-up: How can you reduce computational cost while maintaining reliable evaluation?**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from scipy.stats import uniform, randint
import time

# Setup
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)

# Scenario 1: Full Grid Search with 5-fold CV
print("=" * 60)
print("SCENARIO 1: Full Grid Search (Most Expensive)")
print("=" * 60)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

total_fits = 5 * 4 * 2 * 5  # params × CV folds
print(f"Parameter combinations: {5 * 4 * 2}")
print(f"CV folds: 5")
print(f"Total model fits: {total_fits}\n")

start = time.time()
grid_full = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_full.fit(X, y)
time_full = time.time() - start

print(f"Time: {time_full:.2f}s")
print(f"Best score: {grid_full.best_score_:.4f}")
print(f"Best params: {grid_full.best_params_}\n")

# Scenario 2: Randomized Search (Faster)
print("=" * 60)
print("SCENARIO 2: Randomized Search (Much Faster)")
print("=" * 60)

param_distributions = {
    'C': uniform(0.01, 100),
    'gamma': uniform(0.001, 1),
    'kernel': ['rbf', 'poly']
}

n_iter = 20  # Sample only 20 combinations instead of 40
total_fits = n_iter * 5

print(f"Random parameter samples: {n_iter}")
print(f"CV folds: 5")
print(f"Total model fits: {total_fits}\n")

start = time.time()
random_search = RandomizedSearchCV(
    SVC(), param_distributions, n_iter=n_iter, cv=5, 
    random_state=42, n_jobs=-1, verbose=1
)
random_search.fit(X, y)
time_random = time.time() - start

print(f"Time: {time_random:.2f}s ({time_random/time_full*100:.1f}% of grid search)")
print(f"Best score: {random_search.best_score_:.4f}")
print(f"Best params: {random_search.best_params_}\n")

# Scenario 3: Reduced CV Folds
print("=" * 60)
print("SCENARIO 3: Fewer CV Folds (3 instead of 5)")
print("=" * 60)

start = time.time()
grid_3fold = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_3fold.fit(X, y)
time_3fold = time.time() - start

print(f"Time: {time_3fold:.2f}s ({time_3fold/time_full*100:.1f}% of 5-fold)")
print(f"Best score: {grid_3fold.best_score_:.4f}\n")

# Scenario 4: Subset of Data for Fast Iteration
print("=" * 60)
print("SCENARIO 4: Use Data Subset for Initial Exploration")
print("=" * 60)

X_subset = X[:1000]  # Use only 20% of data
y_subset = y[:1000]

start = time.time()
grid_subset = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_subset.fit(X_subset, y_subset)
time_subset = time.time() - start

print(f"Time: {time_subset:.2f}s ({time_subset/time_full*100:.1f}% of full data)")
print(f"Best params on subset: {grid_subset.best_params_}")
print("Then validate with full data and refined parameter range\n")

# Cost comparison summary
print("=" * 60)
print("COST-EFFECTIVENESS SUMMARY")
print("=" * 60)

strategies = {
    'Full Grid Search (5-fold)': time_full,
    'Randomized Search (20 iter, 5-fold)': time_random,
    'Grid Search (3-fold)': time_3fold,
    'Grid Search on Subset': time_subset
}

for strategy, duration in strategies.items():
    speedup = time_full / duration
    print(f"{strategy:40s}: {duration:6.2f}s (×{speedup:.1f} speedup)")

# Practical recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("""
1. Start with Randomized Search for initial exploration
2. Use 3-fold CV during development, 5-fold for final evaluation
3. For very large datasets (>100k samples):
   - Use subset for hyperparameter search
   - Use single validation split instead of CV
   - Consider Halving GridSearch (successive halving)

4. Parallel processing: Use n_jobs=-1 to utilize all CPU cores

5. For nested CV: computational cost is outer_folds × inner_folds × n_params
   Example: 5×3×20 = 300 model fits!
""")

# Bonus: Halving Grid Search (sklearn 0.24+)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

print("BONUS: Halving Grid Search")
print("-" * 60)

start = time.time()
halving_search = HalvingGridSearchCV(
    SVC(), param_grid, cv=5, factor=3, 
    resource='n_samples', random_state=42
)
halving_search.fit(X, y)
time_halving = time.time() - start

print(f"Time: {time_halving:.2f}s ({time_halving/time_full*100:.1f}% of full grid)")
print(f"Best score: {halving_search.best_score_:.4f}")
print("\nHalving Search progressively eliminates poor hyperparameters,")
print("training only promising candidates on full data.")
```


