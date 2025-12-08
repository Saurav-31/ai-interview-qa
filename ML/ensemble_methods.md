# Ensemble Methods in Machine Learning

## Question
Explain ensemble methods in machine learning. What are the differences between bagging, boosting, and stacking?

## Answer

### Overview
Ensemble methods combine multiple models to produce better predictions than any single model. They work on the principle that diverse models can compensate for each other's weaknesses.

### Core Principle
**"Wisdom of the Crowd"** → Multiple weak learners combine to form a strong learner

## 1. Bagging (Bootstrap Aggregating)

### Concept
- Train multiple models **independently** on different random subsets of data
- Aggregate predictions by averaging (regression) or voting (classification)
- **Reduces variance** without increasing bias

### Key Characteristics
- **Parallel training**: Models trained simultaneously
- **Bootstrap sampling**: Random sampling with replacement
- **High variance models** benefit most (e.g., deep decision trees)

### Random Forest (Bagging + Feature Randomness)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_features='sqrt',   # Features per split
    max_depth=None,        # Fully grown trees
    bootstrap=True         # Bootstrap sampling
)
rf.fit(X_train, y_train)
```

### Pros and Cons
✅ Reduces overfitting  
✅ Handles high-dimensional data well  
✅ Parallelizable  
❌ May underfit if base learners too simple  
❌ Less interpretable  

## 2. Boosting

### Concept
- Train models **sequentially**
- Each model focuses on correcting previous model's errors
- **Reduces bias** and can reduce variance

### Key Characteristics
- **Sequential training**: Each model depends on previous ones
- **Weighted instances**: Misclassified samples get higher weight
- **Weak learners**: Typically shallow trees (stumps)

#### XGBoost (Extreme Gradient Boosting)
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0      # L2 regularization
)
xgb_model.fit(X_train, y_train)
```

### Pros and Cons
✅ High predictive accuracy  
✅ Handles complex patterns  
✅ Feature importance  
❌ Prone to overfitting  
❌ Slow training (sequential)  
❌ Sensitive to noisy data  

## 3. Stacking (Stacked Generalization)

### Concept
- Train multiple diverse base models
- Use a **meta-model** to learn how to best combine them
- **Learns optimal combination** rather than simple averaging

### Implementation
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5  # Cross-validation to avoid overfitting
)
stacking.fit(X_train, y_train)
```

### Pros and Cons
✅ Can achieve best performance  
✅ Flexible (any model combination)  
❌ Complex to implement and tune  
❌ Risk of overfitting  
❌ Computationally expensive  

## Comparison Table

| Aspect | Bagging | Boosting | Stacking |
|--------|---------|----------|----------|
| **Training** | Parallel | Sequential | Multi-level |
| **Base Learners** | Strong (low bias) | Weak (high bias) | Diverse |
| **Reduces** | Variance | Bias + Variance | Both |
| **Weighting** | Equal | Adaptive | Learned |
| **Speed** | Fast | Slower | Slowest |
| **Overfitting Risk** | Low | Medium-High | Medium |
| **Example** | Random Forest | XGBoost | Custom ensembles |

## When to Use What?

**Use Bagging when:**
- Model has high variance (overfitting)
- You want parallelization

**Use Boosting when:**
- Model has high bias (underfitting)
- You need maximum accuracy

**Use Stacking when:**
- You have computational resources
- You need state-of-the-art performance

## Tags
#MachineLearning #Ensemble #Bagging #Boosting #RandomForest #XGBoost #Stacking

## Difficulty
Medium-Hard

## Related Questions
- How does Random Forest work?
- Explain XGBoost algorithm
- What is the bias-variance tradeoff?

---

## Additional Interview Questions & Answers

### Q1: Why does bagging reduce variance but not bias?

**Answer:**

**Analogy:** Bagging is like asking 10 doctors for independent diagnoses and averaging their opinions. If all doctors use the same flawed diagnostic method (bias), averaging won't fix it. But if each doctor makes different random mistakes (variance), averaging cancels them out.

**Mathematical Intuition:**

For n independent models with variance σ²:
$$\text{Var}(\text{Average}) = \frac{\sigma^2}{n}$$

**Bias stays constant:**
$$\text{Bias}(\text{Average}) = \text{Bias}(\text{Individual})$$

**Why?**
- **Bias** = systematic error, same across all bootstrap samples
- **Variance** = random fluctuations, different across bootstrap samples
- Averaging reduces random errors, not systematic ones

**Example:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# High variance base model (deep tree)
base = DecisionTreeClassifier(max_depth=20)  # Overfits

# Bagging reduces variance
bagging = BaggingClassifier(base, n_estimators=100)

# Result:
# Base model: Train=99%, Val=75% (high variance)
# Bagging: Train=95%, Val=85% (reduced variance, same bias)
```

**Key Insight:** To reduce bias, use boosting or a more complex base model.

### Q2: Explain the difference between Random Forest and Extra Trees.

**Answer:**

**Analogy:** 
- **Random Forest:** Careful gardener who finds the best spot to prune each branch
- **Extra Trees:** Fast gardener who prunes randomly but grows more trees to compensate

**Key Differences:**

| Aspect | Random Forest | Extra Trees |
|--------|---------------|-------------|
| **Split Selection** | Best split among random features | Random split among random features |
| **Bootstrap** | Yes (sample with replacement) | No (use full dataset) |
| **Training Speed** | Slower | Faster |
| **Variance** | Lower | Higher (but compensated by more trees) |
| **Overfitting** | Less prone | More prone (needs more trees) |

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Random Forest: Finds best split
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
rf.fit(X_train, y_train)

# Extra Trees: Random split
et = ExtraTreesClassifier(n_estimators=100, max_features='sqrt')
et.fit(X_train, y_train)

# Extra Trees trains ~3x faster
# But may need more estimators for same performance
```

**When to Use:**
- **Random Forest**: Better accuracy, moderate training time
- **Extra Trees**: Faster training, large datasets, can tolerate more trees

**Practical Tip:**
```python
# Extra Trees often needs more trees
et = ExtraTreesClassifier(n_estimators=200)  # vs 100 for RF
```

### Q3: How does AdaBoost assign weights to training samples?

**Answer:**

**Analogy:** AdaBoost is like a teacher who gives harder problems to students who got previous questions wrong. Students who struggle get more attention (higher weights).

**Weight Update Mechanism:**

**Initial:** All samples have equal weight
$$w_i^{(1)} = \frac{1}{n}$$

**After each model t:**

1. **Train** model on weighted data
2. **Compute** error: $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i^{(t)}$
3. **Compute** model weight: $\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
4. **Update** sample weights:

$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(\alpha_t \cdot \mathbb{1}_{h_t(x_i) \neq y_i})$$

**Misclassified:** weight increases
**Correctly classified:** weight decreases

**Example:**
```python
import numpy as np

# Sample weights after iteration 1
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # 5 samples
predictions = [1, 1, 0, 1, 1]
true_labels = [1, 0, 0, 1, 1]

# Error rate
errors = [0, 1, 0, 0, 0]  # Sample 2 misclassified
error_rate = sum(w * e for w, e in zip(weights, errors))  # 0.1

# Model weight
alpha = 0.5 * np.log((1 - error_rate) / error_rate)  # 1.099

# Update weights
new_weights = []
for i in range(5):
    if errors[i]:
        new_weights.append(weights[i] * np.exp(alpha))  # Increase
    else:
        new_weights.append(weights[i] * np.exp(-alpha))  # Decrease

# Normalize
new_weights = np.array(new_weights)
new_weights /= new_weights.sum()

print(new_weights)  # [0.07, 0.43, 0.07, 0.07, 0.07]
# Sample 2 now has 6x higher weight!
```

**Visualization:**
```
Iteration 1: ○ ○ ○ ○ ○  (all equal)
             ✓ ✗ ✓ ✓ ✓
             
Iteration 2: ○ ●●● ○ ○ ○  (misclassified sample emphasized)
             ✓ ✓ ✗ ✓ ✓
             
Iteration 3: ○ ● ●●● ○ ○  (focus shifts to new mistakes)
```

### Q4: What is gradient boosting and how does it differ from AdaBoost?

**Answer:**

**Analogy:**
- **AdaBoost:** Teacher who highlights hard problems in red (reweights samples)
- **Gradient Boosting:** Teacher who creates custom problems targeting each student's specific weaknesses (fits residuals)

**Key Differences:**

| Aspect | AdaBoost | Gradient Boosting |
|--------|----------|-------------------|
| **Target** | Reweighted samples | Residuals/gradients |
| **Loss Function** | Exponential (fixed) | Any differentiable |
| **Flexibility** | Classification focus | Regression & classification |
| **Learning** | Adjusts weights | Fits residuals |
| **Complexity** | Simpler | More flexible |

**Gradient Boosting Process:**

```python
# Pseudocode
F_0(x) = initial_prediction  # e.g., mean of y

for m in range(1, M+1):
    # 1. Compute residuals (negative gradient)
    residuals = y - F_{m-1}(x)
    
    # 2. Fit tree to residuals
    h_m = fit_tree(X, residuals)
    
    # 3. Update model
    F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)

return F_M(x)
```

**Implementation:**
```python
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

# AdaBoost
ada = AdaBoostRegressor(n_estimators=100)
ada.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
gb.fit(X_train, y_train)

# GB typically performs better for regression
```

**Visual:**
```
AdaBoost:
Sample weights: [0.1, 0.3, 0.1, 0.2, ...]
Next model focuses on high-weight samples

Gradient Boosting:
Residuals: [2.1, -1.3, 0.5, -0.8, ...]
Next model predicts these residuals
```

### Q5: Explain the bias-variance tradeoff in boosting.

**Answer:**

**Analogy:** Boosting is like a student who keeps taking practice tests and learning from mistakes. Early rounds reduce bias (learn patterns), but too many rounds increase variance (memorize noise).

**Evolution During Boosting:**

```
Error
  |
  |  Bias↓       Variance↑
  |   \\          //
  |    \\        //
  |     \\      //
  |      \\    //
  |       \\  //
  |________\\//________
          Optimal
       # Estimators
```

**Stage-by-Stage:**

**Early Stage (Few Trees):**
- High bias, low variance
- Underfitting
- Model too simple

**Optimal Stage:**
- Balanced bias-variance
- Best generalization
- **Stop here!**

**Late Stage (Many Trees):**
- Low bias, high variance
- Overfitting training data
- Memorizing noise

**Code Example:**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

scores_train = []
scores_val = []

for n in [10, 50, 100, 200, 500, 1000]:
    gb = GradientBoostingClassifier(n_estimators=n, max_depth=3)
    
    gb.fit(X_train, y_train)
    scores_train.append(gb.score(X_train, y_train))
    scores_val.append(gb.score(X_val, y_val))

# Results:
# n=10:   Train=0.75, Val=0.73 (high bias)
# n=100:  Train=0.92, Val=0.88 (optimal)
# n=1000: Train=0.99, Val=0.85 (high variance, overfitting)
```

**Prevention:**
```python
# Use early stopping
gb = GradientBoostingClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.1,
    validation_fraction=0.1,
    n_iter_no_change=10,  # Stop if no improvement for 10 rounds
    tol=0.001
)
```

### Q6: What is stacking and when should you use it?

**Answer:**

**Analogy:** Stacking is like having specialist doctors (heart, lungs, brain) examine a patient, then a general practitioner (meta-learner) combines their diagnoses for the final decision.

**Architecture:**

```
Level 0 (Base Models):
Model 1 (RF)  →  pred_1
Model 2 (GB)  →  pred_2     ──┐
Model 3 (SVM) →  pred_3       │
Model 4 (NN)  →  pred_4     ──┘
                              ↓
Level 1 (Meta-Model):   [pred_1, pred_2, pred_3, pred_4]
Logistic Regression  →  Final Prediction
```

**Implementation:**
```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Level 0: Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svm', SVC(probability=True))
]

# Level 1: Meta-model
meta_model = LogisticRegression()

# Stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Cross-validation to avoid overfitting
)

stacking.fit(X_train, y_train)
```

**When to Use:**

✅ **Good for:**
- Kaggle competitions (squeeze last 1-2% accuracy)
- Different model types (e.g., tree + linear + neural)
- Enough data for meta-learning
- Computational resources available

❌ **Avoid when:**
- Small datasets
- Need interpretability
- Limited compute resources
- Models are too similar (correlated predictions)

**Comparison:**

| Method | Complexity | Performance | Interpretability |
|--------|-----------|-------------|------------------|
| Single Model | Low | Good | High |
| Bagging | Medium | Better | Medium |
| Boosting | Medium | Better | Medium |
| Stacking | High | Best | Low |

### Q7: How do you choose the number of trees in an ensemble?

**Answer:**

**Analogy:** Like adding musicians to an orchestra. Too few (10) sounds incomplete, enough (50-100) sounds full, too many (1000) doesn't improve much but costs more.

**Method 1: Validation Curve**

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

n_estimators_range = [10, 50, 100, 200, 500, 1000]
train_scores = []
val_scores = []

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n)
    rf.fit(X_train, y_train)
    
    train_scores.append(rf.score(X_train, y_train))
    val_scores.append(rf.score(X_val, y_val))

# Plot: validation score plateaus around 100-200
```

**Method 2: Learning Curve (for Boosting)**

```python
# XGBoost with early stopping
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

params = {'max_depth': 3, 'learning_rate': 0.1}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    early_stopping_rounds=10,
    evals=[(dval, 'validation')],
    verbose_eval=False
)

print(f"Best iteration: {model.best_iteration}")  # e.g., 247
```

**Guidelines:**

| Ensemble Type | Typical Range | Rule of Thumb |
|---------------|---------------|---------------|
| **Random Forest** | 100-500 | More is better (plateaus) |
| **Extra Trees** | 200-1000 | Needs more than RF |
| **AdaBoost** | 50-200 | Can overfit with too many |
| **Gradient Boosting** | 100-1000 | Use early stopping |
| **XGBoost/LightGBM** | 100-5000 | Always use early stopping |

**Practical:**
```python
# For Random Forest: Go big (won't overfit)
rf = RandomForestClassifier(n_estimators=500)

# For Boosting: Use early stopping
gb = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=20
)
```

**Cost-Benefit:**
```
Performance
    |        _____
    |      _/
    |    _/
    |  _/
    |_/____________
     10  100  500
     Trees

Cost = O(n_trees)
Benefit plateaus!
```

### Q8: Explain out-of-bag (OOB) error estimation.

**Answer:**

**Analogy:** In bagging, each tree only sees ~63% of data (bootstrap). The remaining ~37% act as a free validation set. It's like having a built-in test set without using extra data!

**Mathematical:**

For each bootstrap sample:
- Probability of selecting a sample: $1 - (1 - \frac{1}{n})^n \approx 0.632$
- Probability of NOT selecting: $\approx 0.368$

**OOB Evaluation:**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True  # Enable OOB estimation
)

rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.3f}")
# No need for separate validation set!

# OOB score ≈ cross-validation score
cv_score = cross_val_score(rf, X_train, y_train, cv=5).mean()
print(f"CV Score: {cv_score:.3f}")
# Usually within 1-2% of each other
```

**How It Works:**

```
Training Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Tree 1 bootstrap: [1, 2, 2, 4, 5, 7, 8, 9, 9]
   OOB samples: [3, 6, 10]  ← Use Tree 1 to predict these

Tree 2 bootstrap: [1, 3, 3, 4, 5, 6, 7, 8, 10, 10]
   OOB samples: [2, 9]  ← Use Tree 2 to predict these

... for all trees ...

Final: Each sample predicted by ~37% of trees
       Aggregate predictions → OOB score
```

**Advantages:**

✅ Free validation estimate
✅ No need to set aside validation data
✅ Computationally efficient
✅ Good approximation of test error

**Limitations:**

❌ Only for bagging-based methods
❌ Slightly less reliable than true validation
❌ Not suitable for hyperparameter tuning (use CV)

### Q9: What is the difference between bagging and pasting?

**Answer:**

**Analogy:**
- **Bagging:** Polling with replacement - same person can be surveyed multiple times
- **Pasting:** Polling without replacement - each person surveyed once max

**Comparison:**

| Aspect | Bagging | Pasting |
|--------|---------|---------|
| **Sampling** | With replacement | Without replacement |
| **Sample Size** | n samples | m < n samples |
| **Diversity** | Higher (duplicates) | Lower (unique) |
| **Bias** | Same | Same |
| **Variance** | Lower | Slightly higher |
| **Popular** | Yes (Random Forest) | Rare |

**Implementation:**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging (default)
bagging = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100,
    bootstrap=True,  # With replacement
    max_samples=1.0   # 100% of data
)

# Pasting
pasting = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100,
    bootstrap=False,  # Without replacement
    max_samples=0.7    # 70% of data
)
```

**Why Bagging is More Popular:**

1. **Higher Diversity:** Duplicates create more variation between trees
2. **OOB Estimation:** Can use out-of-bag samples for free validation
3. **Better Variance Reduction:** More effective at reducing overfitting

**When to Use Pasting:**
- Very large datasets (sampling without replacement is faster)
- Want to ensure each sample used at most once per tree
- Computational constraints

### Q10: How does learning rate affect gradient boosting?

**Answer:**

**Analogy:** Learning rate is like step size when climbing down a mountain. Large steps (high LR) get down fast but might overshoot the valley. Small steps (low LR) are precise but take forever.

**Mathematical:**

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $\eta$ is the learning rate (typically 0.01-0.3)

**Effect on Training:**

| Learning Rate | Trees Needed | Training Time | Risk |
|---------------|-------------|---------------|------|
| **High (0.3)** | Fewer (~100) | Fast | Overfitting |
| **Medium (0.1)** | Moderate (~500) | Medium | Balanced |
| **Low (0.01)** | Many (~5000) | Slow | Underfitting |

**Code Example:**
```python
from sklearn.ensemble import GradientBoostingClassifier

# High learning rate: Fast but risky
gb_fast = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.3,
    max_depth=3
)

# Low learning rate: Slow but stable
gb_slow = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3
)

# Optimal: Medium LR with early stopping
gb_optimal = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.1,
    n_iter_no_change=20
)
```

**Visualization:**
```
High LR (0.5):           Low LR (0.01):
Error                    Error
  |                        |
  |\                       | \
  | \                      |  \
  |  \                     |   \___
  |   \/\  Oscillates     |       \___
  |______                 |___________\__
    Iterations              Iterations
    (Unstable)             (Stable, slow)
```

**Best Practice (Shrinkage + Trees):**

**Trade-off formula:**
```
Total capacity = learning_rate × n_estimators

Same capacity:
- LR=0.1, Trees=1000
- LR=0.01, Trees=10000
- LR=1.0, Trees=100

Lower LR + More trees = Better generalization
```

**Practical Recommendation:**
```python
# Start here
lr = 0.1
n_estimators = 1000

# If overfitting: Decrease LR, increase trees
lr = 0.05
n_estimators = 2000

# Always use early stopping!
```

### Q11: What is the role of max_features in Random Forest?

**Answer:**

**Analogy:** max_features is like limiting each expert to considering only a subset of evidence. If everyone considers everything (max_features='all'), they'll make similar decisions. Random subsets create diversity.

**Purpose: Increase Tree Diversity**

**Options:**

| max_features | Features per Split | Use Case |
|--------------|-------------------|----------|
| **'auto' or 'sqrt'** | √p | Classification (default) |
| **'log2'** | log₂(p) | Very high dimensions |
| **None or p** | p (all features) | Low diversity, high bias |
| **int (e.g., 5)** | Fixed number | Manual control |
| **float (e.g., 0.3)** | 30% of features | Proportional |

**Impact:**

```python
from sklearn.ensemble import RandomForestClassifier

# Too many features: Low diversity
rf_all = RandomForestClassifier(max_features=None)
# Trees will be very similar → Less variance reduction

# Too few features: High diversity but weak trees
rf_few = RandomForestClassifier(max_features=2)
# Trees differ but individually weak → May miss patterns

# Optimal (sqrt)
rf_optimal = RandomForestClassifier(max_features='sqrt')
# Good balance: diverse yet strong trees
```

**Example (p=100 features):**

```
max_features=None (100): All trees consider all features
  → High correlation between trees
  → Less effective variance reduction

max_features='sqrt' (10): Each split considers 10 random features
  → Low correlation between trees
  → Maximum variance reduction

max_features=1: Each split considers 1 random feature
  → Trees too weak individually
  → Poor performance
```

**Mathematical:**

**Correlation between trees:**
- max_features=p → correlation = high
- max_features=√p → correlation = medium
- max_features=1 → correlation = low (but trees weak)

**Optimal is usually √p** for classification, p/3 for regression

**Practical:**
```python
# Classification
rf = RandomForestClassifier(max_features='sqrt')  # Good default

# Regression
rf = RandomForestRegressor(max_features='sqrt')  # Or try p/3

# High-dimensional (p > 10,000)
rf = RandomForestClassifier(max_features='log2')
```

### Q12: How do you handle class imbalance in ensemble methods?

**Answer:**

**Analogy:** Class imbalance is like learning from a history book with 99 pages about wars and 1 page about peace. The model becomes an expert on wars but ignores peace (minority class).

**Strategies:**

**1. Class Weights:**
```python
from sklearn.ensemble import RandomForestClassifier

# Automatically balance
rf = RandomForestClassifier(class_weight='balanced')

# Or manually
rf = RandomForestClassifier(class_weight={0: 1, 1: 10})
# Penalize misclassifying class 1 (minority) 10x more
```

**2. Balanced Bagging:**
```python
from imblearn.ensemble import BalancedBaggingClassifier

# Each bag has balanced classes
bbc = BalancedBaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    sampling_strategy='auto',  # Balance each bag
    n_estimators=100
)
```

**3. Easy Ensemble:**
```python
from imblearn.ensemble import EasyEnsembleClassifier

# Multiple balanced subsets of majority class
eec = EasyEnsembleClassifier(
    n_estimators=10,
    sampling_strategy='auto'
)
# Creates 10 balanced datasets, trains model on each
```

**4. SMOTE + Ensemble:**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Oversample minority class, then ensemble
pipeline = Pipeline([
    ('smote', SMOTE()),
    ('classifier', RandomForestClassifier())
])
```

**5. Adjust Decision Threshold:**
```python
# Instead of default 0.5 threshold
probas = rf.predict_proba(X_test)

# Lower threshold for minority class
threshold = 0.3
predictions = (probas[:, 1] > threshold).astype(int)
```

**Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| **Class Weights** | Simple, fast | May not fix severe imbalance |
| **Balanced Bagging** | Effective, reduces data waste | More complex |
| **Easy Ensemble** | Very effective | Training time increases |
| **SMOTE** | Creates synthetic samples | May create noise |
| **Threshold Tuning** | Post-hoc, flexible | Requires probability calibration |

**Example:**
```python
# Imbalanced data: 95% class 0, 5% class 1
# Without handling
rf_basic = RandomForestClassifier()
rf_basic.fit(X_train, y_train)
# Predicts class 0 for everything → 95% accuracy but useless!

# With class weighting
rf_weighted = RandomForestClassifier(class_weight='balanced')
rf_weighted.fit(X_train, y_train)
# Balances precision/recall for both classes

# Metrics to use
from sklearn.metrics import f1_score, roc_auc_score
print(f"F1: {f1_score(y_test, predictions)}")
print(f"ROC-AUC: {roc_auc_score(y_test, probas[:, 1])}")
# Don't rely on accuracy alone!
```

### Q13: Explain feature importance in Random Forests.

**Answer:**

**Analogy:** Feature importance is like credit distribution in a group project. Features that consistently help make correct decisions get more credit.

**Two Methods:**

**1. Impurity-Based (Gini Importance):**

Average decrease in impurity when splitting on that feature

$$\text{Importance}(f) = \sum_{t: \text{feature } f} \frac{n_t}{n} \cdot \Delta \text{impurity}_t$$

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Get importances
importances = rf.feature_importances_
feature_names = X_train.columns

# Sort
indices = np.argsort(importances)[::-1]
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.3f}")

# Visualize
import matplotlib.pyplot as plt
plt.barh(feature_names[indices[:10]], importances[indices[:10]])
```

**2. Permutation Importance:**

Measure accuracy drop when feature values are randomly shuffled

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42
)

sorted_idx = perm_importance.importances_mean.argsort()[::-1]
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: "
          f"{perm_importance.importances_mean[i]:.3f} "
          f"± {perm_importance.importances_std[i]:.3f}")
```

**Comparison:**

| Method | Pros | Cons |
|--------|------|------|
| **Gini** | Fast, built-in | Biased toward high-cardinality |
| **Permutation** | Unbiased, model-agnostic | Slower, needs test set |

**Pitfalls:**

**❌ Issue 1: Correlated Features**
```python
# If feature A and B are correlated
# Importance splits between them
# Neither looks important individually
```

**❌ Issue 2: High-Cardinality Bias**
```python
# Feature with 100 categories
# Gets more chances to split
# Appears more important (Gini)
```

**✅ Best Practice:**
```python
# Use both methods
gini_imp = rf.feature_importances_
perm_imp = permutation_importance(rf, X_test, y_test)

# Features important in both → truly important
# Important in Gini only → possibly biased
# Important in permutation only → investigate
```

### Q14: What is calibration and why does it matter for ensembles?

**Answer:**

**Analogy:** Calibration is like weather forecast accuracy. If a forecaster says "80% chance of rain," it should rain 80% of the time they say that. Uncalibrated models might say 80% but actually rain only 50%.

**The Problem:**

Many ensembles (especially RF and GB) output uncalibrated probabilities:

```python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

proba = rf.predict_proba(X_test)[:, 1]
# Model says 0.9 probability
# But only correct 70% of the time!
```

**Calibration:**

**Perfect Calibration:**
```
Among samples with predicted prob 0.7,
70% should be positive class
```

**Checking Calibration:**
```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_test,
    probas,
    n_bins=10
)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], '--', color='gray')  # Perfect calibration
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
```

**Calibration Methods:**

**1. Platt Scaling (Logistic):**
```python
from sklearn.calibration import CalibratedClassifierCV

# Fit on training, calibrate on validation
rf = RandomForestClassifier()
calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train)

# Now probabilities are calibrated
```

**2. Isotonic Regression:**
```python
calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=5)
# More flexible but needs more data
```

**When Calibration Matters:**

✅ **Critical:**
- Medical diagnosis (need actual probabilities)
- Risk assessment (insurance, credit)
- Decision-making with costs
- Probability-based thresholds

❌ **Less Important:**
- Just need rankings (recommendation systems)
- Classification with fixed threshold
- Feature importance analysis

**Comparison:**

| Model | Calibration | When to Calibrate |
|-------|-------------|-------------------|
| **Logistic Regression** | Good | Rarely needed |
| **SVM** | Poor | Always |
| **Random Forest** | Medium | Often helpful |
| **Gradient Boosting** | Medium | Often helpful |
| **Naive Bayes** | Good | Rarely needed |
| **Neural Network** | Variable | Check first |

### Q15: How do you debug poor ensemble performance?

**Answer:**

**Systematic Debugging Framework:**

**Step 1: Check Base Model Performance**

```python
from sklearn.tree import DecisionTreeClassifier

# Train single tree
single_tree = DecisionTreeClassifier(max_depth=10)
single_tree.fit(X_train, y_train)

print(f"Single tree train: {single_tree.score(X_train, y_train)}")
print(f"Single tree test: {single_tree.score(X_test, y_test)}")

# If single tree is bad, ensemble won't help much
```

**Analogy:** If individual musicians can't play, an orchestra won't sound good.

**Step 2: Check for Data Issues**

```python
# Class imbalance?
print(np.bincount(y_train))

# Missing values?
print(X_train.isnull().sum())

# Feature scaling (for distance-based models)?
print(X_train.describe())

# Data leakage?
# Check if any features perfectly predict target
for col in X_train.columns:
    corr = np.corrcoef(X_train[col], y_train)[0, 1]
    if abs(corr) > 0.95:
        print(f"Suspicious: {col} has {corr:.3f} correlation!")
```

**Step 3: Diagnose Bias vs Variance**

```python
train_score = rf.score(X_train, y_train)
val_score = rf.score(X_val, y_val)

if train_score < 0.7 and val_score < 0.7:
    print("HIGH BIAS - Model too simple")
    print("→ Increase max_depth, decrease min_samples_split")
    print("→ Or use boosting instead of bagging")
    
elif train_score > 0.95 and val_score < 0.8:
    print("HIGH VARIANCE - Overfitting")
    print("→ Increase n_estimators")
    print("→ Decrease max_depth, increase min_samples_leaf")
    print("→ Add regularization")
```

**Step 4: Hyperparameter Search**

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=20,
    cv=5,
    scoring='f1',
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

**Step 5: Check Feature Quality**

```python
# Remove low-importance features
importances = rf.feature_importances_
low_importance = importances < 0.01

X_train_filtered = X_train.loc[:, ~low_importance]
rf_filtered = RandomForestClassifier()
rf_filtered.fit(X_train_filtered, y_train)

# Compare performance
print(f"All features: {rf.score(X_val, y_val)}")
print(f"Filtered: {rf_filtered.score(X_val.loc[:, ~low_importance], y_val)}")
```

**Step 6: Try Different Ensemble Types**

```python
from sklearn.ensemble import (RandomForestClassifier, 
                               GradientBoostingClassifier,
                               AdaBoostClassifier)

models = {
    'RF': RandomForestClassifier(),
    'GB': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Common Issues & Fixes:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Train & val both low | High bias | Deeper trees, boosting |
| Train high, val low | High variance | More trees, regularization |
| Slow training | Too many trees/depth | Reduce n_estimators, max_depth |
| Bad on imbalanced data | Class imbalance | class_weight='balanced' |
| Unstable predictions | Too few trees | Increase n_estimators |
| Poor probabilities | Uncalibrated | Use CalibratedClassifierCV |

**Checklist:**
```python
✓ Base model performance decent?
✓ Data quality checked?
✓ Bias vs variance diagnosed?
✓ Hyperparameters tuned?
✓ Feature engineering done?
✓ Correct evaluation metric?
✓ Cross-validation used?
✓ Tried different ensemble types?
```
