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
