# Regularization Techniques in Machine Learning

## Question
What are the main regularization techniques in machine learning? Compare L1 and L2 regularization.

## Answer

### Overview
Regularization adds a penalty term to the loss function to prevent overfitting by constraining model complexity. It helps models generalize better to unseen data.

### Main Techniques

#### 1. L2 Regularization (Ridge)

**Mathematical Form:**
$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} w_i^2$$

**Characteristics:**
- Penalizes squared magnitude of coefficients
- Shrinks weights toward zero but rarely exactly zero
- Produces smoother, more stable models
- Differentiable everywhere

**Use Cases:**
- When you want to keep all features but reduce their impact
- When features are correlated
- Neural networks (weight decay)

**Example:**
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha is λ
model.fit(X_train, y_train)
```

#### 2. L1 Regularization (Lasso)

**Mathematical Form:**
$$\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} |w_i|$$

**Characteristics:**
- Penalizes absolute magnitude of coefficients
- Drives some weights to exactly zero → **feature selection**
- Produces sparse models
- Not differentiable at zero

**Use Cases:**
- When you suspect many features are irrelevant
- Feature selection is desired
- Interpretable models needed

**Example:**
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
# Some coefficients will be exactly 0
```

#### 3. Elastic Net

**Mathematical Form:**
$$\text{Loss} = \text{MSE} + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$$

**Characteristics:**
- Combines L1 and L2
- Balance between feature selection and coefficient shrinkage
- More stable than Lasso when features are correlated

### Comparison Table

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|------------|------------|-------------|
| **Sparsity** | Yes (feature selection) | No | Yes (partial) |
| **Coefficient Behavior** | Some → 0 | All → small | Mixed |
| **Correlated Features** | Picks one arbitrarily | Keeps all | Keeps all |
| **Computational Cost** | Higher | Lower | Moderate |
| **Differentiability** | No (at 0) | Yes | Partially |
| **Stability** | Less stable | More stable | Most stable |

### Other Regularization Methods

**4. Dropout (Neural Networks)**
- Randomly drop units during training
- Prevents co-adaptation of neurons

**5. Early Stopping**
- Stop training when validation error increases
- Prevents overfitting without explicit penalty

**6. Data Augmentation**
- Artificially increase training data

**7. Batch Normalization**
- Normalizes layer inputs
- Has regularizing effect

## Tags
#MachineLearning #Regularization #L1 #L2 #FeatureSelection #Overfitting

## Difficulty
Medium

## Related Questions
- What is the bias-variance tradeoff?
- How does cross-validation work?
- Explain gradient descent with regularization
