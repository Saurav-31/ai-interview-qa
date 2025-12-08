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

---

## Additional Interview Questions & Answers

### Q1: Why does L1 regularization lead to sparse solutions while L2 doesn't?

**Answer:**

**Analogy:** Imagine you're packing a suitcase with a weight limit. L1 says "eliminate entire items" (sparse), while L2 says "reduce weight of all items proportionally" (dense).

**Geometric Intuition:**

```
L1 (Diamond shape):        L2 (Circle):
    |                          .---.
   /|\                       .'     '.
  / | \                     /         \
 /  |  \                   |     o     |
/__\|/__\                   \         /
    |                        '._____.'
    
Solution hits corner        Solution hits edge
(some weights = 0)          (no weights = 0)
```

**Mathematical:**

**L1:** $|\theta_i - \epsilon|$ for small $\epsilon$ ≈ $|\theta_i|$
- Derivative is constant: sign($\theta_i$)
- Can push weights exactly to zero

**L2:** $\theta_i^2$
- Derivative is proportional: $2\theta_i$
- Approaches zero asymptotically, never reaches it

**Code Example:**
```python
from sklearn.linear_model import Lasso, Ridge
import numpy as np

X = np.random.randn(100, 20)
y = X[:, :3] @ [1, 2, 3] + np.random.randn(100) * 0.1  # Only 3 features matter

# L1: Sparse weights
lasso = Lasso(alpha=0.1).fit(X, y)
print(np.sum(lasso.coef_ == 0))  # ~17 weights are exactly zero

# L2: Dense weights  
ridge = Ridge(alpha=0.1).fit(X, y)
print(np.sum(ridge.coef_ == 0))  # 0 weights are exactly zero
```

### Q2: When should you use L1 vs L2 vs Elastic Net?

**Answer:**

**Analogy:** 
- **L1**: Minimalist - "Keep only essentials"
- **L2**: Moderate - "Keep everything but downsized"
- **Elastic Net**: Best of both - "Keep essentials full size, others downsized"

**Decision Table:**

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Many irrelevant features** | L1 (Lasso) | Feature selection |
| **All features relevant** | L2 (Ridge) | Keeps all information |
| **Correlated features** | Elastic Net | L1 picks 1, L2 keeps groups |
| **n < p (more features than samples)** | L2 or Elastic Net | L1 max selects n features |
| **Interpretability needed** | L1 | Sparse solution easier to explain |
| **Multicollinearity** | Elastic Net or L2 | Handles correlation |
| **Unknown feature relevance** | Elastic Net | Safe default |

**Real Example:**
```python
# Genomics: 20,000 genes, 100 samples
# Most genes irrelevant → Use L1

# Image pixels: All pixels matter
# High correlation between neighbors → Use Elastic Net

# Economics: 10 well-understood predictors
# All likely relevant → Use L2
```

### Q3: How does dropout relate to regularization in neural networks?

**Answer:**

**Analogy:** Dropout is like practicing a team sport where random players are benched each game. The team learns not to depend on any single player (prevents co-adaptation).

**Mechanism:**
```python
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            # Randomly zero out p% of activations
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)  # Scale to maintain expected value
        return x  # No dropout during inference
```

**Why it's Regularization:**

1. **Ensemble Effect:** Training 2^n different sub-networks
2. **Feature Co-adaptation:** Prevents neurons from relying on each other
3. **Implicit L2:** Can be shown equivalent to adaptive L2 on weights

**Comparison:**

| Technique | Sparsity | Training Speed | Best For |
|-----------|----------|----------------|----------|
| **Dropout** | Random (temporary) | Slower | Deep networks |
| **L1** | Fixed (permanent) | Fast | Linear models |
| **L2** | None | Fast | All models |

**Practical Tips:**
```python
# Common dropout rates
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% for fully connected
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),  # Lower for smaller layers
    nn.Linear(128, 10)
)
```

### Q4: Explain early stopping as a form of regularization.

**Answer:**

**Analogy:** Like stopping at a yellow light. Going through (training longer) risks running a red (overfitting). Stopping early keeps you safe (good generalization).

**How it Works:**

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(1000):
    # Train
    train_loss = train_epoch(model, train_loader)
    
    # Validate
    val_loss = validate(model, val_loader)
    
    # Check if improving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Stop if no improvement
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        load_checkpoint(model)  # Restore best
        break
```

**Why it's Regularization:**

**Training Progress:**
```
Loss
  |
  | Train \_____
  |        \____
  |  Val   \  /  ← Stop here!
  |         \/
  |_____________
     Epochs
```

**Mathematical View:**
- Each epoch increases model complexity (learns more patterns)
- Early stopping = limiting effective capacity
- Equivalent to L2 regularization in some settings

**Advantages:**
- ✅ Computationally cheap
- ✅ No hyperparameter tuning (just patience)
- ✅ Works with any model

**Disadvantages:**
- ❌ Requires validation set
- ❌ May stop too early with noisy data

### Q5: What is the effect of regularization on the bias-variance tradeoff?

**Answer:**

**Analogy:** Regularization is like adding bumpers to a bowling alley. It constrains your throws (adds bias) but makes scores more consistent (reduces variance).

**Mathematical:**

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**Impact of Increasing Regularization:**
- Bias ↑: Model constrained, can't fit all patterns
- Variance ↓: Less sensitive to training data
- Sweet spot: Bias² + Variance minimized

**Visualization:**
```
Error
  |
  |        Total Error
  |       /--\
  |      /    \___  ← Optimal λ
  |     /         \___
  | Bias²             \___
  |    ________            \___
  |   /        Variance        \___
  |__________________________________
    λ=0              λ increasing
```

**Empirical:**
```python
from sklearn.model_selection import validation_curve

alphas = np.logspace(-4, 4, 20)
train_scores, val_scores = validation_curve(
    Ridge(), X, y,
    param_name='alpha',
    param_range=alphas,
    cv=5
)

# Plot to find optimal alpha
plt.plot(alphas, train_scores.mean(axis=1), label='Train')
plt.plot(alphas, val_scores.mean(axis=1), label='Validation')
plt.xscale('log')
```

### Q6: How do you choose the regularization strength (λ or α)?

**Answer:**

**Analogy:** Like finding the right amount of salt in cooking - too little is bland (underfitting), too much is inedible (overfitting), need to taste-test (cross-validation).

**Methods:**

**1. Cross-Validation (Most Common):**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': np.logspace(-4, 4, 20)}
grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
```

**2. Validation Curve:**
```python
# Plot performance vs alpha
for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"alpha={alpha}: {scores.mean():.3f}")
```

**3. Analytical (Special Cases):**
- **Bayesian view:** α = noise_variance / prior_variance
- **AIC/BIC:** Information criteria

**4. Rule of Thumb:**
```python
# Start with these ranges
L1 (Lasso): alpha in [0.001, 0.01, 0.1, 1.0, 10]
L2 (Ridge): alpha in [0.01, 0.1, 1.0, 10, 100]
Elastic Net: alpha in [0.001, 0.1, 1.0], l1_ratio in [0.1, 0.5, 0.9]
```

**Practical Strategy:**
1. Start with wide range (log scale)
2. Find approximate region
3. Narrow search in that region
4. Use 5-fold or 10-fold CV

### Q7: What is the difference between L1/L2 regularization and feature selection?

**Answer:**

**Analogy:**
- **Feature Selection**: Hiring process - hire or reject (binary decision)
- **L1 Regularization**: Performance review - some fired (zero), others get varying salaries
- **L2 Regularization**: Everyone gets salary cut proportionally

**Comparison:**

| Aspect | Feature Selection | L1 | L2 |
|--------|------------------|----|----|
| **Mechanism** | Remove features | Shrink coefficients | Shrink all |
| **Sparsity** | Explicit | Automatic | No |
| **Gradient** | Discrete | Continuous | Continuous |
| **Computation** | NP-hard | Convex | Convex |
| **Interpretability** | High | High | Low |

**Feature Selection Methods:**
```python
# 1. Filter: Select by correlation
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)

# 2. Wrapper: Forward/Backward selection
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
rfe.fit(X, y)

# 3. Embedded: Use L1
lasso = Lasso(alpha=0.1).fit(X, y)
selected_features = np.where(lasso.coef_ != 0)[0]
```

**When to Use What:**
- **Many features, few relevant**: Feature selection or L1
- **All features relevant**: L2
- **Need interpretability**: Feature selection
- **Need automation**: L1

### Q8: Explain batch normalization as a form of regularization.

**Answer:**

**Analogy:** Batch normalization is like giving everyone in a class the same study materials (normalize inputs), reducing dependency on who you sit next to (reduces co-adaptation).

**Mechanism:**
```python
class BatchNorm(nn.Module):
    def forward(self, x):
        if self.training:
            # Normalize using batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            x_norm = (x - mean) / sqrt(var + eps)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / sqrt(self.running_var + eps)
        
        # Scale and shift (learnable)
        return self.gamma * x_norm + self.beta
```

**Regularization Effect:**

1. **Noise Injection:** Batch statistics add noise (like dropout)
   - Different batches → different normalizations
   - Forces robust representations

2. **Smoother Loss Landscape:** Makes optimization easier
   - Reduces internal covariate shift
   - Allows higher learning rates

**Empirical Evidence:**
```python
# With BatchNorm: Can use less/no other regularization
model_with_bn = nn.Sequential(
    nn.Linear(100, 50),
    nn.BatchNorm1d(50),  # BN provides regularization
    nn.ReLU(),
    nn.Linear(50, 10)
)
# Often works without dropout!

# Without BatchNorm: Need explicit regularization
model_without_bn = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(0.5),  # Need dropout
    nn.Linear(50, 10)
)
```

**Controversy:**
- Some argue it's not "true" regularization
- Effect depends on batch size
- Recent work shows it may not be necessary with modern techniques

### Q9: What is data augmentation and how does it act as regularization?

**Answer:**

**Analogy:** Like a teacher giving students different versions of the same problem. Students learn concepts rather than memorizing specific examples.

**Mechanism:**
- Generate synthetic training samples
- Increase effective dataset size
- Force invariance to transformations

**Image Augmentation:**
```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomRotation(15),  # ±15 degrees
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomErasing(p=0.5)
])

# Training with augmentation
for images, labels in train_loader:
    # Each epoch sees different versions
    augmented = augmentation(images)
    outputs = model(augmented)
    loss = criterion(outputs, labels)
```

**Text Augmentation:**
```python
# Synonym replacement
"The cat sat on the mat"
→ "The feline sat on the rug"

# Back-translation
English → French → English
"I love machine learning"
→ "J'adore l'apprentissage automatique"
→ "I adore automatic learning"

# Random insertion/deletion
"The [quick] brown fox jumps"
```

**Why it's Regularization:**
1. **Prevents memorization**: Can't memorize augmented versions
2. **Invariance**: Learns features robust to transformations
3. **Effective capacity**: Dataset appears larger
4. **Smooth decision boundaries**: Similar inputs → similar outputs

**Effectiveness:**
```python
# Without augmentation
model.fit(X_train, y_train)
# Test accuracy: 85%

# With augmentation
model.fit(augmented(X_train), y_train)
# Test accuracy: 92% (7% improvement!)
```

### Q10: How does regularization affect gradient descent convergence?

**Answer:**

**Analogy:** Regular gradient descent is like skiing down a mountain in the dark - might get stuck in valleys. Regularization smooths the mountain, making descent easier.

**Impact on Optimization:**

**Without Regularization:**
```
Loss surface (rough):
     /\  /\
    /  \/  \
   /        \
  
- Many local minima
- Sharp valleys
- Slow convergence
```

**With Regularization:**
```
Loss surface (smooth):
      __
    _/  \_
  _/      \_
  
- Fewer local minima
- Smoother gradients
- Faster convergence
```

**Mathematical:**

**Gradient without regularization:**
$$\nabla_\theta L = \nabla_\theta \text{DataLoss}$$

**Gradient with L2:**
$$\nabla_\theta L = \nabla_\theta \text{DataLoss} + \lambda \theta$$
- Extra term pulls weights toward zero
- Acts as damping/friction

**Convergence Properties:**

| Aspect | No Regularization | With Regularization |
|--------|------------------|-------------------|
| **Convergence Rate** | Slower | Faster |
| **Stability** | Less stable | More stable |
| **Final Loss** | Lower | Higher (by design) |
| **Generalization** | Worse | Better |

**Code:**
```python
# Without regularization
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# May oscillate

# With weight decay (L2)
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    weight_decay=0.0001  # λ
)
# Smoother convergence
```

**Practical Observation:**
- Can often use higher learning rates with regularization
- Reduces sensitivity to initialization
- More robust to hyperparameter choices

### Q11: What is the connection between regularization and Bayesian inference?

**Answer:**

**Analogy:** Regularization is like having prior beliefs before seeing data. L2 says "I believe weights are probably small" (Gaussian prior). L1 says "I believe most weights are zero" (Laplace prior).

**Mathematical Connection:**

**Maximum A Posteriori (MAP) = Regularized MLE**

**Bayes Rule:**
$$P(\theta | D) \propto P(D | \theta) \cdot P(\theta)$$
- $P(\theta | D)$: Posterior (what we want)
- $P(D | \theta)$: Likelihood (fits data)
- $P(\theta)$: Prior (regularization)

**Taking Logs:**
$$\log P(\theta | D) = \log P(D | \theta) + \log P(\theta) + const$$

**L2 Regularization:**
```
Prior: θ ~ N(0, 1/λ)
→ log P(θ) = -λ/2 ||θ||²

MAP Estimate:
argmax log P(D|θ) + log P(θ)
= argmax log P(D|θ) - λ/2 ||θ||²
= Ridge Regression!
```

**L1 Regularization:**
```
Prior: θ ~ Laplace(0, 1/λ)
→ log P(θ) = -λ ||θ||₁

MAP Estimate:
= argmax log P(D|θ) - λ ||θ||₁
= Lasso!
```

**Interpretation:**
```python
# Ridge = Gaussian prior on weights
ridge = Ridge(alpha=1.0)  # Strong belief weights small

# Lasso = Laplace prior (sparse belief)
lasso = Lasso(alpha=1.0)  # Strong belief most weights = 0
```

**Full Bayesian (not just MAP):**
```python
# Instead of point estimate, get distribution
import pymc3 as pm

with pm.Model():
    # Prior
    theta = pm.Normal('theta', mu=0, sigma=1)
    
    # Likelihood
    y_pred = X @ theta
    y_obs = pm.Normal('y_obs', mu=y_pred, sigma=noise, observed=y)
    
    # Sample posterior
    trace = pm.sample(1000)
    
# Now have uncertainty estimates!
```

### Q12: When can regularization hurt model performance?

**Answer:**

**Analogy:** Like tying your shoelaces too tight. Some constraint is good (prevents tripping), but too much restricts movement (can't run properly).

**Scenarios Where Regularization Hurts:**

**1. Large, Clean Dataset:**
```python
# 1M samples, low noise
# Model won't overfit anyway
model = Ridge(alpha=0.001)  # Very weak regularization
# Too much regularization → underfit
```

**2. All Features Relevant:**
```python
# Physics simulation: all variables matter
# L1 will incorrectly remove features
lasso = Lasso(alpha=1.0)  # Bad choice!
ridge = Ridge(alpha=0.1)  # Better
```

**3. Wrong Type of Regularization:**
```python
# Sparse ground truth
y = X[:, [0, 5, 10]] @ [1, 2, 3] + noise

# L2: Doesn't exploit sparsity
ridge = Ridge(alpha=1.0)  # Suboptimal

# L1: Exploits sparsity
lasso = Lasso(alpha=0.1)  # Better!
```

**4. Pre-regularized Data:**
```python
# Data already includes domain constraints
# Additional regularization may conflict
# E.g., PCA-transformed data
```

**5. Too Strong Regularization:**
```python
# Underfitting example
model = Ridge(alpha=1000)  # Too strong
train_score = 0.5  # Bad even on training data!
val_score = 0.48

# Solution: Reduce alpha
model = Ridge(alpha=1.0)
train_score = 0.85
val_score = 0.83  # Better!
```

**Detection:**
```python
# Both train and val scores are low → overregularized
if train_score < 0.7 and val_score < 0.7:
    print("Reduce regularization strength")
```

### Q13: Explain the relationship between regularization and feature scaling.

**Answer:**

**Analogy:** Imagine measuring a recipe in different units - cups vs tablespoons. Without standardization, regularization penalizes tablespoons more (they're larger numbers).

**Problem:**

**Unscaled features:**
```python
X = pd.DataFrame({
    'age': [25, 30, 35],        # Scale: 20-80
    'income': [50000, 60000, 70000],  # Scale: 20k-200k
    'score': [0.7, 0.8, 0.9]    # Scale: 0-1
})

# L2 regularization: ||θ||²
# Penalizes θ_income much more than θ_age
# Even if both equally important!
```

**Solution:**
```python
from sklearn.preprocessing import StandardScaler

# Standardize: mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now all features on same scale
# Regularization treats them equally
```

**Mathematical:**

**L2 penalty without scaling:**
$$\lambda (\theta_1^2 + \theta_2^2 + \theta_3^2)$$

If $X_1$ has scale 100, $X_2$ has scale 1:
- $\theta_1$ will be small (~0.01)
- $\theta_2$ will be large (~1)
- Penalty unfairly targets $\theta_2$!

**After scaling:**
- All features have comparable $\theta$ magnitudes
- Fair penalization

**Best Practice:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Always scale before regularization
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
```

**Exception:**
- Tree-based models (Random Forest, XGBoost)
- Don't need scaling (use splits, not coefficients)

### Q14: What is mixup and how does it regularize models?

**Answer:**

**Analogy:** Mixup is like creating hybrid species in biology. Combine a cat and a dog to create something in between - forces the model to learn smooth transitions.

**Mechanism:**

**Standard Training:**
```python
# Sample → Label
image1 → cat (100%)
image2 → dog (100%)
```

**Mixup:**
```python
# Linear interpolation
λ = np.random.beta(α, α)  # α typically 0.2-0.4

mixed_image = λ * image1 + (1-λ) * image2
mixed_label = λ * cat + (1-λ) * dog

# Example: λ=0.7
# 70% cat + 30% dog → [0.7, 0.3]
```

**Implementation:**
```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    
    # Random permutation
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    # Mix inputs
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # Mix labels
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

# Training
for x, y in train_loader:
    x_mixed, y_a, y_b, lam = mixup_data(x, y)
    
    pred = model(x_mixed)
    loss = lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)
    loss.backward()
```

**Why it's Regularization:**

1. **Smooth Decision Boundaries:**
   - Forces model to handle interpolations
   - Reduces overconfidence

2. **More Training Samples:**
   - Infinite combinations possible
   - Effective data augmentation

3. **Better Calibration:**
   - Softens one-hot labels
   - Probabilistic outputs more meaningful

**Variants:**
```python
# CutMix: Mix spatial regions
mixed_image = image1.copy()
mixed_image[x1:x2, y1:y2] = image2[x1:x2, y1:y2]

# ManifoldMixup: Mix hidden representations
hidden1 = model.encoder(image1)
hidden2 = model.encoder(image2)
mixed_hidden = λ * hidden1 + (1-λ) * hidden2
output = model.decoder(mixed_hidden)
```

**Results:**
- ImageNet: ~1-2% accuracy improvement
- Works across domains (vision, NLP, tabular)

### Q15: How do you decide between different regularization techniques in practice?

**Answer:**

**Decision Framework:**

**Step 1: Understand Your Problem**

| Characteristic | Recommended Approach |
|---------------|---------------------|
| **High-dimensional (n << p)** | L1 or Elastic Net |
| **Correlated features** | L2 or Elastic Net |
| **Known feature importance** | Feature selection + L2 |
| **Black box is okay** | Ensemble + Dropout |
| **Need interpretability** | L1 (sparse) |
| **Deep neural network** | Dropout + BN + Data Aug |
| **Small dataset** | Strong regularization + augmentation |
| **Large clean dataset** | Light regularization |

**Step 2: Start Simple**
```python
# Baseline: No regularization
baseline = LinearRegression().fit(X, y)
baseline_score = cross_val_score(baseline, X, y).mean()

# If overfitting detected (train >> val):
# Try L2 first (usually safe default)
ridge = Ridge(alpha=1.0).fit(X, y)
ridge_score = cross_val_score(ridge, X, y).mean()
```

**Step 3: Experiment Systematically**
```python
from sklearn.model_selection import GridSearchCV

models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

results = {}
for name, model in models.items():
    grid = GridSearchCV(model, {'alpha': [0.01, 0.1, 1, 10]}, cv=5)
    grid.fit(X, y)
    results[name] = grid.best_score_

best_model = max(results, key=results.get)
print(f"Best: {best_model} with score {results[best_model]}")
```

**Step 4: Combine Techniques**
```python
# Often best to use multiple
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.BatchNorm1d(50),      # Regularization 1
    nn.ReLU(),
    nn.Dropout(0.5),         # Regularization 2
    nn.Linear(50, 10)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001      # Regularization 3 (L2)
)

# + Early stopping              # Regularization 4
# + Data augmentation           # Regularization 5
```

**Real-World Example:**

**Problem:** Predict house prices (1000 samples, 50 features)

**Approach:**
1. ✅ Check train vs val: Train 0.95, Val 0.75 → Overfitting
2. ✅ Try Ridge: Val improves to 0.82
3. ✅ Try Lasso: Val reaches 0.84 (some features removed)
4. ✅ Try Elastic Net (α=0.5): Val peaks at 0.85
5. ✅ Add feature engineering + Elastic Net: Val reaches 0.87
6. ✅ **Final choice:** Elastic Net with α=0.5, l1_ratio=0.7

**Key Principle:** 
> "Use as little regularization as needed for good validation performance. More isn't always better!"
