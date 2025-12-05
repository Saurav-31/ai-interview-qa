# Bias-Variance Tradeoff

## Question
Explain the bias-variance tradeoff in machine learning. How does model complexity affect both?

## Answer

### Overview
The bias-variance tradeoff is a fundamental concept in supervised learning that describes the tension between two sources of error that prevent models from generalizing beyond training data.

### Key Components

**1. Bias (Underfitting)**
- **Definition**: Error from incorrect assumptions in the learning algorithm
- **Characteristics**: 
  - High training error
  - High test error
  - Model is too simple to capture underlying patterns
- **Example**: Using linear regression for highly non-linear data

**2. Variance (Overfitting)**
- **Definition**: Error from sensitivity to small fluctuations in training data
- **Characteristics**:
  - Low training error
  - High test error
  - Model memorizes noise instead of learning patterns
- **Example**: Deep decision tree fitting every training point

**3. Irreducible Error**
- Noise inherent in the data
- Cannot be reduced regardless of model choice

### Mathematical Formulation

Expected test error can be decomposed as:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### Impact of Model Complexity

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| Too Low | High | Low | High |
| Optimal | Medium | Medium | **Lowest** |
| Too High | Low | High | High |

### Practical Strategies

**Reducing High Bias:**
- Increase model complexity
- Add more features
- Reduce regularization
- Train longer

**Reducing High Variance:**
- Get more training data
- Use regularization (L1/L2)
- Reduce model complexity
- Use ensemble methods
- Apply dropout (neural networks)
- Early stopping

### Real-World Example

Consider predicting house prices:
- **High Bias**: Linear model with only square footage → misses non-linear relationships
- **Optimal**: Polynomial features with regularization → captures patterns without overfitting
- **High Variance**: 20-degree polynomial → fits training noise, fails on new data

## Tags
#MachineLearning #Fundamentals #ModelSelection #Overfitting #Underfitting

## Difficulty
Medium

## Related Questions
- What is regularization and how does it help?
- Explain cross-validation and its role in model selection
- What are ensemble methods?
