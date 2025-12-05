# Gradient Descent Variants

## Question
Explain the different variants of gradient descent. What are the tradeoffs between batch, mini-batch, and stochastic gradient descent?

## Answer

### Overview
Gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent.

### Core Update Rule
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

## 1. Batch Gradient Descent (BGD)

Computes gradient using **entire training dataset**.

$$\nabla_\theta J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta J(\theta; x^{(i)}, y^{(i)})$$

✅ Stable convergence  
❌ Slow for large datasets  

## 2. Stochastic Gradient Descent (SGD)

Updates parameters using **one sample** at a time.

✅ Fast updates  
✅ Can escape local minima  
❌ Noisy convergence  

## 3. Mini-Batch Gradient Descent

Updates using **small batch** of samples (typically 32-512).

✅ Balance speed and stability  
✅ Efficient GPU utilization  
✅ **Most common in practice**  

## Advanced Variants

### Adam (Adaptive Moment Estimation)

```python
# Combines momentum and RMSprop
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient ** 2
theta = theta - learning_rate * m / (sqrt(v) + epsilon)
```

**Most popular in deep learning**

## Comparison Table

| Aspect | Batch GD | Mini-Batch GD | Stochastic GD |
|--------|----------|---------------|---------------|
| **Samples per Update** | All (n) | Batch (b) | 1 |
| **Convergence** | Smooth | Smoother | Noisy |
| **Memory Usage** | High | Medium | Low |
| **Speed per Epoch** | Slow | Fast | Fastest |
| **Common in Practice** | Rare | **Very Common** | Moderate |

## Tags
#MachineLearning #Optimization #GradientDescent #SGD #Adam #DeepLearning

## Difficulty
Medium

## Related Questions
- How does backpropagation work?
- Explain learning rate scheduling
