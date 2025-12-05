# Backpropagation Algorithm

## Question
Explain how backpropagation works in neural networks. Walk through the mathematical derivation.

## Answer

### Overview
Backpropagation is the algorithm used to compute gradients of the loss function with respect to network parameters using the chain rule.

### Key Insight
**"Backpropagation = Chain Rule + Dynamic Programming"**

## Forward Pass

Simple 2-layer network:

$$z^{[1]} = W^{[1]}x + b^{[1]}$$
$$a^{[1]} = \sigma(z^{[1]})$$
$$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$$
$$\hat{y} = \sigma(z^{[2]})$$
$$L = \frac{1}{2}(\hat{y} - y)^2$$

## Backward Pass

### Output Layer

$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$

$$\delta^{[2]} = (\hat{y} - y) \cdot \sigma'(z^{[2]})$$

$$\frac{\partial L}{\partial W^{[2]}} = \delta^{[2]} \cdot (a^{[1]})^T$$

$$\frac{\partial L}{\partial b^{[2]}} = \delta^{[2]}$$

### Hidden Layer

$$\delta^{[1]} = (W^{[2]})^T \delta^{[2]} \odot \sigma'(z^{[1]})$$

$$\frac{\partial L}{\partial W^{[1]}} = \delta^{[1]} \cdot x^T$$

## General Formulas

For layer l:

$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot \sigma'(z^{[l]})$$

$$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Forward pass
output = model(x)
loss = criterion(output, y)

# Backward pass (automatic differentiation)
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

## Vanishing/Exploding Gradients

**Vanishing:** Gradients become very small → early layers don't learn

**Solutions:**
- Use ReLU activation
- Batch normalization
- Residual connections
- Proper initialization

## Key Takeaways

1. **Backpropagation = efficient chain rule**
2. **Forward pass stores activations** for backward pass
3. **Gradients computed layer-by-layer** from output to input
4. **Modern frameworks automate** this (PyTorch, TensorFlow)

## Tags
#DeepLearning #Backpropagation #GradientDescent #NeuralNetworks

## Difficulty
Hard

## Related Questions
- What are vanishing and exploding gradients?
- How does automatic differentiation work?
