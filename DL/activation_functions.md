# Activation Functions in Neural Networks

## Question
What are activation functions and why are they needed? Compare different activation functions (ReLU, Sigmoid, Tanh, etc.).

## Answer

### Overview
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

### Why Activation Functions?

Without activation: Multiple layers collapse into single linear transformation.
With activation: Can approximate any continuous function (Universal Approximation Theorem).

## Main Activation Functions

### 1. Sigmoid (Logistic)

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Range:** (0, 1)

✅ Output in (0,1) → suitable for probabilities  
❌ Vanishing gradients  
❌ Not zero-centered  

**Use Cases:** Binary classification output layer, LSTM gates

### 2. Tanh (Hyperbolic Tangent)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Range:** (-1, 1)

✅ Zero-centered  
❌ Still suffers from vanishing gradients  

**Use Cases:** RNN/LSTM hidden states

### 3. ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

**Range:** [0, ∞)

✅ **Computationally efficient**  
✅ No vanishing gradient for positive values  
✅ **Default choice for hidden layers**  
❌ Dying ReLU problem  

```python
import torch.nn as nn

layer = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

### 4. Leaky ReLU

$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$

Typical: α = 0.01

✅ Fixes dying ReLU problem  

### 5. GELU (Gaussian Error Linear Unit)

Used in **BERT, GPT** and modern transformers.

```python
nn.GELU()  # Standard in transformers
```

### 6. Swish (SiLU)

$$\text{Swish}(x) = x \cdot \sigma(x)$$

Used in **EfficientNet**.

## Comparison Table

| Function | Range | Use Case |
|----------|-------|----------|
| **Sigmoid** | (0, 1) | Binary classification (output) |
| **Tanh** | (-1, 1) | RNNs |
| **ReLU** | [0, ∞) | **Default for hidden layers** |
| **GELU** | (-∞, ∞) | **Transformers (BERT, GPT)** |
| **Swish** | (-∞, ∞) | EfficientNet, modern CNNs |

## Decision Guide

**Default Choice:**
```python
nn.ReLU()  # Start here
```

**For Transformers/NLP:**
```python
nn.GELU()  # Standard in BERT, GPT
```

**For Output Layers:**
- Binary: `nn.Sigmoid()`
- Multi-class: `nn.Softmax(dim=1)`
- Regression: No activation

## Tags
#DeepLearning #ActivationFunctions #NeuralNetworks #ReLU #GELU

## Difficulty
Medium

## Related Questions
- What is the vanishing gradient problem?
- How does backpropagation work?
