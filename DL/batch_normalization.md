# Batch Normalization

## Question
What is batch normalization and why does it help training deep neural networks?

## Answer

### Overview
Batch Normalization normalizes inputs of each layer to have zero mean and unit variance across a mini-batch.

### The Problem: Internal Covariate Shift

**Definition:** The distribution of layer inputs changes during training as parameters update.

**Impact:**
- Requires smaller learning rates
- Careful initialization needed
- Slows down training

## How Batch Normalization Works

For mini-batch of inputs x = {x₁, ..., xₘ}:

**Step 1: Compute statistics**
$$\mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma^2_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**Step 2: Normalize**
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$$

**Step 3: Scale and shift**
$$y_i = \gamma \hat{x}_i + \beta$$

γ and β are **learnable parameters**.

## Implementation

### PyTorch

```python
import torch.nn as nn

class NetworkWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),  # BEFORE activation
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training vs Evaluation
model.train()  # Uses batch statistics
model.eval()   # Uses running statistics
```

### For CNNs

```python
nn.Conv2d(3, 64, kernel_size=3),
nn.BatchNorm2d(64),  # Per-channel normalization
nn.ReLU()
```

## Benefits

### 1. Faster Training
✅ Higher learning rates possible  
✅ Faster convergence  

### 2. Reduces Sensitivity to Initialization
✅ Less dependent on weight initialization  

### 3. Regularization Effect
✅ Acts as implicit regularizer  
✅ Can reduce dropout  

### 4. Enables Deeper Networks
✅ Stable gradients in deep networks  

## Inference Mode

At test time, use **running statistics** (exponential moving average computed during training).

## When to Use

✅ **Use when:**
- Training CNNs or fully connected networks
- Batch size ≥ 8
- Standard supervised learning

❌ **Avoid when:**
- Very small batch sizes (< 8) → Use GroupNorm/LayerNorm
- RNNs/Transformers → Use LayerNorm
- Online learning (single samples)

## Alternatives

- **Layer Normalization**: Transformers, RNNs
- **Group Normalization**: Small batches
- **Instance Normalization**: Style transfer

## Common Pitfalls

**Wrong:**
```python
model.train()  # But evaluating!
test_acc = evaluate(model, test_loader)  # Incorrect!
```

**Correct:**
```python
model.eval()  # Switch to evaluation mode
test_acc = evaluate(model, test_loader)
```

## Tags
#DeepLearning #BatchNormalization #Normalization #Training

## Difficulty
Medium

## Related Questions
- What is Layer Normalization?
- Explain internal covariate shift
