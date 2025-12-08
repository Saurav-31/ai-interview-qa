# Transformer Architecture

## Question
Explain the Transformer architecture. What are its key components and why has it become the foundation for modern LLMs?

## Answer

### Overview
The Transformer (Vaswani et al., 2017) replaced recurrence with attention mechanisms, enabling parallel processing and better long-range dependencies.

### Why Transformers?

**Problems with RNNs/LSTMs:**
❌ Sequential processing (no parallelization)  
❌ Vanishing gradients  
❌ Limited context window  

**Transformer Advantages:**
✅ Fully parallelizable  
✅ Captures long-range dependencies  
✅ Scalable to massive models  

## Architecture Overview

```
Input → Embedding → Encoder → Decoder → Output
          ↓           ↓         ↓
    Positional  Multi-Head  Multi-Head
    Encoding    Attention   Attention
```

## Key Components

### 1. Input Embeddings + Positional Encoding

**Positional Encoding:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Adds position information since Transformers have no inherent order.

### 2. Multi-Head Self-Attention

**Self-Attention:**
$$\text{Attention}(Q, K, V) = \softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- Q: Queries (what I'm looking for)
- K: Keys (what I have)
- V: Values (actual content)

**Multi-Head:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**Benefits:**
- Learn different relationship types
- Attend to different positions
- Ensemble effect

**Standard:** 8 heads, d_model = 512

### 3. Feed-Forward Networks

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Adds non-linearity after attention.

### 4. Residual Connections + Layer Normalization

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

Enables gradient flow (like ResNet).

## Encoder Architecture

```
Input → [Multi-Head Self-Attention]
     → Add & Norm
     → [Feed-Forward Network]
     → Add & Norm
     → Output
```

Stacked 6 times in original Transformer.

## Decoder Architecture

```
Input → [Masked Self-Attention]
     → Add & Norm
     → [Cross-Attention with Encoder]
     → Add & Norm
     → [Feed-Forward Network]
     → Add & Norm
     → Output
```

**Key differences:**
1. **Masked attention** - prevents looking ahead
2. **Cross-attention** - attends to encoder output

## Masking

### Causal Mask (Look-Ahead)

```
Position:  1  2  3  4
Token 1:  [1, 0, 0, 0]  ← Only sees itself
Token 2:  [1, 1, 0, 0]  ← Sees tokens 1-2
Token 3:  [1, 1, 1, 0]  ← Sees tokens 1-3
Token 4:  [1, 1, 1, 1]  ← Sees all
```

## PyTorch Implementation

```python
import torch.nn as nn

model = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)
```

## Modern Variants

### Encoder-Only (BERT-style)
- Understanding tasks (classification, NER)
- Bidirectional context

### Decoder-Only (GPT-style)
- Generation tasks
- Causal (left-to-right) attention
- **Most common for LLMs**

### Encoder-Decoder (T5, BART)
- Seq2seq tasks (translation, summarization)

## Complexity

**Time:** O(n² · d) where n = sequence length  
**Space:** O(n²)

**Problem:** Quadratic scaling limits long sequences

**Solutions:**
- Sparse attention
- Linear attention (Performer)
- Sliding window (Longformer)

## Why Transformers Dominate

1. **Parallelization** → Fast training
2. **Long-range dependencies** → Better context
3. **Scalability** → Billions of parameters
4. **Transfer learning** → Pre-train then fine-tune
5. **Versatility** → NLP, Vision, Multimodal

## Training Configuration

**Original Transformer (Base):**
- d_model = 512
- heads = 8
- layers = 6 encoder + 6 decoder
- parameters = ~65M

**Modern LLMs (GPT-3):**
- d_model = 12288
- heads = 96
- layers = 96
- parameters = 175B

## Tags
#LLMs #Transformer #Attention #NLP #DeepLearning #BERT #GPT

## Difficulty
Hard

## Related Questions
- What is self-attention mechanism?
- How does BERT differ from GPT?
- Explain positional encoding
