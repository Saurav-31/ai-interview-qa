# CLIP (Contrastive Language-Image Pre-training)

## Question
Explain CLIP. How does it enable zero-shot image classification?

## Answer

### Overview
CLIP is a multimodal model that learns to align images and text in a shared embedding space. Developed by OpenAI, it powers many modern vision-language applications.

## Core Idea

**Learn from image-text pairs:**
- Train on 400M (image, caption) pairs from internet
- Match images to correct text descriptions
- **No need for labeled classes!**

## Architecture

### Dual Encoder Design

```
Image Encoder          Text Encoder
(Vision Transformer    (Transformer)
 or ResNet)
     ↓                      ↓
Image Embedding       Text Embedding
     \                    /
      \                  /
       \                /
        Cosine Similarity
```

**Key Components:**

1. **Image Encoder:** ViT or ResNet → d-dimensional embedding
2. **Text Encoder:** Transformer → d-dimensional embedding
3. **Projection:** Both map to shared embedding space

## Training Objective: Contrastive Learning

Given batch of N (image, text) pairs:

**Goal:** Maximize similarity for correct pairs, minimize for incorrect

### InfoNCE Loss

$$\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j) / \tau)}$$

**Symmetric:** Also compute loss from text to image

```python
# CLIP Training (Simplified)

# Encode batch
image_features = image_encoder(images)  # (N, d)
text_features = text_encoder(texts)      # (N, d)

# Normalize
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

# Compute similarity matrix
logits = image_features @ text_features.T / temperature  # (N, N)

# Labels: diagonal (i matches i)
labels = torch.arange(N)

# Contrastive loss (both directions)
loss_i2t = F.cross_entropy(logits, labels)
loss_t2i = F.cross_entropy(logits.T, labels)

loss = (loss_i2t + loss_t2i) / 2
```

**Intuition:**
- Image i should match text i (diagonal)
- Image i should NOT match other texts (off-diagonal)

## Zero-Shot Classification

**Key Insight:** Turn classification into image-text matching!

### Process

**1. Create text prompts for each class:**
```python
# For ImageNet classification
classes = ["dog", "cat", "car", "airplane", ...]

prompts = [f"a photo of a {c}" for c in classes]
# "a photo of a dog"
# "a photo of a cat"
# ...
```

**2. Encode all prompts:**
```python
text_features = clip.encode_text(prompts)  # (num_classes, d)
text_features = F.normalize(text_features, dim=-1)
```

**3. Encode image:**
```python
image_features = clip.encode_image(image)  # (1, d)
image_features = F.normalize(image_features, dim=-1)
```

**4. Compute similarities:**
```python
similarities = image_features @ text_features.T  # (1, num_classes)
predicted_class = similarities.argmax()
```

### Example Code

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("cat.jpg")

# Define classes
classes = ["cat", "dog", "bird", "fish"]
texts = [f"a photo of a {c}" for c in classes]

# Prepare inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# Forward pass
outputs = model(**inputs)

# Get similarities
logits_per_image = outputs.logits_per_image  # (1, num_classes)
probs = logits_per_image.softmax(dim=1)

# Predict
predicted_idx = probs.argmax()
print(f"Predicted: {classes[predicted_idx]} ({probs[0, predicted_idx]:.2%})")
```

## Prompt Engineering

**Performance depends on prompt!**

### Strategies

```python
# Basic
"cat"

# Better
"a photo of a cat"

# Best (ensembling)
templates = [
    "a photo of a {}",
    "a image of a {}",
    "a picture showing a {}",
    "a good photo of a {}",
]

# Encode all, average
embeddings = []
for template in templates:
    text = template.format(class_name)
    emb = clip.encode_text(text)
    embeddings.append(emb)

class_embedding = torch.stack(embeddings).mean(dim=0)
```

## Performance

**ImageNet Zero-Shot:**
- CLIP (ViT-L/14): **76.2%** accuracy
- **No training on ImageNet labels!**
- Rivals some supervised models

**Robustness:**
- Better on distribution shifts vs standard supervised models
- Generalizes to new domains

## Applications

### 1. Text-to-Image Search
```python
# Search image database with text query
query = "a sunset over mountains"
query_emb = clip.encode_text(query)

# Compute similarities with all images
similarities = query_emb @ image_embeddings.T
top_images = similarities.topk(k=10)
```

### 2. Image-to-Text Retrieval
```python
# Find captions for an image
image_emb = clip.encode_image(image)
similarities = image_emb @ text_embeddings.T
top_captions = similarities.topk(k=5)
```

### 3. Conditioning for Diffusion Models

**Stable Diffusion uses CLIP text encoder!**
```
Text → CLIP Text Encoder → Embedding → [UNet Denoising] → Image
```

### 4. Vision-Language Tasks
- Visual Question Answering
- Image Captioning
- Multimodal Retrieval

## Architecture Variants

| Model | Image Encoder | Params | Performance |
|-------|---------------|--------|-------------|
| CLIP RN50 | ResNet-50 | 102M | 59.6% |
| CLIP ViT-B/32 | ViT-B/32 | 151M | 63.2% |
| CLIP ViT-L/14 | ViT-L/14 | 428M | 76.2% |

## Key Innovations

1. **Contrastive learning** at scale (400M pairs)
2. **Natural language supervision** (no manual labels)
3. **Zero-shot transfer** via text prompts
4. **Robust** to distribution shift

## Limitations

❌ **Object counting:** "one cat" vs "two cats" often confused  
❌ **Spatial relationships:** "cat on table" vs "table on cat"  
❌ **Fine-grained:** Struggles with similar classes  
❌ **Abstract concepts:** Better with concrete objects

## Evolution

**CLIP → Follow-ups:**
- **ALIGN (Google):** 1.8B pairs, similar approach
- **FLIP:** More efficient training
- **OpenCLIP:** Open reproduction with better data
- **BLIP/BLIP-2:** Better captioning + retrieval

## Key Takeaways

1. **Dual encoder** for images and text
2. **Contrastive learning** on (image, text) pairs
3. **Zero-shot classification** via prompt matching
4. **No labeled data** needed (learns from web captions)
5. **Powers modern applications:** Stable Diffusion, DALL-E
6. **Prompt engineering** matters!
7. **Strong generalization** to new domains

## Tags
#CLIP #VisionLanguage #Multimodal #ZeroShot #Contrastive #StableDiffusion

## Difficulty
Medium-Hard

## Related Questions
- What is contrastive learning?
- How does Stable Diffusion use CLIP?
- Compare CLIP with ALIGN
