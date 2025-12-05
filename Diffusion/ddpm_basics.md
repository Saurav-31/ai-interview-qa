# Diffusion Models (DDPM)

## Question
Explain how diffusion models work. What are the forward and reverse processes?

## Answer

### Overview
Diffusion models generate data by learning to reverse a gradual noising process. State-of-the-art for image generation (DALL-E 2, Stable Diffusion, Imagen).

### Core Idea

**Two Processes:**
1. **Forward (Diffusion):** Gradually add noise to data
2. **Reverse (Denoising):** Learn to remove noise

## Forward Process (Adding Noise)

Gradually destroy data by adding Gaussian noise:

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**Nice Property:** Can sample x_t directly from x_0:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

Where ε ~ N(0, I)

**Intuition:**
- At t=0: x₀ (clean image)
- At t=T: Pure Gaussian noise
- Smoothly interpolates between them

## Reverse Process (Denoising)

Start from noise and iteratively denoise:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

## Training Objective

**Train model to predict noise:**

$$L = \mathbb{E}_{q(x_0)} \left[ \mathbb{E}_{q(x_t|x_0)} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right] \right]$$

### Training Algorithm

```python
# DDPM Training

for iteration in range(num_iterations):
    # 1. Sample image
    x_0 = sample_from_dataset()
    
    # 2. Sample timestep
    t = random_uniform(1, T)
    
    # 3. Sample noise
    epsilon = torch.randn_like(x_0)
    
    # 4. Create noisy image
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    
    # 5. Predict noise
    epsilon_pred = model(x_t, t)
    
    # 6. Compute loss
    loss = MSE(epsilon, epsilon_pred)
    
    # 7. Update
    loss.backward()
    optimizer.step()
```

**Key:** Train to predict noise, not the image!

## Sampling (Generation)

```python
# Start from pure noise
x_T = torch.randn(batch_size, channels, height, width)

x_t = x_T
for t in reversed(range(1, T+1)):
    # Predict noise
    epsilon_pred = model(x_t, t)
    
    # Compute mean
    mu_t = (1 / sqrt(alpha_t)) * (
        x_t - ((1 - alpha_t) / sqrt(1 - alpha_bar_t)) * epsilon_pred
    )
    
    # Add noise (except last step)
    if t > 1:
        noise = torch.randn_like(x_t)
        x_t = mu_t + sigma_t * noise
    else:
        x_t = mu_t

x_0 = x_t  # Generated image
```

Process: x_T → x_{T-1} → ... → x_1 → x_0

**Time:** 1000 steps → slow (10-50 seconds per image)

## Architecture: U-Net

**Structure:**
```
Input (noisy image + timestep)
    ↓
[Downsampling Path] → Conv → Downsample
    ↓
[Bottleneck] → Attention
    ↓
[Upsampling Path] → Upsample → Conv
    ↓
Output (predicted noise)
```

## Faster Sampling: DDIM

**Problem:** DDPM requires 1000+ steps

**DDIM:** Non-Markovian process allows skipping steps
- Generate with 50-100 steps instead of 1000
- **10-20× faster sampling!**

## Conditioning (Guided Generation)

### Classifier-Free Guidance

Train single model for both conditional and unconditional:

$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))$$

**Used by:** Stable Diffusion, DALL-E 2

```python
# Training: randomly drop condition
if random.random() < 0.1:
    condition = None

# Sampling: guided prediction
noise_uncond = model(x_t, t, condition=None)
noise_cond = model(x_t, t, condition=condition)
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

## Applications

1. **Text-to-Image:** Stable Diffusion, DALL-E 2, Imagen
2. **Image-to-Image:** Inpainting, super-resolution, editing
3. **Other:** Audio, video, 3D generation

## Advantages vs GANs

| Aspect | Diffusion | GANs |
|--------|-----------|------|
| **Training Stability** | ✅ Stable | ❌ Unstable |
| **Sample Quality** | ✅ Excellent | ✅ Excellent |
| **Diversity** | ✅ High | ⚠️ Can lack |
| **Sampling Speed** | ❌ Slow | ✅ Fast |
| **Training** | ✅ Simple (MSE) | ❌ Complex |

## Key Takeaways

1. **Two processes:** Forward (add noise), Reverse (denoise)
2. **Train to predict noise** at each step
3. **Sampling is iterative:** 1000 steps (DDPM) or 50-100 (DDIM)
4. **U-Net architecture** with time embeddings
5. **Conditioning via guidance** (classifier-free is standard)
6. **Stable training** vs GANs
7. **SOTA for image generation**

## Tags
#Diffusion #GenerativeModels #DDPM #DDIM #StableDiffusion #ImageGeneration

## Difficulty
Hard

## Related Questions
- What is Stable Diffusion?
- Compare diffusion models with GANs
- Explain classifier-free guidance
