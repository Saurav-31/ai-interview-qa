# 🎯 Mastery File: 3D/Generative Vision Interview Preparation

This document is structured in **dependency order** to build a strong theoretical foundation before tackling the latest research. Focus on understanding the **connections** and **contrasts** between the Classic CV and the Neural Rendering concepts.

---

## 📚 Phase 1: Geometric Foundations & Single-View Ambiguity (Classical CV)

### Concept 1.1: Multi-View Geometry, SfM, and MVS

| Concept | Key Knowledge / Interview Focus | Famous Blog / Paper Link |
| :--- | :--- | :--- |
| **Pinhole Model** | **Intrinsics ($K$):** $f_x, f_y, c_x, c_y$. **Extrinsics ($R, T$):** Camera pose. | *Tool Doc:* [COLMAP Documentation](https://demuc.de/colmap/) |
| **SfM/MVS** | **SfM:** Sparse points, $R, T$ estimation. **MVS:** Dense reconstruction (depth/mesh). **Contrast:** Failure modes (textureless, reflective). | *Survey:* [A Survey of 3D Reconstruction: The Evolution from Multi-View Geometry to NeRF and 3DGS](https://www.mdpi.com/1424-8220/25/18/5748) |
| **Single-View 3D Challenge**| Inherent depth **ambiguity**. Reliance on learned **shape priors** to hallucinate depth. Voxel vs. Implicit SDF output. | *Survey:* [Single-View 3D Reconstruction: A Survey of Deep Learning Methods](https://www.researchgate.net/publication/348324048_Single-View_3D_Reconstruction_A_Survey_of_Deep_Learning_Methods) |

#### 15 High-Quality Questions & Answers

1.  **Q: Explain the role of the Essential Matrix ($E$) vs. the Fundamental Matrix ($F$).**
    * **A:** $F$ relates corresponding points in two images without requiring camera calibration (intrinsics). $E$ is $F$ post-calibration ($E = K^T F K$), relating points in normalized coordinates. $E$ is crucial for recovering the relative camera pose ($R, T$) up to a scale factor.
2.  **Q: Why is RANSAC necessary in the SfM pipeline?**
    * **A:** **RANSAC (Random Sample Consensus)** is used to robustly estimate the camera pose ($R, T$) by filtering out outliers (incorrect feature matches) that would otherwise skew the least-squares optimization, ensuring only inlier matches are used for geometry calculation.
3.  **Q: What specific data does a NeRF pipeline inherit from a classic SfM process?**
    * **A:** The **Extrinsic Parameters** ($R, T$ matrices) and the **Intrinsic Parameters** ($K$ matrix). NeRF needs these precise camera poses to accurately trace rays for training.
4.  **Q: Describe the input and output of **Bundle Adjustment**.**
    * **A:** **Input:** Initial camera poses ($R, T$), intrinsic parameters ($K$), and a sparse 3D point cloud. **Output:** A globally optimized, refined set of camera poses and 3D point locations that minimizes the total reprojection error across all views simultaneously.
5.  **Q: Why does classical MVS struggle with reflective or transparent objects?**
    * **A:** MVS relies on the **photo-consistency** principle (a 3D point has the same color across views). Reflective/transparent objects are **non-Lambertian**, meaning their color *changes* with the viewing direction, violating the photo-consistency assumption.
6.  **Q: What is the main drawback of using Voxel-based output for Single-View 3D reconstruction?**
    * **A:** Voxel output suffers from **cubic complexity** ($O(N^3)$), meaning memory and computation requirements increase drastically for high-resolution scenes. This limits the achievable detail compared to implicit or point-based methods.
7.  **Q: How does a learning-based Single-View 3D model overcome the lack of depth information?**
    * **A:** By learning a powerful **shape prior** from large 3D datasets (e.g., ShapeNet). The network learns the statistical regularity of object shapes and uses this learned knowledge to hallucinate the most probable 3D structure corresponding to the 2D input.
8.  **Q: Explain the meaning of **Homography** and when it's used in CV.**
    * **A:** A Homography is a $3\times3$ matrix that describes the perspective transformation between two images of the **same plane**. It's used for tasks like image stitching, warping, and perspective correction.
9.  **Q: What is the **Epipolar Constraint** and how does it simplify geometry estimation?**
    * **A:** The constraint dictates that a 3D point projected onto one image must lie on the **epipolar line** in the second image. This reduces the search space for correspondences from a 2D area to a 1D line, greatly increasing efficiency and robustness.
10. **Q: What is the primary advantage of a Point Cloud output over a Voxel output in Single-View 3D?**
    * **A:** Point clouds are **scale-free** and **memory efficient** for sparse surfaces. They avoid the $O(N^3)$ complexity of voxel grids and are a direct input to many graphics pipelines.
11. **Q: Contrast **Monocular** vs. **Stereo** Depth Estimation techniques.**
    * **A:** **Stereo** explicitly calculates depth via triangulation using **disparity** (the difference in pixel coordinates between two calibrated cameras). **Monocular** implicitly predicts depth using a learned model (CNN) and priors, relying on scene context and training data.
12. **Q: What is the difference between a **Dense** and a **Sparse** point cloud, and which is the initial output of SfM?**
    * **A:** **Sparse** (initial SfM output) contains only the 3D points corresponding to matched features. **Dense** (MVS output) attempts to reconstruct all visible surfaces.
13. **Q: How can feature drift in a long SfM video sequence be mitigated?**
    * **A:** By using **global optimization** techniques like **Bundle Adjustment**, which minimizes the reprojection error across *all* frames simultaneously, thus distributing the error and preventing accumulation.
14. **Q: What problem does **Normalized Cross-Correlation (NCC)** help solve in MVS?**
    * **A:** NCC is a metric used to find image correspondences by measuring the similarity of pixel patches while being **invariant to affine changes in brightness and contrast**, making it robust to slight lighting variations.
15. **Q: In the Pinhole model, what does the $Z$ coordinate of a 3D point represent, and why is it crucial?**
    * **A:** $Z$ is the **depth** of the point along the camera axis. It is crucial because the 2D projection is inversely proportional to $Z$. Without $Z$, the actual size and position of the object cannot be determined (the scale ambiguity). .

---

## 🧠 Phase 2: Neural Rendering (NeRF & 3DGS)

### Concept 2.1: NeRF - Implicit Representation and Volumetric Rendering

| Concept | Key Knowledge / Interview Focus | Famous Blog / Paper Link |
| :--- | :--- | :--- |
| **NeRF Core Theory** | MLP: $(\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$. **Volumetric Rendering** integral. **Limitations:** Slow inference, large model. | *Paper:* [NeRF: Representing Scenes as Neural Radiance Fields...](https://arxiv.org/abs/2003.08934) |
| **Positional Encoding (PE)** | **Function:** Maps inputs to a high-dimensional space using $\sin(\cdot)$ and $\cos(\cdot)$ to counteract the network's low-frequency bias. | *Paper Link:* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://neuralfields.cs.brown.edu/paper_33.html) |

#### 15 High-Quality Questions & Answers

1.  **Q: Write the core integral for Volumetric Rendering and explain the terms $T(t)$ and $\sigma$.**
    * **A:** $$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$$ $T(t)$ (Transmittance) is the accumulated opacity, representing the probability the light has *not* been blocked. $\sigma$ (Volume Density) is the probability the ray terminates *at* point $t$, defining geometry.
2.  **Q: What is the mathematical formulation of the **Positional Encoding** applied to the coordinate $p$?**
    * **A:** The mapping $\gamma(p)$ uses sinusoidal functions: $\gamma(p) = \left( \sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \right)$ where $L$ is the number of frequencies.
3.  **Q: Why are $\mathbf{x}$ and $\mathbf{d}$ concatenated at different stages in the NeRF MLP architecture?**
    * **A:** $\mathbf{x}$ (position) is used early to determine the geometric properties ($\sigma$ and the base color). $\mathbf{d}$ (viewing direction) is introduced later to only modulate the final color $\mathbf{c}$. This ensures $\sigma$ (geometry) remains view-independent while the color can be view-dependent (for reflections).
4.  **Q: What is the function of the **Hierarchical Sampling** component in the original NeRF?**
    * **A:** It is a variance reduction technique. A **Coarse Network** uses uniform sampling to identify high-density areas, and a **Fine Network** then samples more points densely *only* in those crucial areas, speeding up convergence and improving detail.
5.  **Q: How does NeRF explicitly model non-Lambertian (reflective) surfaces?**
    * **A:** Through the inclusion of the **viewing direction $\mathbf{d}$** in the color prediction part of the MLP. When $\mathbf{d}$ changes, the predicted color $\mathbf{c}$ can change, allowing the network to encode specular reflections.
6.  **Q: What is the fundamental difference in the loss function used by NeRF versus a classical MVS algorithm?**
    * **A:** NeRF uses a simple **Mean Squared Error (MSE)** between the **rendered pixel color** and the **ground truth pixel color**. MVS typically uses **photo-consistency metrics** (e.g., NCC) applied to image patches across views. NeRF's loss is end-to-end differentiable.
7.  **Q: What is the "baking" problem that NeRF and its static successors suffer from?**
    * **A:** The model *bakes* the scene **lighting and shadows** directly into the learned color and density. If the lighting changes or an object moves, the model becomes invalid because the color field is static.
8.  **Q: How is the geometry (e.g., a mesh) extracted from a trained NeRF?**
    * **A:** By treating the volume density $\sigma$ as an **implicit function** and applying an isosurface extraction algorithm, such as **Marching Cubes**, at a certain density threshold. .
9.  **Q: Why is NeRF inference (rendering a new view) so much slower than training?**
    * **A:** Inference requires **forward propagation** through the deep MLP **hundreds of times** (one for each sample along the ray) for **every single pixel** in the output image, which is computationally intensive and not easily cached.
10. **Q: What is the main idea behind **Mip-NeRF**?**
    * **A:** Mip-NeRF addresses **aliasing** by modeling the scene not with single rays, but with **conical frustums** (integrated rays). This allows the network to represent the scene at continuous scales, reducing flickering and jagged edges.
11. **Q: How does NeRF represent **empty space** vs. **occupied space**?**
    * **A:** Empty space is represented by $\sigma \approx 0$ (zero opacity). Occupied space (surfaces) is represented by high, localized $\sigma$ values, ensuring the ray terminates at that location.
12. **Q: What is the typical network architecture used for the NeRF MLP?**
    * **A:** Typically, an 8-layer, 256-width fully connected ReLU network (MLP) with skip connections from the input to the middle layers to aid gradient flow.
13. **Q: Why are camera poses required to be known precisely for NeRF training?**
    * **A:** NeRF is an **optimization** problem, not a feature-matching one. If the ray paths defined by the camera poses are incorrect, the network will learn to map the wrong 3D points to the input pixels, resulting in blurry and inconsistent rendering.
14. **Q: What is the significance of the $\frac{1}{\sin(\theta)}$ factor often used in the discretized volumetric integral?**
    * **A:** This factor, or similar inverse delta-distance terms, is used in the discrete approximation of the integral to ensure that volume density $\sigma$ is correctly scaled by the distance between samples, which is crucial for accurate opacity calculation.
15. **Q: Describe the main difference between NeRF and **Signed Distance Fields (SDFs)** for implicit representation.**
    * **A:** NeRF represents density and color ($\sigma, c$), which is **view-synthesis friendly**. SDFs represent the **distance to the surface** ($d$), which is **geometry-friendly** (guaranteed watertight meshes).

### Concept 2.2: 3D Gaussian Splatting (3DGS)

| Concept | Key Knowledge / Interview Focus | Famous Blog / Paper Link |
| :--- | :--- | :--- |
| **3DGS Core Theory** | **Explicit** scene representation. Storing position, covariance, opacity, and Spherical Harmonics (SH). **Differentiable Rasterization**. | *Paper:* [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf) |
| **Rasterization** | **Forward Splatting** vs. Ray Marching. **Adaptive Densification** and **Pruning** for optimization. | *YouTube:* [The 3D Gaussian Splatting Adventure: Past, Present, Future (George Drettakis)](https://www.youtube.com/watch?v=DjOqkVIlEGY) |

#### 15 High-Quality Questions & Answers

1.  **Q: What is the primary reason 3DGS achieves real-time rendering speed, unlike NeRF?**
    * **A:** 3DGS uses an **explicit** representation and relies on fast, well-optimized **GPU rasterization** (forward splatting) for rendering, which replaces NeRF's computationally expensive, sequential **ray marching** through a deep neural network.
2.  **Q: How does 3DGS model view-dependent color without an MLP?**
    * **A:** It uses **Spherical Harmonics (SH)** coefficients stored per Gaussian. SH are analytic functions that can approximate complex lighting and view-dependent effects by varying the color based on the viewing direction.
3.  **Q: Explain the role of the **covariance matrix** $\Sigma$ for a 3D Gaussian.**
    * **A:** $\Sigma$ defines the **shape, scale, and orientation** of the Gaussian ellipsoid in 3D space. When projected to 2D, it determines the shape and size of the splat on the screen, allowing anisotropic modeling (stretched/compressed shapes).
4.  **Q: How does 3DGS achieve **differentiability** for training?**
    * **A:** It uses a custom-designed, fully differentiable **tile-based rasterizer**. This component is optimized for GPUs and allows gradients to flow back from the 2D pixel loss through the 3D Gaussian properties (position, covariance, opacity).
5.  **Q: Describe the **adaptive densification** strategy in 3DGS.**
    * **A:** During training, if a Gaussian has a high color error (poor fit) or has become overly large, it is **densified** by cloning itself or splitting into two smaller Gaussians. This increases complexity only in high-error areas.
6.  **Q: How does 3DGS ensure correct blending with transparency (correct rendering order)?**
    * **A:** The Gaussians are sorted **globally by depth** (farthest to nearest) relative to the camera viewpoint at every frame. This ensures that the alpha blending for accumulation correctly follows the standard front-to-back compositing rule.
7.  **Q: What are the primary inputs required to begin a 3DGS training process?**
    * **A:** A sparse **Point Cloud** (usually generated by COLMAP/SfM) to serve as the initialization for the Gaussian positions, and the corresponding **Calibrated Camera Poses** ($R, T, K$) for all input images.
8.  **Q: What is the difference between **densification** (cloning/splitting) and **pruning** in 3DGS?**
    * **A:** **Densification** adds new Gaussians (increasing detail) in high-error regions. **Pruning** removes useless Gaussians (low opacity or far outside the desired viewing frustum) to maintain memory efficiency and stability.
9.  **Q: What is the advantage of using **Spherical Harmonics (SH)** over a simple RGB color value in 3DGS?**
    * **A:** SH allows the Gaussian's color to change smoothly based on the viewing direction, encoding **view-dependent effects** (like reflections and shadows) without the computational overhead of an MLP.
10. **Q: How does 3DGS manage memory and prevent the creation of useless 'floaters'?**
    * **A:** It applies **pruning** to remove Gaussians whose opacity is very low or whose scale has grown excessively large without contributing meaningfully to the rendered image.
11. **Q: Why is the initialization of the 3D Gaussians from the sparse SfM point cloud important?**
    * **A:** A good initialization (position, initial opacity/scale) significantly reduces the training time and improves the quality of the final model, providing a strong geometric starting point that the optimizer refines.
12. **Q: What problem does **alpha blending** solve in the 3DGS rendering equation?**
    * **A:** Alpha blending is used to **accumulate color and transparency** along the ray (in the forward splatting process) to produce the final, composited pixel color, correctly accounting for occlusion by previous Gaussians.
13. **Q: How does 3DGS handle unbounded scenes or scenes captured with 360° cameras?**
    * **A:** Like Mip-NeRF 360, 3DGS often employs a **scene parameterization** (e.g., mapping to a bounding sphere or using a specialized coordinate system) to efficiently represent the vast, empty space in unbounded scenes.
14. **Q: What would happen if the Gaussians were restricted to be **isotropic** (spherical)?**
    * **A:** The model would struggle to represent thin structures (like wires or edges) accurately. Anisotropic Gaussians are essential for representing geometry with high fidelity using fewer primitives.
15. **Q: What is the core difference in the **data structure** used by NeRF vs. 3DGS?**
    * **A:** NeRF uses the **weights of a neural network** (implicit, unstructured). 3DGS uses an **explicit list/array of structured primitives** (the Gaussians), which makes it amenable to traditional graphics pipelines and explicit manipulation. .

---

## 🌊 Phase 3: Generative Models (Diffusion & SDS)

### Concept 3.1: Diffusion Models and Score Distillation

| Concept | Key Knowledge / Interview Focus | Famous Blog / Paper Link |
| :--- | :--- | :--- |
| **Core Diffusion Theory** | **DDPMs:** Forward/Reverse processes. **U-Net** as the noise predictor. **Loss:** Minimizing the difference between predicted and actual noise $\epsilon$. | *Blog:* [The Illustrated Diffusion Model](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) |
| **SDS (Score Distillation Sampling)** | Using the 2D Diffusion prior's gradient to optimize a 3D model. **DreamFusion** implementation. **View-dependent prompt** conditioning. | *Paper:* [DREAMFUSION: TEXT-TO-3D USING 2D DIFFUSION](https://openreview.net/pdf?id=FjNys5c7VyY) |

#### 15 High-Quality Questions & Answers

1.  **Q: Why is the Diffusion model trained to predict the **noise** ($\epsilon$) instead of the clean image ($x_0$)?**
    * **A:** Predicting the noise is simpler and more numerically stable. The reweighted loss function derived in DDPM simplifies to a single MSE term between the predicted noise and the actual noise, providing a clean training objective.
2.  **Q: How does **Classifier-Free Guidance (CFG)** enhance the quality of generated diffusion images?**
    * **A:** CFG combines the output of a **conditional model** (guided by a prompt) and an **unconditional model** during inference. By extrapolating in the direction of the conditional output, it increases the adherence of the generated image to the text prompt (fidelity).
3.  **Q: What is the mathematical concept of the **score function** that SDS is based on?**
    * **A:** The score function is the gradient of the log probability density of the data distribution: $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. It points in the direction that maximizes the likelihood of $\mathbf{x}$ being a real sample. SDS uses the Diffusion model to approximate this gradient for a rendered 3D view.
4.  **Q: Explain the **Score Distillation Sampling (SDS) loss** and what parameters it optimizes.**
    * **A:** The loss uses the gradient of the frozen 2D Diffusion model to update the parameters ($\theta$) of the 3D representation. The goal is to optimize the 3D model such that its renderings have a high likelihood under the 2D model's learned distribution.
5.  **Q: Why does the basic SDS loss often result in **over-saturated** and **over-smooth/blobby** 3D objects?**
    * **A:** The SDS loss is an **unbiased estimator** of a gradient that attempts to maximize the likelihood of a single rendered view. This causes the optimizer to focus too much on simple, consistent, and easy-to-render solutions, often lacking high-frequency detail (the "Janus" problem is related to this).
6.  **Q: How do later methods like **Variational Score Distillation (VSD)** (used in ProlificDreamer) address the limitations of basic SDS?**
    * **A:** VSD reformulates the problem from maximizing the likelihood of one view to a **variational objective** that optimizes a full distribution of images, leading to better **multi-view consistency** and preventing the over-smoothing seen with SDS.
7.  **Q: What is the benefit of using **Latent Diffusion Models (LDMs)** over pixel-space diffusion models?**
    * **A:** LDMs perform the forward/reverse diffusion process in a compressed, lower-dimensional **latent space** encoded by a VAE. This drastically reduces the computational cost and memory footprint, enabling training on consumer GPUs.
8.  **Q: How is the **time step** $t$ used as a condition in a Diffusion Model's U-Net?**
    * **A:** The time step $t$ (which indicates the level of noise) is typically encoded using **sinusoidal positional encoding** and then injected into the U-Net layers via **cross-attention** or simple concatenation to inform the network about *how much* noise it should be predicting.
9.  **Q: Contrast a **Generative Adversarial Network (GAN)** with a Diffusion Model regarding training stability.**
    * **A:** GANs suffer from **training instability** and require careful balancing of the Generator and Discriminator. Diffusion Models are trained with a simple, fixed **MSE loss**, making them significantly easier and more stable to optimize.
10. **Q: What is the role of the **Cross-Attention** mechanism in text-conditional diffusion models?**
    * **A:** Cross-attention links the **text embeddings** (generated by a model like CLIP) to the **image features** within the U-Net. This allows the network to condition its denoising steps specifically on the semantic content of the text prompt.
11. **Q: Why are **Spherical Harmonics (SH)** often used to shade the NeRF/3DGS during SDS optimization?**
    * **A:** Using SH allows for simple, fast **relighting** of the rendered scene with a random light source. Shading helps the 2D prior to focus on geometric details and normals, rather than just the texture.
12. **Q: What is the core challenge of integrating 3DGS with SDS/Diffusion for text-to-3D?**
    * **A:** The primary challenge is that 3DGS's **explicit** geometry (the Gaussians) is highly sensitive to the noisy gradients produced by the SDS loss, making the geometry initialization and stability even more critical than when optimizing a NeRF MLP.
13. **Q: What does the **$\alpha$ (alpha)** parameter represent in the forward diffusion process?**
    * **A:** $\alpha_t = 1 - \beta_t$ (where $\beta$ is the noise variance schedule). $\alpha_t$ controls the amount of signal retained at each step. $\bar{\alpha}_t$ is the cumulative product of $\alpha$ values, which determines the signal level for sampling $x_t$ from $x_0$ at any time $t$.
14. **Q: What is the relationship between the **VAE Encoder/Decoder** and Latent Diffusion?**
    * **A:** The **Encoder** compresses the image into a low-dimensional **latent code** for diffusion training, and the **Decoder** reconstructs the final high-resolution image from the denoised latent code during inference.
15. **Q: Define the **score field** in the context of sampling from a data distribution.**
    * **A:** The score field is a vector field where, at any point $\mathbf{x}$, the vector $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ points in the direction of maximum probability density of the data distribution (towards the "nearest" real data sample).

The key takeaway from the interview is to demonstrate not just knowledge of the concepts, but the ability to **compare, contrast, and integrate** them across the classical and neural domains. .

To help you understand the real-time performance of 3DGS, watch this video: [3D Gaussian Splatting for Real-Time Radiance Field Rendering - YouTube](https://www.youtube.com/watch?v=Ikqj1B-AfUE).


http://googleusercontent.com/youtube_content/2