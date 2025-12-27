# Staff ML System Design Interview — Visual Search (No Personalization)

**[Harder Scenario Questions: [GEMINI Based QAs ↗](#gemini-qas)]**


---

## 1. Requirements Gathering

- **Interviewer:** How would you design a visual search system like Google Lens?

- **Candidate (Short Answer):**  
  I’ll start by clarifying the goal and assumptions, then walk through the system at a high level before diving into trade-offs.

- **Candidate (Assumptions):**
  - Product Goal: Return the top 10–15 visually and semantically relevant results.
  - Success Metric: Semantic relevance and correctness, not engagement.
  - Latency: p95 ~200 ms end-to-end for camera-based search.
  - Scale: ~1B indexed images, growing continuously.
  - Availability: High availability since this is a user-facing feature.

- **Candidate (Deep Dive):**  
  Visual search is latency-sensitive and ambiguity-heavy. Camera-based input means we must assume noisy queries, partial objects, and motion blur. This pushes us toward a representation-first design where retrieval dominates latency, and downstream logic is lightweight. I’m intentionally excluding personalization and engagement metrics to keep this a pure search problem.

### Staff-Level Follow-ups (Requirements)

1. **How do you validate these assumptions quickly?**  
   - *Short:* Through logs, synthetic benchmarks, and early latency probes.  
   - *Deep:* I’d inspect real query distributions, simulate camera noise, and run latency budgets per stage to confirm p95 feasibility.

2. **What assumption is most likely wrong?**  
   - *Short:* Latency expectations.  
   - *Deep:* Camera pipelines often add hidden latency (capture, preprocessing, network), so embedding + retrieval budgets may be tighter than expected.

3. **What happens if top-K changes from 10 to 50?**  
   - *Short:* Retrieval cost increases significantly.  
   - *Deep:* Larger K increases ANN fanout and ranking cost; may require multi-stage retrieval or early cutoff.

4. **What does “semantic correctness” mean here?**  
   - *Short:* Results align with user intent, not just visual similarity.  
   - *Deep:* Multiple answers can be valid; correctness is probabilistic, not absolute.

5. **When would you refuse to answer?**  
   - *Short:* Sensitive or unsafe domains.  
   - *Deep:* Medical, legal, or misleading queries require restricted handling or abstention.


---

## 2. High-Level System Overview

- **Interviewer:** Walk me through the system at a high level.

- **Candidate (Short Answer):**  
  The system has four stages: query embedding, ANN retrieval, lightweight ranking, and response formatting.

- **Candidate (Deep Dive):**  
  1. **Query preprocessing:** Normalize, crop, resize, and optionally run object detection to focus on salient regions.  
  2. **Embedding generation:** A fast online vision or vision-language model converts the image into a vector.  
  3. **ANN retrieval:** The embedding is used to retrieve a candidate set from a large vector index.  
  4. **Ranking:** Apply similarity scoring and metadata constraints.  
  5. **Response:** Return results with minimal post-processing to stay within latency budgets.

  The key insight is that embeddings and index design dominate both quality and cost.

### Staff-Level Follow-ups (System Design)

1. **Where is the real bottleneck?**  
   - *Short:* Retrieval and index memory.  
   - *Deep:* ANN fanout and memory locality dominate tail latency.

2. **What would you remove first under latency pressure?**  
   - *Short:* Ranking complexity.  
   - *Deep:* Keep retrieval intact; simplify post-processing.

3. **How do partial failures affect results?**  
   - *Short:* Degraded recall.  
   - *Deep:* System should return partial results rather than fail hard.

4. **Where do you cache?**  
   - *Short:* Query embeddings and hot index shards.  
   - *Deep:* Especially effective for popular landmarks or objects.

5. **What must be synchronous vs async?**  
   - *Short:* Query embedding and retrieval are synchronous.  
   - *Deep:* Corpus embedding and index rebuilds are async.


---

## 3. Representation Learning & Model Choice

- **Interviewer:** What model would you use to generate embeddings?

- **Candidate (Short Answer):**  
  A vision or vision-language model trained with contrastive objectives.

- **Candidate (Deep Dive):**  
  Contrastive learning directly optimizes semantic similarity, which is exactly what visual search needs. Classification models collapse information into labels, whereas embeddings preserve fine-grained similarity. Vision-language training further aligns images with concepts expressed in text, which helps generalization across objects, landmarks, food, and products.

---

- **Interviewer:** Online or offline inference?

- **Candidate (Short Answer):**  
  Queries are embedded online; the corpus is embedded offline.

- **Candidate (Deep Dive):**  
  Online models must be small, fast, and predictable. Offline models can be larger and more expressive since they run asynchronously. This separation lets us optimize latency and cost independently. Corpus embeddings are regenerated periodically and versioned.

### Staff-Level Follow-ups (Embeddings)

1. **How do you know embeddings haven’t collapsed?**  
   - *Short:* Distribution monitoring.  
   - *Deep:* Track variance, nearest-neighbor entropy, and domain-wise recall.

2. **What breaks if embeddings are too small?**  
   - *Short:* Semantic aliasing.  
   - *Deep:* Distinct concepts collapse into similar vectors.

3. **Too large?**  
   - *Short:* Cost and latency.  
   - *Deep:* Memory pressure and ANN inefficiency increase sharply.

4. **How do you evaluate without labels?**  
   - *Short:* Proxy tasks and retrieval consistency.  
   - *Deep:* Stability across augmentations and temporal slices.

5. **How do you handle domain imbalance?**  
   - *Short:* Balanced sampling.  
   - *Deep:* Prevent web images from drowning specialized domains.


---

## 4. Data Sources & Coverage

- **Interviewer:** Users search for objects, landmarks, food, shoes, clothing. Where does all this data come from?

- **Candidate (Short Answer):**  
  From multiple overlapping data sources aligned into a shared embedding space.

- **Candidate (Deep Dive):**  
  - **Web-scale images:** Crawled images with surrounding text cover generic objects and concepts.  
  - **Product catalogs:** High-quality images with structured metadata cover commerce items.  
  - **Landmark datasets:** Geo-tagged images and descriptions enable place recognition.  
  - **Videos:** Frames extracted from videos are treated as images and embedded the same way.

  No single dataset covers everything. Scale and diversity come from combining weakly supervised sources.
### Staff-Level Follow-ups (Data Reality)

1. **How do you prevent noisy web data from dominating?**  
   - *Short:* Reweighting and filtering.  
   - *Deep:* Quality heuristics, domain caps, curriculum training.

2. **What about adversarial metadata?**  
   - *Short:* Don’t trust it blindly.  
   - *Deep:* Metadata is soft signal, never ground truth.

3. **How do new categories appear?**  
   - *Short:* Zero-shot generalization.  
   - *Deep:* Emergent clustering in embedding space.

4. **How do you backfill historical data?**  
   - *Short:* Batch embedding jobs.  
   - *Deep:* Prioritized by query frequency.

5. **What data do you exclude entirely?**  
   - *Short:* Unsafe or low-quality domains.  
   - *Deep:* Especially sensitive or misleading content.
---

## 5. Common Embedding Space Alignment

- **Interviewer:** How do you train one embedding space across so many domains?

- **Candidate (Short Answer):**  
  By mixing multiple supervision signals during training.

- **Candidate (Deep Dive):**  
  Training uses:
  - Image–text pairs (web pages, products, captions)
  - Image–image similarity (duplicates, augmentations)
  - Cross-domain batch mixing

  The goal is not explicit category separation, but semantic alignment. Food, landmarks, and products emerge naturally when trained at scale.
### Staff-Level Follow-ups (Alignment)

1. **What if one domain regresses another?**  
   - *Short:* Multi-objective trade-offs.  
   - *Deep:* Domain-specific eval gates before rollout.

2. **How do you detect spurious correlations?**  
   - *Short:* Counterfactual tests.  
   - *Deep:* Remove cues and observe embedding shifts.

3. **What alignment failures are silent?**  
   - *Short:* Long-tail degradation.  
   - *Deep:* Rare queries lose recall unnoticed.

4. **How do you align video and images?**  
   - *Short:* Shared contrastive training.  
   - *Deep:* Frame-caption supervision.

5. **What happens if alignment drifts slowly?**  
   - *Short:* Gradual quality erosion.  
   - *Deep:* Requires trend-based monitoring, not alerts.

---

## 6. Data Ingestion & Training Pipeline

- **Interviewer:** How do you gather training data and keep it updated?

- **Candidate (Short Answer):**  
  Periodic full retraining with incremental updates in between.

- **Candidate (Deep Dive):**  
  Full retraining happens weekly or biweekly to refresh representations. New images are embedded daily or near-real-time using the latest stable model and added incrementally to the index. This avoids blocking ingestion while maintaining consistency.
### Staff-Level Follow-ups (Pipelines)

1. **What if retraining hurts quality?**  
   - *Short:* Rollback.  
   - *Deep:* Parallel indexes + shadow evaluation.

2. **How incremental is incremental?**  
   - *Short:* Daily ingestion.  
   - *Deep:* Full retrain still needed for global alignment.

3. **How do you validate new data?**  
   - *Short:* Sampling and checks.  
   - *Deep:* Detect label leakage or noise spikes.

4. **What’s the most fragile stage?**  
   - *Short:* Data collection.  
   - *Deep:* Garbage in silently poisons embeddings.

5. **How do you handle schema changes?**  
   - *Short:* Versioned pipelines.  
   - *Deep:* Backward compatibility enforced.
---

## 7. Large-Scale Retrieval

- **Interviewer:** How do you retrieve similar images at billion scale?

- **Candidate (Short Answer):**  
  Approximate nearest neighbor search over sharded vector indexes.

- **Candidate (Deep Dive):**  
  Exact search is infeasible under latency constraints. ANN methods trade a small amount of recall for massive speedups. Indexes are sharded and replicated across zones to balance latency and availability.
### Staff-Level Follow-ups (Retrieval)

1. **What if a shard goes down?**  
   - *Short:* Route around it.  
   - *Deep:* Accept partial recall over failure.

2. **How do you debug ANN recall drops?**  
   - *Short:* Offline probes.  
   - *Deep:* Fixed query canaries.

3. **What about hot shards?**  
   - *Short:* Replication.  
   - *Deep:* Load-aware routing.

4. **Would you ever return fewer results?**  
   - *Short:* Yes, under latency pressure.  
   - *Deep:* Better partial answers than timeout.

5. **When does exact search make sense?**  
   - *Short:* Never at this scale.  
   - *Deep:* Only for offline eval.
---

## 8. Storage, Infrastructure & Cost

- **Interviewer:** Give me a ballpark estimate of storage and infra.

- **Candidate (Short Answer):**  
  ~6–8 TB including indexing overhead and replication.

- **Candidate (Deep Dive):**  
  - 1B embeddings × 512 dims × 4 bytes ≈ 2 TB raw  
  - Index overhead + replication increases this to ~6–8 TB  
  Major costs come from memory-heavy indexes and offline embedding computation. Quantization can reduce memory at the cost of recall.
### Staff-Level Follow-ups (Cost)

1. **Biggest cost driver?**  
   - *Short:* Memory.  
   - *Deep:* ANN indexes dominate infra cost.

2. **Where do you compress first?**  
   - *Short:* Embeddings.  
   - *Deep:* Quantization before reducing dimensions.

3. **What if costs double?**  
   - *Short:* Reduce recall slightly.  
   - *Deep:* Tune ANN params.

4. **What doesn’t scale linearly?**  
   - *Short:* Index rebuilds.  
   - *Deep:* Operational overhead explodes.

5. **What’s cheap but impactful?**  
   - *Short:* Smarter caching.  
   - *Deep:* Popular landmarks amortize cost.

---

## 9. Ranking & Metadata (Late Fusion)

- **Interviewer:** Is nearest neighbor retrieval enough?

- **Candidate (Short Answer):**  
  Not always — ranking refines results.

- **Candidate (Deep Dive):**  
  Ranking uses embedding similarity plus metadata constraints like category compatibility or freshness. Metadata is not embedded jointly; it is applied as features at ranking time to avoid polluting the embedding space.
### Staff-Level Follow-ups (Ranking)

1. **Why not embed metadata?**  
   - *Short:* Noise.  
   - *Deep:* Pollutes representation space.

2. **What if metadata conflicts with visual signal?**  
   - *Short:* Visual wins.  
   - *Deep:* Metadata is secondary.

3. **How complex should ranking be?**  
   - *Short:* Very light.  
   - *Deep:* Latency budget is tight.

4. **What about freshness bias?**  
   - *Short:* Optional constraint.  
   - *Deep:* Domain-dependent.

5. **When does ranking fail?**  
   - *Short:* Poor candidates.  
   - *Deep:* Garbage in, garbage out.

---

## 10. Video & Social Content Retrieval

- **Interviewer:** Why does Lens sometimes return Instagram or video content?

- **Candidate (Short Answer):**  
  Because video frames are embedded like images.

- **Candidate (Deep Dive):**  
  Videos are decomposed into keyframes. Each frame is embedded into the same space. If a query image matches a frame closely, the system retrieves the associated video. This is image-to-image retrieval with videos attached downstream.
### Staff-Level Follow-ups (Video)

1. **How many frames per video?**  
   - *Short:* Keyframes only.  
   - *Deep:* Trade coverage vs cost.

2. **What about temporal context?**  
   - *Short:* Ignored initially.  
   - *Deep:* Added later if needed.

3. **How do you avoid spammy videos?**  
   - *Short:* Quality filters.  
   - *Deep:* Domain trust scores.

4. **Do videos dominate results?**  
   - *Short:* They shouldn’t.  
   - *Deep:* Balanced at ranking.

5. **What’s the hardest part?**  
   - *Short:* Scale.  
   - *Deep:* Explosion of frames.
---

## 11. Camera-Based Search Constraints

- **Interviewer:** How does camera-based input change things?

- **Candidate (Short Answer):**  
  Latency becomes the dominant constraint.

- **Candidate (Deep Dive):**  
  Online models must be smaller, retrieval fanout lower, and caching more aggressive. Early exits and approximate results are acceptable to meet strict p95 budgets.
### Staff-Level Follow-ups (Camera)

1. **What adds hidden latency?**  
   - *Short:* Preprocessing.  
   - *Deep:* Capture + upload overhead.

2. **What if network is slow?**  
   - *Short:* Degrade gracefully.  
   - *Deep:* Lower fanout, cache.

3. **Do you ever run on-device?**  
   - *Short:* Possibly embeddings.  
   - *Deep:* Hybrid designs.

4. **What’s the worst case?**  
   - *Short:* Motion blur.  
   - *Deep:* Unreliable embeddings.

5. **How do you test this?**  
   - *Short:* Synthetic noise.  
   - *Deep:* Real camera logs.
---

## 12. Sensitive & Medical Queries

- **Interviewer:** What about skin conditions or medical queries?

- **Candidate (Short Answer):**  
  These require restricted handling.

- **Candidate (Deep Dive):**  
  The system detects sensitive intent and routes queries to a conservative pipeline that avoids diagnosis, prioritizes authoritative content, and includes disclaimers. The embedding backbone may be shared, but ranking and response logic are domain-restricted.
### Staff-Level Follow-ups (Safety)

1. **How do you detect sensitive intent?**  
   - *Short:* Classifier.  
   - *Deep:* Conservative thresholds.

2. **What if classifier is wrong?**  
   - *Short:* Fail safe.  
   - *Deep:* Prefer under-answering.

3. **Do you ever show nothing?**  
   - *Short:* Yes.  
   - *Deep:* Abstention is acceptable.

4. **Who owns this policy?**  
   - *Short:* Cross-functional.  
   - *Deep:* Legal + policy teams.

5. **What’s the biggest risk?**  
   - *Short:* False authority.  
   - *Deep:* User harm.

---

## 13. Failure Modes & Monitoring

- **Interviewer:** What can go wrong?

- **Candidate (Short Answer):**  
  Drift, stale indexes, and recall degradation.

- **Candidate (Deep Dive):**  
  Visual distributions evolve. Retraining can shift embeddings. Index freshness and compression can silently hurt recall. Monitoring includes offline recall metrics, embedding stability checks, and periodic human evaluation.
### Staff-Level Follow-ups (Ops)

1. **What’s the hardest failure to detect?**  
   - *Short:* Slow drift.  
   - *Deep:* Metrics look stable.

2. **What do you alert on?**  
   - *Short:* Recall drops.  
   - *Deep:* Tail latency anomalies.

3. **How do you debug “feels worse”?**  
   - *Short:* Query sampling.  
   - *Deep:* Before/after embedding diffs.

4. **What telemetry is critical?**  
   - *Short:* Latency, recall.  
   - *Deep:* Per-domain breakdowns.

5. **What’s manual vs automated?**  
   - *Short:* Both.  
   - *Deep:* Humans catch nuance.
---

## 14. System Evolution

- **Interviewer:** How does this system evolve over time?

- **Candidate (Short Answer):**  
  Through strict versioning and gradual rollouts.

- **Candidate (Deep Dive):**  
  Models, embeddings, and indexes are versioned independently. New versions are rolled out via parallel indexes and controlled traffic shifts to minimize risk.
### Staff-Level Follow-ups (Long Term)

1. **What assumption breaks first?**  
   - *Short:* Data distribution.  
   - *Deep:* Visual trends evolve.

2. **What tech debt grows fastest?**  
   - *Short:* Index versions.  
   - *Deep:* Compatibility matrix explodes.

3. **How do you add new modality?**  
   - *Short:* Shared backbone.  
   - *Deep:* Separate heads.

4. **What would you redesign today?**  
   - *Short:* Simpler pipelines.  
   - *Deep:* Fewer moving parts.

5. **What defines success long-term?**  
   - *Short:* Stability.  
   - *Deep:* Predictable evolution.
---

## End of Interview

- **Interviewer:** Anything else to add?

- **Candidate:**  
  The core challenge is balancing representation quality, latency, and cost while keeping the system evolvable.


## <a name="gemini-qas"></a>GEMINI Based QAs
## 🏗️ Part I: Scoping & Architecture (The Setup)

---

### 1️⃣ "Back of Napkin" Resource Estimation

> **Scenario:** "Design a Visual Search engine for 1 Billion images. You have a budget. Latency must be <200ms."

**Staff-Level Answer:**
- **Constraint Calculation:**  
    - 1B images × 512-dim (float32) ≈ 2TB of raw vectors (too large for pure RAM)
- **Solution:**  
    - Use **Product Quantization (PQ)** to compress vectors to 64 bytes (32x compression), dropping index to ~64GB.
- **Infrastructure:**  
    - Fits on high-memory instances (e.g., AWS r5.24xlarge).
    - Enables index replication (for availability) instead of strict sharding for capacity.
    - This decision sets the hardware strategy.

---

### 2️⃣ "Cold Start" for Infrastructure

> **Scenario:** "How do we handle the initial ingestion of 1 billion images? We can't just run a Python script."

**Staff-Level Answer:**
- **Distributed Processing:**  
    - Use a Spark/Ray cluster.
    - "Image Factory" pipeline for fetching, resizing, inference.
- **Cost Check:**  
    - Use Spot Instances (stateless batch jobs) to save ~60% compute cost.
- **Prioritization:**  
    - Index the "Head" (popular products) first to unblock product launch; backfill the "Long Tail" later.

---

### 3️⃣ Latency vs. Recall Trade-offs

> **Scenario:** "Your P99 latency is spiking to 500ms. We need it under 200ms. What do you cut?"

**Staff-Level Answer:**
- **Diagnosis:**  
    - Profile system: Is Model (inference) or Index (search) responsible?
- **Kill Chain:**  
    - ✂️ **Cut the Re-Ranker:** Replace heavy cross-encoder with dot-product MLP or remove stage-2 completely.
    - 🧱 **Quantization:** Increase compression (more aggressive PQ) for modest recall loss.
    - 🎚️ **Search Parameters:** Lower `ef_search` (beam size in HNSW) to visit fewer nodes.

---

## 🧑‍💻 Part II: Data Strategy & Representation (The Model)

---

### 4️⃣ The "Universal" Dataset Problem

> **Scenario:** "Our click data is 80% fashion, but we need to support landmarks and food. The model fails on non-fashion items."

**Staff-Level Answer:**
- **Strategy:**  
    - Implement **Stratified Sampling**: Ensure training batches contain 20% Fashion, 20% Food, 20% Landmarks.
- **External Data:**  
    - Ingest public datasets (e.g., Google Landmarks v2, Wikipedia) when internal data is scarce.
- **Hard Negatives:**  
    - Generate hard negatives (same object, different angle/lighting) to enforce fine-grained features.

---

### 5️⃣ Handling Noisy Training Data

> **Scenario:** "We use user clicks as training labels, but users click randomly. The data is noisy."

**Staff-Level Answer:**
- **Proxy Signals:**  
    - Define "High-Confidence" interactions.
    - Filter out clicks with dwell time < 5s.
    - Prioritize "Add to Cart" events over simple clicks.
- **Visual Verification:**  
    - Use a teacher model to pre-filter training pairs.
    - If visual distance between Query/Clicked-Item is large, discard as anomaly/mis-click.

---

### 6️⃣ Fine-Grained vs. Semantic Retrieval

> **Scenario:**  
> - User scans Harry Potter book but sees movie DVD (wrong).  
> - User scans grey shirt and sees another grey shirt (acceptable).  
> - How to handle both?

**Staff-Level Answer:**
- **Hierarchical Embeddings:**
    - **Head A (Semantic):** Softmax on Category (clusters all grey shirts)
    - **Head B (Instance):** ArcFace on SKU ID (distinguishes Book vs DVD)
- **Inference:**  
    - Use Category head for coarse filtering.  
    - Use Instance head for fine ranking.

---

### 7️⃣ Distillation for Inference Speed

> **Scenario:** "ViT-Large model = best accuracy, but 100ms latency. Too slow."

**Staff-Level Answer:**
- **Knowledge Distillation:**  
    - Train ViT-L offline (Teacher)
    - Distill into ResNet-50 or MobileNet (Student) for online inference
    - Achieve <20ms latency while retaining transformer's understanding

---

### 8️⃣ Domain Gap (Studio vs. Real World)

> **Scenario:** "Model works on catalog images but fails on user-uploaded phone photos."

**Staff-Level Answer:**
- **Data Augmentation:**  
    - Use noise, blur, rotation, color jitter to simulate "bad" user photos during training.
- **Query Expansion:**  
    - Average query vector with vectors of visually similar catalog items (from session history) to denoise intent.

---

### 9️⃣ Handling "Nothing Matches"

> **Scenario:** "User scans their cat. We sell furniture. The system returns a fur coat. User is offended."

**Staff-Level Answer:**
- **OOD Detection:**  
    - Set a "Not Found" threshold.
- **Distance Threshold:**  
    - If nearest neighbor distance > $X$, return "No visual matches found".
- **Category Classification:**  
    - Run a cheap classifier (e.g. "Is this furniture?"). If no, abort search immediately.

---

## ⚙️ Part III: Indexing & Retrieval (The Engine)

---

### 🔟 Indexing Strategy (HNSW vs. IVF)

> **Scenario:** "HNSW is fast but RAM-heavy; IVF is memory-efficient but slower. What to use for 1B images?"

**Staff-Level Answer:**
- **Hybrid IVFPQ + HNSW:**
    - Cluster space into 100K centroids (IVF)
    - Build small HNSW graph on centroids
    - Search: traverse HNSW to best centroid → scan PQ codes within
- **Why:**  
    - Balance memory efficiency (IVF) with navigation speed (HNSW)

---

### 1️⃣1️⃣ Sharding Strategy

> **Scenario:** "How to shard 1B vectors across 50 nodes? By Category or Randomly?"

**Staff-Level Answer:**
- **Random sharding (by Hash(ID)):** Prevents hot spots (e.g., "Fashion" node doesn't crash on Black Friday)
- **Trade-off:**  
    - Requires scatter-gather (query all 50 nodes).
    - Acceptable for load-balancing.

---

### 1️⃣2️⃣ Real-Time Indexing ("Backfill" Problem)

> **Scenario:** "A celebrity wears a new sneaker, but it takes 24h to appear in search. Fix it."

**Staff-Level Answer:**
- **Lambda Architecture:**
    - **Main Index (static):** 99.9% of data, rebuilt weekly/monthly.
    - **Delta Index (dynamic):** In-memory HNSW for new uploads.
    - **Query:** Search both → merge results.
    - **Compaction:** Nightly merge Delta into Main.

---

### 1️⃣3️⃣ Index Updates & Deletions

> **Scenario:** "Item goes out of stock or is legally taken down. How to instantly remove from HNSW graph?"

**Staff-Level Answer:**
- **Soft Deletion:** Don't rebuild graph.
- **Bitmasking:** Parallel bit-array (0=Active, 1=Deleted)
- **Filter:**  
    - During traversal, check bitmask.  
    - If "Deleted", skip & continue to neighbors.

---

### 1️⃣4️⃣ Handling Duplicates (De-duping)

> **Scenario:** "Index is bloated with 50 versions of the same stock photo. Wastes RAM, ruins diversity."

**Staff-Level Answer:**
- **Pre-index De-duping:** Use perceptual hash (SimHash, pHash) to detect near-duplicates.
- **Policy:** Only index canonical version (e.g., highest res).  
    - Map duplicates to canonical ID in key-value store.

---

## 🧠 Part IV: Ranking & Business Logic (The Intelligence)

---

### 1️⃣5️⃣ The Profitability Trade-off

> **Scenario:** "Best visual match is out of stock. Second best is in stock and high margin. What do we show?"

**Staff-Level Answer:**
- **Stage-2 Ranker (MLP/LightGBM):**
    - Inputs: Visual score, Stock status, Margin, CTR.
    - Logic: Model optimizes for Expected Value (P(Conversion) × Price).
- **Hard Filter:**  
    - Hard-filter archived/dead items at the index so they never reach ranker.

---

### 1️⃣6️⃣ Multi-modal Fusion (Text + Image)

> **Scenario:** "User uploads a dress but types 'in red'. The dress is blue."

**Staff-Level Answer:**
- **Late Fusion:** Retrieve by image, then text-filter or re-score results by "red" query.
- **Early Fusion (advanced):** Use model like CLIP-Text to create a "modifier vector" and add to image vector before retrieval.

---

### 1️⃣7️⃣ Diversity Re-ranking

> **Scenario:** "User searches for 'patterned rug'. We return 10 identical-looking rugs. Looks like a bug."

**Staff-Level Answer:**
- **MMR (Maximal Marginal Relevance):**
    - Pick best match first.
    - For each next slot, pick item similar to query **and** dissimilar to those already picked.
- **Clustering:**  
    - Group top 100 by visual clusters; pick a representative from each.

---

### 1️⃣8️⃣ Geometric Verification (Staff Differentiator)

> **Scenario:** "Good retrieval, but precision is low (e.g., 'striped shirts' → 'striped pants')."

**Staff-Level Answer:**
- **Stage-3 Geometric Re-ranking:**  
    - Use local features (DELF/SuperPoint) + RANSAC.
    - Actually check: "Does spatial layout in query match candidate?"
    - Run only on top-10 candidates (expensive).

---

## 🛠️ Part V: Production & Operations (The War Room)

---

### 1️⃣9️⃣ Debugging "The CEO's Query"

> **Scenario:** "CEO searches for a shoe and gets a fridge. Fix it."

**Staff-Level Answer:**
- **Debugging Pipeline:**
    - 🔍 Check cropping: Did detector crop shoe or floor? (Visualize Grad-CAM)
    - 📈 Check embedding:  
        - Is vector distance close? (Model failure)  
        - Or did index miss it? (Search failure)
    - 🦠 Root cause: If vector close, model is overfitting to background. Retrain with background augmentation.

---

### 2️⃣0️⃣ Seasonal Drift

> **Scenario:** "Trained in July, now December. Model drops on winter coats."

**Staff-Level Answer:**
- **Continual Learning:** Model hasn't seen new trends.
- **Fine-tuning:** Periodically fine-tune on last 30 days of data.
- **Drift Monitoring:** Track embedding norms and cluster centers. If "Winter Coats" cluster moves, trigger retrain.

---

### 2️⃣1️⃣ A/B Testing & Cannibalization

> **Scenario:** "How to test new embedding model without risking revenue?"

**Staff-Level Answer:**
- **Interleaving:** Show Model A & Model B results in mixed list (e.g., positions 1,3,5 vs 2,4,6).
- **Metric:** Compare click rates per slot (reduces UI/time-of-day bias).

---

### 2️⃣2️⃣ Capacity Planning

> **Scenario:** "Traffic doubling soon. What breaks first?"

**Staff-Level Answer:**
- **Bottleneck Analysis:**
    - Network: Image thumbnail bandwidth?
    - CPU: Re-ranking usually CPU bound.
    - Memory: Does index fit in RAM?
- **Plan:**  
    - If RAM-limited, increase PQ compression (short term) or add nodes (long term).

---

### 2️⃣3️⃣ Dealing with Adversarial Attacks

> **Scenario:** "Competitors scraping us; users upload adversarial noise."

**Staff-Level Answer:**
- 🚦 **Rate Limiting:** Standard IP limits.
- 🛡️ **Adversarial Training:** Add adversarial noise to model's training.
- 🔒 **Hash Blocking:** Hash and instantly block malicious images.

---

### 2️⃣4️⃣ Multi-Tenant Isolation

> **Scenario:** "Want to sell B2B. Ensure Tenant A doesn't see Tenant B's images."

**Staff-Level Answer:**
- **Physical Partitioning:** Separate indexes per tenant is safest.
- **Logical Partitioning:**  
    - Attach metadata `tenant_id` to each vector.
    - HNSW must strictly filter by `tenant_id` during search.

---

### 2️⃣5️⃣ "Cost Cut" Mandate

> **Scenario:** "AWS bill is $10M/year. CFO: cut by 30%. Can't reduce item count."

**Staff-Level Answer:**
- **Switch to Spot Instances:** For all stateless (Inference/Rendering) nodes.
- **Aggressive PQ:** 64-byte → 32-byte vectors (halve RAM cost).
- **Tiered Storage:**  
    - Keep "Head" (popular items) in RAM.
    - Move "Tail" (rarely accessed items) to SSD-based indices (like DiskANN).  
    - Latency increases for rare items, but cost drops by 10x.

---
