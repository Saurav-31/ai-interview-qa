# ML Model Serving at Scale

## Question
How do you design a system to serve ML models at scale? What are the key considerations?

## Answer

### Overview
ML model serving involves deploying trained models to production and handling inference requests efficiently, reliably, and at scale.

## Serving Patterns

### 1. Real-Time Serving (Online)
**Characteristics:**
- Low latency (< 100ms)
- Single predictions
- Synchronous requests

**Use Cases:** Fraud detection, recommendation systems, search ranking

### 2. Batch Serving (Offline)
**Characteristics:**
- High throughput
- Process large datasets
- Scheduled jobs

**Use Cases:** Email campaigns, analytics, report generation

### 3. Streaming Serving
**Characteristics:**
- Process data streams
- Near real-time
- Event-driven

**Use Cases:** Real-time monitoring, IoT, financial trading

## System Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/gRPC
       ↓
┌─────────────────┐
│  Load Balancer  │
└──────┬──────────┘
       │
       ↓
┌──────────────────────────────┐
│    API Gateway/Router        │
│  - Auth, Rate Limiting       │
│  - Request Validation        │
└──────┬───────────────────────┘
       │
       ↓
┌──────────────────────────────┐
│   Model Serving Service      │
│  ┌────────────────────────┐  │
│  │  Prediction Service    │  │
│  │  - Preprocessing       │  │
│  │  - Model Inference     │  │
│  │  - Postprocessing      │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │  Model Management      │  │
│  │  - Versioning          │  │
│  │  - A/B Testing         │  │
│  │  - Canary Deployment   │  │
│  └────────────────────────┘  │
└──────┬───────────────────────┘
       │
       ↓
┌──────────────────────────────┐
│    Monitoring & Logging      │
│  - Latency, Throughput       │
│  - Model Metrics             │
│  - Data Drift Detection      │
└──────────────────────────────┘
```

## Model Optimization Techniques

### 1. Quantization

Convert weights to lower precision:

```python
# PyTorch Dynamic Quantization
import torch

model_fp32 = torch.load('model.pth')

# Quantize to INT8
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)

# 4x smaller, 2-4x faster
```

**Types:**
- **Post-Training Quantization (PTQ):** After training
- **Quantization-Aware Training (QAT):** During training

**Trade-off:** Size/speed ↑, Accuracy ↓ (~1-2%)

### 2. Model Pruning

Remove unnecessary weights:

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights in linear layers
prune.l1_unstructured(module, name='weight', amount=0.3)

# Remove pruning reparameterization
prune.remove(module, 'weight')
```

**Result:** 30-50% reduction with minimal accuracy loss

### 3. Knowledge Distillation

Train small model (student) from large model (teacher):

```python
# Teacher predictions (soft targets)
teacher_logits = teacher_model(x)
teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

# Student predictions
student_logits = student_model(x)
student_probs = F.softmax(student_logits / temperature, dim=1)

# Distillation loss
distillation_loss = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=1),
    teacher_probs,
    reduction='batchmean'
) * (temperature ** 2)

# Total loss
loss = alpha * distillation_loss + (1 - alpha) * student_task_loss
```

**DistilBERT:** 40% smaller, 60% faster, 97% of BERT performance

### 4. Model Compilation

Optimize computation graph:

```python
# TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")

# ONNX Runtime
import onnx
torch.onnx.export(model, dummy_input, "model.onnx")

# TensorRT (NVIDIA GPUs)
import tensorrt as trt
# Convert ONNX → TensorRT engine
```

**Speedup:** 2-5x depending on hardware

## Serving Frameworks

| Framework | Strengths | Use Case |
|-----------|-----------|----------|
| **TensorFlow Serving** | Mature, gRPC/REST, versioning | TF models, production |
| **TorchServe** | Official PyTorch, easy setup | PyTorch models |
| **ONNX Runtime** | Cross-framework, optimized | Multi-framework |
| **Triton (NVIDIA)** | Multi-framework, GPU-optimized | High-performance GPU |
| **FastAPI + Custom** | Flexible, full control | Custom logic |
| **Ray Serve** | Distributed, scalable | Large-scale deployments |
| **BentoML** | Easy packaging, multi-framework | Rapid deployment |

### Example: TorchServe

```bash
# 1. Archive model
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --serialized-file resnet50.pth \
  --handler image_classifier

# 2. Start server
torchserve --start --model-store model_store --models resnet50=resnet50.mar

# 3. Query
curl -X POST http://localhost:8080/predictions/resnet50 -T image.jpg
```

### Example: FastAPI Custom Serving

```python
from fastapi import FastAPI
import torch

app = FastAPI()

# Load model once at startup
@app.on_event("startup")
async def load_model():
    global model
    model = torch.jit.load("model.pt")
    model.eval()

@app.post("/predict")
async def predict(data: dict):
    # Preprocess
    input_tensor = preprocess(data["input"])
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    result = postprocess(output)
    
    return {"prediction": result}

# Run: uvicorn main:app --workers 4
```

## Batching Strategies

### Dynamic Batching

Group requests to maximize throughput:

```python
class BatchPredictor:
    def __init__(self, max_batch_size=32, max_wait_time=0.01):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = []
    
    async def predict(self, input):
        # Add to queue
        future = asyncio.Future()
        self.queue.append((input, future))
        
        # Process batch if full or timeout
        if len(self.queue) >= self.max_batch_size:
            await self.process_batch()
        
        return await future
    
    async def process_batch(self):
        batch_inputs = [item[0] for item in self.queue]
        batch_tensor = torch.stack(batch_inputs)
        
        # Single inference call
        outputs = model(batch_tensor)
        
        # Return results
        for i, (_, future) in enumerate(self.queue):
            future.set_result(outputs[i])
        
        self.queue.clear()
```

**Benefit:** 5-10x higher throughput

## Caching

Cache predictions for common inputs:

```python
from functools import lru_cache
import hashlib

class CachedPredictor:
    def __init__(self, model):
        self.model = model
        self.cache = {}
    
    def predict(self, input):
        # Hash input
        input_hash = hashlib.md5(input.tobytes()).hexdigest()
        
        # Check cache
        if input_hash in self.cache:
            return self.cache[input_hash]
        
        # Compute prediction
        output = self.model(input)
        
        # Store in cache
        self.cache[input_hash] = output
        
        return output
```

**Use Redis** for distributed caching

## Monitoring

### Key Metrics

**System Metrics:**
- Latency (p50, p95, p99)
- Throughput (requests/sec)
- Error rate
- CPU/GPU utilization
- Memory usage

**Model Metrics:**
- Prediction distribution
- Confidence scores
- Model version performance

**Data Metrics:**
- Input distribution (drift detection)
- Feature statistics

### Implementation

```python
from prometheus_client import Counter, Histogram
import time

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
async def predict(data: dict):
    start = time.time()
    
    try:
        result = model.predict(data)
        prediction_counter.inc()
        return {"prediction": result}
    
    finally:
        latency = time.time() - start
        latency_histogram.observe(latency)
```

## Deployment Strategies

### 1. Blue-Green Deployment
- Two identical environments
- Switch traffic instantly
- Easy rollback

### 2. Canary Deployment
```
Old Model: 90% traffic
New Model: 10% traffic → Monitor → Gradually increase
```

### 3. A/B Testing
```
Model A: 50% of users
Model B: 50% of users
→ Compare metrics → Choose winner
```

### 4. Shadow Deployment
- New model runs in parallel
- Results logged but not returned
- Safe testing

## Scaling Strategies

### Horizontal Scaling
- Multiple replicas behind load balancer
- Kubernetes autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server
spec:
  scaleTargetRef:
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling
- More powerful instances
- Larger GPUs

### Model Parallelism
- Split model across devices (large models)

## Key Considerations

✅ **Latency Requirements:** Real-time vs batch  
✅ **Throughput:** Expected QPS  
✅ **Cost:** Compute resources  
✅ **Hardware:** CPU vs GPU vs TPU  
✅ **Model Size:** Optimization needs  
✅ **SLA:** Availability guarantees  
✅ **Security:** Auth, encryption  
✅ **Monitoring:** Observability  
✅ **Versioning:** Model lifecycle  
✅ **Fallback:** Error handling

## Key Takeaways

1. **Choose serving pattern:** Real-time, batch, or streaming
2. **Optimize models:** Quantization, pruning, distillation
3. **Use batching** for throughput
4. **Cache** common predictions
5. **Monitor** latency, throughput, model metrics
6. **Deploy gradually:** Canary, A/B testing
7. **Scale horizontally** with load balancers
8. **Choose right framework:** TorchServe, Triton, custom

## Tags
#SystemDesign #ModelServing #MLOps #Deployment #Scaling #Inference #Production

## Difficulty
Hard

## Related Questions
- How to monitor ML models in production?
- Explain model quantization techniques
- Design a recommendation system at scale
