# TensorGuard Deployment Guide v2.0.0 (FedMoE Paradigm)
## Complete Step-by-Step for Production Deployments

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Step-by-Step User Flow](#2-step-by-step-user-flow)
3. [Deployment Scenarios](#3-deployment-scenarios)
4. [Production Blueprint](#4-production-blueprint)
5. [SDK Reference](#5-sdk-reference)
6. [Hardware Requirements](#6-hardware-requirements)
7. [FAQ](#7-faq)

---

## 1. Getting Started

### Prerequisites
- Python 3.10+
- 8GB+ RAM
- CUDA-capable GPU (optional, for acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/tensorguard.git
cd tensorguard

# Install SDK
pip install -e .

# Verify installation
tensorguard info
```

### Generate Your First Key

```python
from tensorguard.core.crypto import generate_key

# Generate 128-bit post-quantum secure key
generate_key("keys/my_enterprise_key.npy", security_level=128)
```

---

## 2. Step-by-Step User Flow

### Phase 1: Initial Setup (Day 1)

**Step 1: Generate Enterprise Key**
```python
from tensorguard.core.crypto import generate_key

# One-time key generation for your fleet
generate_key("keys/warehouse_fleet_v1.npy", security_level=128)
```

**Step 2: Register Key with KMS**
```python
from tensorguard.core.production import KeyManagementSystem, KeyMetadata
from datetime import datetime
from pathlib import Path

kms = KeyManagementSystem(audit_log_path=Path("logs/key_audit.log"))
kms.register_key("warehouse_v1", KeyMetadata(
    key_id="warehouse_v1",
    created_at=datetime.utcnow(),
    owner="GlobalLogistics",
    security_level=128
))
```

**Step 3: Configure Operating Envelope**
```python
from tensorguard.core.production import OperatingEnvelope

envelope = OperatingEnvelope(
    trainable_modules=["policy_head", "vision_adapter"],
    max_trainable_params=5_000_000,  # 5M params max
    round_interval_seconds=3600,      # Hourly updates
    target_update_size_kb=500         # 500KB target
)
```

---

### Phase 2: Edge Deployment (Per Robot)

**Step 4: Initialize Secured Client**
```python
from tensorguard.core.client import create_client
from tensorguard.core.adapters import MoEAdapter

client = create_client(
    security_level=128,
    cid="robot_warehouse_01",
    key_path="keys/warehouse_fleet_v1.npy"
)

# Attach your task-aware FedMoE adapter (v2.0)
client.set_adapter(MoEAdapter())
```

**Step 5: Buffer Training Demonstrations**
```python
from tensorguard.api.schemas import Demonstration
import numpy as np

# As robot operates, buffer successful episodes
demo = Demonstration(
    observations=[np.zeros((224, 224, 3))],
    actions=[np.zeros(7)]
)
client.add_demonstration(demo)
```

**Step 6: Trigger Secure Update**
```python
# After enough demos (e.g., end of shift)
update_package = client.process_round()

# Check update was created
print(f"Package size: {len(update_package.serialize()) / 1024:.1f} KB")
print(f"Privacy spent: ε={update_package.safety_stats.dp_epsilon_consumed:.4f}")
```

---

### Phase 3: Server Aggregation

**Step 7: Run Aggregation Server**
```bash
tensorguard server --port 8080
```

**Step 8: Monitor Dashboard**
```bash
tensorguard dashboard --port 8000
# Open http://localhost:8000 in browser
```

---

### Phase 4: Model Deployment

**Step 9: Pull Aggregated Model**
```python
# After aggregation round completes
from tensorguard.server.aggregator import ExpertDrivenStrategy
strategy = ExpertDrivenStrategy(quorum_threshold=2)
aggregated_params, metrics = strategy.aggregate_fit(1, results, [])
```

**Step 10: Evaluate Before Deployment**
```python
from tensorguard.core.production import EvaluationGate, SafetyThresholds

gate = EvaluationGate(SafetyThresholds(
    min_success_rate=0.85,
    max_kl_divergence=0.5
))

# Only deploy if passes safety checks
passed, failures = gate.evaluate(new_metrics)
if passed:
    deploy_to_fleet(aggregated_params)
```

---

## 3. Deployment Scenarios

### Scenario A: Manufacturing Floor (High Security)

| Parameter | Value |
|-----------|-------|
| **Environment** | Air-gapped network |
| **Robots** | 20x Humanoids |
| **Key Storage** | HSM-backed |
| **Aggregation** | On-premise |

```python
config = ShieldConfig(
    model_type="pi0",
    key_path="hsm://slot/1",
    security_level=192,  # Maximum
    compression_ratio=32
)
```

### Scenario B: Warehouse Fleet (Standard)

| Parameter | Value |
|-----------|-------|
| **Environment** | VPN to cloud |
| **Robots** | 100x Mobile manipulators |
| **Key Storage** | Software keys |
| **Aggregation** | Cloud-managed |

```python
config = ShieldConfig(
    model_type="openvla",
    key_path="/etc/tensorguard/warehouse.pem",
    security_level=128,
    compression_ratio=64  # Higher for scale
)
```

### Scenario C: Healthcare (HIPAA Compliant)

| Parameter | Value |
|-----------|-------|
| **Environment** | Certified cloud region |
| **Robots** | 5x Assistive robots |
| **Privacy** | DP ε=0.5 (strict) |
| **Audit** | Full compliance logging |

```python
config = ShieldConfig(
    model_type="custom",
    key_path="/secure/hipaa_key.pem",
    dp_epsilon=0.5,  # Stricter privacy
    security_level=128
)
```

---

## 4. Production Blueprint

### Operating Envelope

The envelope enforces hard constraints to prevent drift:

```python
envelope = OperatingEnvelope(
    peft_strategy=PEFTStrategy.LORA,
    trainable_modules=["policy_head", "last_4_blocks"],
    max_trainable_params=10_000_000,
    round_interval_seconds=3600,
    target_update_size_kb=500,
    enable_canary=True,
    canary_percentage=0.1
)
```

### Privacy Controls

Separate knobs for security and ML teams:

```python
# DP Profile (Security Team)
dp_profile = DPPolicyProfile(
    clipping_norm=1.0,
    epsilon_budget=10.0,
    hard_stop_enabled=True
)

# Training Profile (ML Team)
training_profile = TrainingPolicyProfile(
    compression_ratio=32,
    sparsity=0.01,
    max_quality_mse=0.05
)
```

### Resilient Aggregation

Handle real-world failures:

```python
from tensorguard.server.aggregator import ExpertDrivenStrategy

strategy = ExpertDrivenStrategy(
    quorum_threshold=2,
    enable_bayesian_eval=True
)
```

---

## 5. SDK Reference

### Core Classes

| Class | Purpose |
|-------|---------|
| `EdgeClient` | Main client for edge robots |
| `TensorGuardStrategy` | Flower-compatible aggregation strategy |
| `N2HEEncryptor` | Homomorphic encryption engine |
| `UpdatePackage` | Wire format for encrypted updates |
| `KeyManagementSystem` | Enterprise key lifecycle |
| `EvaluationGate` | Safety checks before deployment |

### Configuration

```python
@dataclass
class ShieldConfig:
    model_type: str = "pi0"
    key_path: str = "keys/node_key.pem"
    security_level: int = 128
    compression_ratio: int = 32
    sparsity: float = 0.01
    dp_epsilon: float = 1.0
    # v2.0: Skellam DP noise + Expert Gating active
```

### Performance Specs

| Operation | Jetson Orin NX | Jetson AGX Orin |
|-----------|----------------|-----------------|
| Expert Gradient Computation | ~240ms | ~120ms |
| Skellam-N2HE Encryption | ~80ms | ~35ms |
| Upload (Compressed UpdatePackage) | ~2KB/demo | ~2KB/demo |

---

## 6. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Edge Compute | Jetson Orin Nano 8GB | Jetson AGX Orin 64GB |
| Storage | 32GB NVMe | 256GB NVMe |
| Network | 100Mbps | 1Gbps |
| RAM | 8GB | 32GB+ |

---

## 7. FAQ

### General

**Q: Can I use TensorGuard without N2HE encryption?**
A: Yes. Set `security_level=0` in config to use DP-only mode for trusted networks.

**Q: What VLA models are supported?**
A: OpenVLA, Pi0, RT-2, and any custom model with a gradient function.

### Security

**Q: Where are keys stored?**
A: Keys are stored locally on your hardware. TensorGuard never transmits private keys.

**Q: What happens if a robot is stolen?**
A: Use `KeyManagementSystem.revoke_key()` to immediately invalidate that robot's contributions.

### Performance

**Q: How much bandwidth does it save?**
A: Semantic sparsification + compression achieves **50x reduction** (15MB → 300KB).

**Q: Does encryption slow inference?**
A: No. TensorGuard only operates during training. Inference runs at full speed.

---

## Production Checklist

Before deploying TensorGuard in production:

- [ ] Generate enterprise key with `generate_key()`
- [ ] Register key with `KeyManagementSystem`
- [ ] Configure `OperatingEnvelope` with hard limits
- [ ] Set up `DPPolicyProfile` with privacy budget
- [ ] Enable `EvaluationGate` with safety thresholds
- [ ] Configure observability (metrics → monitoring stack)
- [ ] Run benchmarks to validate SLAs
- [ ] Test key rotation procedure
- [ ] Document break-glass key revocation process

---

## Support

- **Documentation**: https://github.com/Danielfoojunwei/TensorGuard
- **Issues**: https://github.com/Danielfoojunwei/TensorGuard/issues

---

© 2025 TensorGuard Team | DTC @ NTU | HintSight Technology
