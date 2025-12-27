# 🛡️ TensorGuard SDK (v1.1.0)

[![Production Ready](https://img.shields.io/badge/status-production--ready-green.svg)](https://github.com/Danielfoojunwei/TensorGuard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TensorGuard** is a professional-grade SDK for **Privacy-Preserving VLA (Vision-Language-Action) Fine-Tuning** in humanoid robotics. It leverages vectorized N2HE (post-quantum) homomorphic encryption and differential privacy to enable secure fleet learning.

> "Securing the future of humanoid intelligence, 128 bits at a time."

---

## 🏗️ Technical Architecture

TensorGuard is built on a modular SaaS architecture designed for scalability and high-performance cryptographic operations:

| Component | Description |
|-----------|-------------|
| **Core** | N2HE Vectorized LWE Encryption, Semantic Sparsification, and Neural Compression. |
| **Server** | Secure Homomorphic Aggregator and Real-time Developer Dashboard. |
| **API** | Standardized Pydantic schemas for cross-platform robotic telemetry. |
| **Utils** | Structured JSON logging and Pydantic-based configuration management. |

---

## 🏭 Production-Grade Features (v1.1.0)

TensorGuard is **production-ready** for secure post-training at scale with enterprise features:

### Operating Envelope
*   **Hard constraints** on trainable parameters (PEFT-only: LoRA/Adapters)
*   **Enforced update size limits** (10KB - 5MB) and round cadence (10min - 24h)
*   **Canary deployment** with automatic rollback (configurable 5-20% canary)

### Canonical UpdatePackage Format
*   **Versioned updates** with full metadata (training config, safety stats, compatibility)
*   **Deterministic serialization** for reproducible audits
*   **Fingerprinting** for safe rollback and integrity verification

### Privacy & Training Controls
*   **Separate DP policy profiles** per customer site (epsilon budget, clipping, noise)
*   **Independent encryption profiles** (key rotation schedule, quorum thresholds)
*   **Training policy profiles** (compression, sparsity, quality thresholds)

### Enterprise Key Management
*   **Cloud never holds decryption keys** (customer-controlled KMS/HSM)
*   **Automatic key rotation** with full audit trail
*   **Break-glass policies** for compromised keys
*   **Disaster recovery** export/import

### Resilient Aggregation
*   **Quorum-based rounds** (min 2-5 clients) with straggler handling
*   **Staleness weighting** (exponential decay for old updates)
*   **Client health tracking** and outlier detection (>3σ rejection)

### Evaluation Gating
*   **Safety checks** before every deployment (success rate, KL divergence, OOD robustness)
*   **Regression detection** (max 3-5% degradation allowed)
*   **Canary → Progressive → Full rollout** with automatic rollback

### SRE Observability
*   **Full latency breakdown** (train/compress/encrypt/upload/aggregate/decrypt/apply)
*   **Compression metrics** (original size, effective ratio, overhead)
*   **Privacy budget tracking** (epsilon consumption rate, remaining budget)
*   **JSONL metrics** for Prometheus/Grafana/DataDog ingestion

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -e .
# or
pip install tensorguard
```

### 2. Configuration
TensorGuard uses Pydantic Settings for zero-config defaults, but supports high-granularity environment overrides:
```powershell
$env:TENSORGUARD_LOG_LEVEL = "DEBUG"
$env:TENSORGUARD_SECURITY_LEVEL = 128
```

### 3. Usage (CLI)
```bash
# Start the security aggregator (Cloud/Local)
tensorguard server --port 8080

# Launch the Developer Showcase Dashboard
tensorguard dashboard
```

---

## 🔒 Security Specifications

- **Encryption**: Vectorized N2HE (Lattice-based LWE) with 128/192-bit security.
- **Privacy**: Epsilon-bounded Differential Privacy with task-aware semantic sparsification.
- **Compression**: APHE (Adaptive Perceptual Homomorphic Encoding) quantization.
- **Gating**: MoI (Mixture of Intelligence) prioritized gradient aggregation.

---

## 🧬 Project Structure

```text
src/tensorguard/
├── api/             # Data schemas (Pydantic/Dataclasses)
├── core/            # Cryptography, Adapters, Privacy Pipeline, Production
├── server/          # Aggregator Strategy, Dashboard API
├── utils/           # Config, Logging, Exceptions
└── cli.py           # Unified entry point
```

---

## 🛠️ Development & Verification

TensorGuard maintains a rigorous 100% pass verification suite:
```bash
$env:PYTHONPATH="src"; python -m pytest tests/
```

Created by **HintSight Technology**.
For inquiries reach out to @danielfoojunwei (Telegram), visit (https://www.hintsighttechnologies.com) for more information on our background. 
