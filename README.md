# APHE-Shield: Privacy-Preserving VLA Fine-Tuning
> **Production Ready (v1.1.0)** | **128-bit Post-Quantum Security** | **Research-Optimized**

**APHE-Shield** is a plug-and-play SDK that enables Robotics System Integrators to fine-tune Vision-Language-Action (VLA) models on humanoid robots deployed in secure customer facilities—**without ever exposing proprietary location data, operational workflows, or sensitive demonstrations to the cloud.**

---

## 🚀 Key Features

*   **🔒 N2HE Homomorphic Encryption**: 128-bit post-quantum secure encryption. cloud servers operate *only* on encrypted ciphertexts (`Enc(a) + Enc(b) = Enc(a+b)`), never seeing the raw gradients.
*   **🤖 Plug-and-Play Edge SDK**: Wraps your existing VLA training loop (PyTorch/JAX) with a single `EdgeClient` adapter.
*   **⚡ SIMD Slot Folding**: lattice-aligned tensor packing for high-throughput encrypted learning.
*   **🧠 Semantic Priority Sparsification**: Task-relevant gradient retention prioritizing **Attention layers** for higher accuracy at 99% compression.
*   **🤖 Mixture of Intelligence (MoI)**: Multi-expert gradient stream that independently weights Visual, Language, and Auxiliary updates.
*   **⚡ 1300x Performance Optimized**: Custom vectorization and serialization protocol (0.22s per 10KB chunk).
*   **♻️ Error Feedback**: Local residual memory ensures ignored sparse gradients are accumulated.
*   **📉 Adaptive Compression**: Dynamically adjusts quantization bits based on payload size.
*   **🔊 Noise-Aware Key Refresh**: Automatically rotates keys when encryption noise budget is exhausted.
*   **👁️ Signal Quality Monitor**: Real-time MSE tracking warns if compression degrades gradient quality.
*   **🛡️ Differential Privacy**: Built-in gradient clipping and noise injection ($ \epsilon \le 1.0 $).
*   **📊 Live Integrator Dashboard**: Monitor fleet learning progress in real-time with **MoI Insight Panel**.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph "Secure Customer Facility (Trusted)"
        R[🤖 Robot / VLA] -->|Compute Gradients| A[APHE Adapter]
        A -->|Clip & Noise| E[N2HE Encryptor]
        E -->|Encrypted Update| Net[Internet (TLS 1.3)]
    end

    subgraph "APHE Cloud (Untrusted)"
        Net -->|Encrypted Blobs| S[Aggregation Server]
        S -->|Homomorphic Sum| S
        S -->|Encrypted Global Model| Net
    end
    
    subgraph "Dashboard"
        S -.->|Metrics Only| D[Integrator Dashboard]
    end
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/tensor-crate/tensor-crate.git
cd tensor-crate

# Install dependencies
pip install -e .
```

---

## 🏃‍♂️ Quick Start

### 1. Run the Aggregation Server
Start the central server that will coordinate the Federated Learning rounds.

```bash
# Run the server (default: localhost:8080)
aphe-shield server --rounds 3
```

### 2. Start the Showcase Dashboard
Launch the web interface to visualize the training process.

```bash
# Start the dashboard server (default: localhost:8000)
aphe-shield dashboard
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### 3. Run an Edge Client (Robot)
Simulate a robot connecting to the server and submitting encrypted updates.

```bash
# Run a client with a mock VLA adapter
aphe-shield client --client-id robot-01
```

---

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

*   **[Deployment Guide](docs/aphe_shield_deployment_guide.md)**: Architecture topologies for air-gapped and networked factories.
*   **[SDK Reference](docs/aphe_shield_sdk_reference.md)**: API documentation for `EdgeClient`, `VLAAdapter`, and `N2HEEncryptor`.
*   **[Case Study: Pi0.5 + RL](docs/case_study_pi0_5_rl.md)**: A step-by-step guide to fine-tuning a Pi0.5 VLA using Reinforcement Learning.

---

## 🔧 Technical Details

### N2HE Encryption Engine
APHE-Shield uses a custom implementation of **N2HE (Neural Network Homomorphic Encryption)** based on the LWE (Learning With Errors) hardness assumption.

*   **Security Level**: 128-bit Post-Quantum.
*   **Optimization**: 
    *   **Vectorized LWE**: NumPy-accelerated matrix operations.
    *   **Zero-Copy Serialization**: Custom protocol replaced `pickle` for a **1300x speedup** (reduced serialization time from ~10s to ~7ms per chunk).
    *   **Chunking**: 1KB block processing to prevent OOM errors on edge devices.

### Project Structure

```
├── docs/                 # Documentation & Case Studies
├── src/
│   └── moai_shield/
│       ├── adapters.py   # VLA Model Adapters (Pi0, OpenVLA, etc.)
│       ├── cli.py        # Command Line Interface Entry Points
│       ├── edge_client.py# FL Client & Privacy Pipeline
│       ├── n2he.py       # N2HE Encryption Library
│       ├── server.py     # Homomorphic Aggregation Server
│       └── dashboard/    # White-label Dashboard Assets
└── tests/                # Benchmarks and Sanity Checks
```

---

## License

Proprietary License - Do not distribute without authorization.
Copyright © 2025 Dynamical Edge.
