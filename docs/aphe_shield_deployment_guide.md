# APHE-Shield Deployment Guide

> Deployment scenarios for System Integrators

---

## Deployment Scenarios

### Scenario 1: Manufacturing Floor (High Security)

```
┌─────────────────────────────────────────────────────────────┐
│                    BMW Assembly Plant                       │
├─────────────────────────────────────────────────────────────┤
│  Security: Air-gapped network, HSM-backed keys              │
│  Robots: 20x Figure 01 humanoids                            │
│  Compute: Jetson AGX Orin per robot                         │
│  Aggregation: On-premise APHE server (self-hosted)          │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```python
config = ShieldConfig(
    model_type="pi0",
    key_path="hsm://slot/1",              # HSM-backed key
    cloud_endpoint="https://internal.bmw.net:8443",
    use_tor=False,                        # Internal network
    security_level=192,                   # Maximum security
)
```

---

### Scenario 2: Warehouse (Standard)

```
┌─────────────────────────────────────────────────────────────┐
│                    Amazon Fulfillment Center                │
├─────────────────────────────────────────────────────────────┤
│  Security: VPN to cloud, software keys                      │
│  Robots: 100x Agility Digit                                 │
│  Compute: Jetson Orin NX per robot                          │
│  Aggregation: APHE Cloud (managed)                          │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```python
config = ShieldConfig(
    model_type="openvla",
    key_path="/etc/moai/keys/amazon_fc.pem",
    cloud_endpoint="https://api.aphe-shield.ai",
    compression_ratio=64,                 # Higher compression for scale
)
```

---

### Scenario 3: Healthcare (HIPAA Compliant)

```
┌─────────────────────────────────────────────────────────────┐
│                    Mayo Clinic                              │
├─────────────────────────────────────────────────────────────┤
│  Security: HIPAA BAA, Tor routing, ephemeral keys           │
│  Robots: 5x assistive humanoids                             │
│  Compute: Jetson AGX Orin                                   │
│  Aggregation: Healthcare-certified cloud region             │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```python
config = ShieldConfig(
    model_type="custom",
    key_path="/secure/hipaa_key.pem",
    cloud_endpoint="https://healthcare.aphe-shield.ai",
    use_tor=True,                         # Maximum anonymity
    dp_epsilon=0.5,                       # Stricter privacy
)
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Edge Compute | Jetson Orin Nano (8GB) | Jetson AGX Orin (64GB) |
| Storage | 32GB NVMe | 256GB NVMe |
| Network | 100Mbps | 1Gbps |
| RAM | 8GB | 32GB+ |

---

## Installation

```bash
# Standard installation
pip install aphe-shield

# With Jetson optimizations
pip install aphe-shield[jetson]

# With ROS2 integration
pip install aphe-shield[ros2]

# Full enterprise (HSM, Tor, audit)
pip install aphe-shield[enterprise]
```

---

## Self-Hosted Cloud Setup

```bash
# Pull aggregation server
docker pull moaishield/aggregator:latest

# Run with config
docker run -d \
  -p 8443:8443 \
  -v /path/to/config:/etc/moai \
  -v /path/to/certs:/certs \
  moaishield/aggregator:latest
```

**Kubernetes Helm:**
```bash
helm repo add moaishield https://charts.tensor-crate.ai
helm install moai moaishield/aggregator \
  --set replicas=3 \
  --set storage.class=fast-ssd
```
