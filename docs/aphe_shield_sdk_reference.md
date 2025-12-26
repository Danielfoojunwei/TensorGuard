# APHE-Shield SDK Reference

> **Version 1.1.0** | Privacy-Preserving VLA Fine-Tuning

---

## Quick Start

```python
from moai_shield import EdgeClient, VLAAdapter, ShieldConfig

# 1. Configure
config = ShieldConfig(
    model_type="pi0",                    # pi0 | openvla | rt2 | custom
    key_path="/secure/customer.pem",     # Customer-controlled encryption key
    cloud_endpoint="https://api.tensor-crate.ai",
)

# 2. Initialize
client = EdgeClient(config)

# 3. Fine-tune with privacy
for demo in demonstrations:
    client.submit_encrypted_update(demo)

# 4. Get improved model
model = client.pull_model()
```

---

## Core API

### EdgeClient

```python
class EdgeClient:
    """Main client for edge devices (runs on robot's compute)."""
    
    def __init__(self, config: ShieldConfig):
        """Initialize with configuration."""
    
    def submit_encrypted_update(
        self, 
        demonstration: Demonstration,
        priority: str = "normal"  # "normal" | "high" | "low"
    ) -> SubmissionReceipt:
        """
        Process demonstration and submit encrypted gradients.
        
        Data flow:
        1. Compute gradients locally
        2. Clip → Sparsify → Compress → Encrypt (N2HE)
        3. Submit to aggregation server
        
        Returns receipt with submission_id and estimated_aggregation_time.
        """
    
    def pull_model(self, version: str = "latest") -> VLAModel:
        """
        Pull aggregated model update.
        
        Decryption happens locally with customer's key.
        """
    
    def get_status(self) -> ClientStatus:
        """Get current learning status and metrics."""
```

### ShieldConfig

```python
@dataclass
class ShieldConfig:
    # Model Configuration
    model_type: str = "pi0"           # VLA model type
    model_path: Optional[str] = None  # Path to local model weights
    
    # Security
    key_path: str                     # Path to encryption key (PEM)
    security_level: int = 128         # 128 | 192 (post-quantum bits)
    
    # Network
    cloud_endpoint: str = "https://api.tensor-crate.ai"
    use_tor: bool = False             # Enable Tor routing for high-security
    
    # Performance
    compression_ratio: int = 32       # 8-128 (higher = smaller upload)
    sparsity: float = 0.01            # Top-K ratio (0.01 = top 1%)
    batch_size: int = 8               # Demos before upload
    
    # Privacy
    dp_epsilon: float = 1.0           # Differential privacy budget
    dp_delta: float = 1e-5            # DP delta parameter
```

### VLAAdapter

```python
class VLAAdapter:
    """Adapter for different VLA architectures."""
    
    @classmethod
    def from_pi0(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for Pi0 VLA."""
    
    @classmethod
    def from_openvla(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for OpenVLA."""
    
    @classmethod
    def from_rt2(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for RT-2."""
    
    @classmethod
    def from_custom(
        cls, 
        model: Any,
        gradient_fn: Callable,
        apply_fn: Callable
    ) -> "VLAAdapter":
        """Create adapter for custom VLA architecture."""
```

---

## Data Types

### Demonstration

```python
@dataclass
class Demonstration:
    """Single demonstration for fine-tuning."""
    
    # Required
    observations: List[np.ndarray]    # Camera frames [T, H, W, C]
    actions: List[np.ndarray]         # Robot actions [T, action_dim]
    
    # Optional metadata
    task_id: Optional[str] = None     # Task identifier
    episode_id: Optional[str] = None  # Episode identifier
    timestamp: Optional[float] = None # Unix timestamp
    
    # Privacy annotations
    contains_pii: bool = False        # If True, extra anonymization applied
```

### SubmissionReceipt

```python
@dataclass
class SubmissionReceipt:
    submission_id: str                # Unique ID for tracking
    encrypted_size_bytes: int         # Size after encryption
    compression_achieved: float       # Actual compression ratio
    estimated_aggregation: datetime   # When model update expected
    privacy_budget_used: float        # DP epsilon consumed this submission
```

---

## Security Features

### Key Management

```python
from moai_shield.security import KeyManager

# Generate new key pair (do once, store securely)
km = KeyManager()
km.generate_keypair(
    output_path="/secure/",
    key_size=2048,
    password="customer_secret"  # Optional HSM integration
)

# Keys generated:
# /secure/customer.pem      - Private key (NEVER leaves customer network)
# /secure/customer.pub      - Public key (shared with aggregation server)
```

### Audit Trail

```python
from moai_shield.audit import AuditLog

# All operations are logged locally
log = AuditLog("/var/log/aphe-shield/")

# Query audit trail
entries = log.query(
    start_time=datetime(2025, 1, 1),
    operation_type="submission"
)

# Export for compliance
log.export_csv("/compliance/audit_2025.csv")
```

---

## Integration Examples

### ROS2 Integration

```python
import rclpy
from moai_shield.ros2 import ShieldNode

class VLAFineTuningNode(ShieldNode):
    def __init__(self):
        super().__init__('vla_finetuning')
        
        # Subscribe to demonstration topic
        self.demo_sub = self.create_subscription(
            DemonstrationMsg,
            '/demonstrations',
            self.on_demonstration,
            10
        )
    
    def on_demonstration(self, msg):
        demo = Demonstration.from_ros_msg(msg)
        self.submit_encrypted_update(demo)
```

### Jetson Deployment

```bash
# Install on Jetson Orin
pip install aphe-shield[jetson]

# Verify CUDA acceleration
python -c "from moai_shield import check_hardware; check_hardware()"
# Output: ✓ CUDA 12.2 | ✓ TensorRT 8.6 | ✓ 32GB RAM | ✓ N2HE optimized
```

---

## API Endpoints (Cloud)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/submit` | POST | Submit encrypted gradient update |
| `/v1/model/{version}` | GET | Pull model (encrypted) |
| `/v1/status` | GET | Fleet learning status |
| `/v1/metrics` | GET | Aggregation metrics (no raw data) |

---

## Error Handling

```python
from moai_shield.exceptions import (
    EncryptionError,      # Key or encryption failure
    NetworkError,         # Cloud connectivity issue
    ModelVersionError,    # Version mismatch
    PrivacyBudgetError,   # DP budget exhausted
)

try:
    client.submit_encrypted_update(demo)
except PrivacyBudgetError:
    # Rotate to new privacy budget or wait for reset
    client.reset_privacy_budget()
```

---

## Performance Specs

| Operation | Jetson Orin NX | Jetson AGX Orin |
|-----------|----------------|-----------------|
| Gradient computation | ~500ms | ~200ms |
| N2HE encryption | ~2s | ~800ms |
| Upload (compressed) | ~50KB/demo | ~50KB/demo |
| Model pull | ~5MB | ~5MB |

---

## License

Apache 2.0 with Commons Clause (no competing SaaS)
