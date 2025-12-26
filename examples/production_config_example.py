"""
TensorGuard Production Configuration Examples

This file demonstrates how to configure TensorGuard for production deployments
with the full production blueprint features.
"""

import tensorguard as tg
from pathlib import Path

# ============================================================================
# Example 1: Conservative Production Profile
# ============================================================================
# Use case: High-security customer with strict privacy requirements

def conservative_production_config():
    """
    Conservative production configuration for high-security environments.
    """

    # Operating envelope - strict constraints
    envelope = tg.OperatingEnvelope(
        peft_strategy=tg.PEFTStrategy.LORA,
        trainable_modules=["policy_head", "last_4_blocks"],
        max_trainable_params=5_000_000,  # 5M params max

        # Round cadence - daily updates
        round_interval_seconds=86400,  # 24 hours

        # Update size - very conservative
        target_update_size_kb=100,
        min_update_size_kb=10,
        max_update_size_kb=1024,  # 1MB max

        # Deployment controls
        enable_canary=True,
        enable_rollback=True,
        canary_percentage=0.05,  # 5% canary
    )

    # DP policy - strong privacy guarantees
    dp_profile = tg.DPPolicyProfile(
        profile_name="conservative",
        clipping_norm=0.5,  # Aggressive clipping
        noise_multiplier=2.0,  # High noise
        epsilon_budget=1.0,  # Strict budget
        delta=1e-6,
        hard_stop_enabled=True,
        accounting_method="rdp"
    )

    # Encryption policy - maximum security
    encryption_profile = tg.EncryptionPolicyProfile(
        profile_name="conservative",
        security_level=192,  # 192-bit post-quantum
        key_rotation_schedule_hours=12,  # Rotate every 12h
        aggregation_quorum_threshold=5  # Need 5 clients minimum
    )

    # Training policy - conservative compression
    training_profile = tg.TrainingPolicyProfile(
        profile_name="conservative",
        learning_rate=1e-5,
        batch_size=4,
        compression_ratio=64,  # High compression
        sparsity=0.001,  # Very sparse (top 0.1%)
        max_quality_mse=0.01  # Strict quality threshold
    )

    # Shield config
    shield_config = tg.ShieldConfig(
        model_type="openvla",
        key_path="/secure/keys/customer_key.pem",
        cloud_endpoint="https://api.tensorguard.ai",
        security_level=192,
        compression_ratio=64,
        sparsity=0.001,
        batch_size=4,
        dp_epsilon=1.0,
        max_gradient_norm=0.5
    )

    # Create client
    client = tg.EdgeClient(
        config=shield_config,
        operating_envelope=envelope,
        dp_profile=dp_profile,
        encryption_profile=encryption_profile,
        training_profile=training_profile,
        enable_observability=True
    )

    return client


# ============================================================================
# Example 2: Balanced Production Profile
# ============================================================================
# Use case: Standard enterprise deployment balancing privacy and performance

def balanced_production_config():
    """
    Balanced production configuration for typical enterprise deployments.
    """

    # Operating envelope - balanced constraints
    envelope = tg.OperatingEnvelope(
        peft_strategy=tg.PEFTStrategy.LORA,
        trainable_modules=["policy_head", "last_6_blocks"],
        max_trainable_params=10_000_000,  # 10M params

        # Round cadence - hourly updates
        round_interval_seconds=3600,  # 1 hour

        # Update size - balanced
        target_update_size_kb=500,
        min_update_size_kb=10,
        max_update_size_kb=5120,  # 5MB max

        # Deployment controls
        enable_canary=True,
        enable_rollback=True,
        canary_percentage=0.1,  # 10% canary
    )

    # DP policy - balanced privacy
    dp_profile = tg.DPPolicyProfile(
        profile_name="balanced",
        clipping_norm=1.0,
        noise_multiplier=1.0,
        epsilon_budget=10.0,
        delta=1e-5,
        hard_stop_enabled=True,
        accounting_method="rdp"
    )

    # Encryption policy - standard security
    encryption_profile = tg.EncryptionPolicyProfile(
        profile_name="balanced",
        security_level=128,  # 128-bit post-quantum
        key_rotation_schedule_hours=24,  # Rotate daily
        aggregation_quorum_threshold=2  # Need 2 clients minimum
    )

    # Training policy - balanced compression
    training_profile = tg.TrainingPolicyProfile(
        profile_name="balanced",
        learning_rate=1e-4,
        batch_size=8,
        compression_ratio=32,
        sparsity=0.01,  # Top 1%
        max_quality_mse=0.05
    )

    # Shield config
    shield_config = tg.ShieldConfig(
        model_type="openvla",
        key_path="/secure/keys/customer_key.pem",
        cloud_endpoint="https://api.tensorguard.ai",
        security_level=128,
        compression_ratio=32,
        sparsity=0.01,
        batch_size=8,
        dp_epsilon=10.0,
        max_gradient_norm=1.0
    )

    # Create client
    client = tg.EdgeClient(
        config=shield_config,
        operating_envelope=envelope,
        dp_profile=dp_profile,
        encryption_profile=encryption_profile,
        training_profile=training_profile,
        enable_observability=True
    )

    return client


# ============================================================================
# Example 3: Performance-Optimized Profile
# ============================================================================
# Use case: Low-latency deployment with moderate privacy requirements

def performance_optimized_config():
    """
    Performance-optimized configuration for low-latency deployments.
    """

    # Operating envelope - optimized for speed
    envelope = tg.OperatingEnvelope(
        peft_strategy=tg.PEFTStrategy.ADAPTER,  # Lighter than LoRA
        trainable_modules=["policy_head", "last_2_blocks"],
        max_trainable_params=3_000_000,  # 3M params

        # Round cadence - frequent updates
        round_interval_seconds=600,  # 10 minutes

        # Update size - smaller for speed
        target_update_size_kb=100,
        min_update_size_kb=10,
        max_update_size_kb=2048,  # 2MB max

        # Deployment controls
        enable_canary=True,
        enable_rollback=True,
        canary_percentage=0.2,  # 20% canary (faster rollout)
    )

    # DP policy - moderate privacy
    dp_profile = tg.DPPolicyProfile(
        profile_name="performance",
        clipping_norm=2.0,  # Less aggressive clipping
        noise_multiplier=0.8,  # Lower noise
        epsilon_budget=50.0,  # Larger budget
        delta=1e-5,
        hard_stop_enabled=False,  # Allow soft limit
        accounting_method="rdp"
    )

    # Encryption policy - optimized
    encryption_profile = tg.EncryptionPolicyProfile(
        profile_name="performance",
        security_level=128,
        key_rotation_schedule_hours=48,  # Rotate every 2 days
        aggregation_quorum_threshold=2
    )

    # Training policy - high compression for speed
    training_profile = tg.TrainingPolicyProfile(
        profile_name="performance",
        learning_rate=5e-4,  # Higher LR
        batch_size=16,  # Larger batches
        compression_ratio=64,  # High compression
        sparsity=0.005,  # Top 0.5%
        max_quality_mse=0.1  # More tolerant
    )

    # Shield config
    shield_config = tg.ShieldConfig(
        model_type="pi0",  # Lighter model
        key_path="/secure/keys/customer_key.pem",
        cloud_endpoint="https://api.tensorguard.ai",
        security_level=128,
        compression_ratio=64,
        sparsity=0.005,
        batch_size=16,
        dp_epsilon=50.0,
        max_gradient_norm=2.0
    )

    # Create client
    client = tg.EdgeClient(
        config=shield_config,
        operating_envelope=envelope,
        dp_profile=dp_profile,
        encryption_profile=encryption_profile,
        training_profile=training_profile,
        enable_observability=True
    )

    return client


# ============================================================================
# Example 4: Training Pipeline Configuration
# ============================================================================

def configure_training_pipeline():
    """
    Configure the full IL -> Offline RL -> On-policy RL pipeline.
    """

    pipeline = tg.TrainingPipeline()

    # Configure IL stage
    pipeline.stages[tg.TrainingStage.IL_PEFT_BASELINE].enabled = True
    pipeline.stages[tg.TrainingStage.IL_PEFT_BASELINE].max_rounds = 5
    pipeline.stages[tg.TrainingStage.IL_PEFT_BASELINE].requires_approval = False

    # Configure Offline RL stage
    pipeline.stages[tg.TrainingStage.OFFLINE_RL_PEFT].enabled = True
    pipeline.stages[tg.TrainingStage.OFFLINE_RL_PEFT].max_rounds = 10
    pipeline.stages[tg.TrainingStage.OFFLINE_RL_PEFT].requires_approval = False

    # Configure On-policy RL stage (disabled by default)
    pipeline.stages[tg.TrainingStage.ONPOLICY_RL_PEFT].enabled = False
    pipeline.stages[tg.TrainingStage.ONPOLICY_RL_PEFT].max_rounds = 20
    pipeline.stages[tg.TrainingStage.ONPOLICY_RL_PEFT].requires_approval = True  # Require approval

    # Start with IL
    pipeline.start_stage(tg.TrainingStage.IL_PEFT_BASELINE, approved=True)

    return pipeline


# ============================================================================
# Example 5: Evaluation Gating Configuration
# ============================================================================

def configure_evaluation_gate():
    """
    Configure production evaluation gating with safety thresholds.
    """

    # Define safety thresholds
    thresholds = tg.SafetyThresholds(
        min_success_rate=0.85,  # 85% success required
        max_constraint_violations=3,
        max_kl_divergence=0.3,
        max_regression_delta=0.03,  # 3% max regression
        min_ood_robustness=0.7
    )

    # Create evaluation gate
    eval_gate = tg.EvaluationGate(thresholds=thresholds)

    # Set baseline metrics (from pre-deployment evaluation)
    baseline = tg.EvaluationMetrics(
        success_rate=0.90,
        constraint_violations=1,
        time_to_complete_mean=15.0,
        collision_proxy_score=0.95,
        kl_divergence_vs_baseline=0.0,
        ood_robustness_score=0.8
    )
    eval_gate.set_baseline(baseline)

    return eval_gate


# ============================================================================
# Example 6: Key Management Configuration
# ============================================================================

def configure_key_management():
    """
    Configure enterprise key management system.
    """

    kms = tg.KeyManagementSystem(
        audit_log_path=Path("/secure/logs/key_audit.log")
    )

    # Register initial key
    initial_key_metadata = tg.KeyMetadata(
        key_id="customer_key_v1",
        created_at=tg.datetime.now(),
        version=1,
        purpose="encryption",
        owner="customer_acme_corp",
        key_type="N2HE",
        security_level=128
    )
    kms.register_key("customer_key_v1", initial_key_metadata)

    return kms


# ============================================================================
# Example 7: Complete Production Deployment
# ============================================================================

def complete_production_deployment():
    """
    Complete production deployment with all features enabled.
    """

    print("=" * 70)
    print("TensorGuard Production Deployment Configuration")
    print("=" * 70)

    # 1. Create client with balanced profile
    print("\n1. Creating EdgeClient with balanced production profile...")
    client = balanced_production_config()

    # 2. Configure training pipeline
    print("2. Configuring training pipeline (IL -> Offline RL)...")
    pipeline = configure_training_pipeline()

    # 3. Configure evaluation gate
    print("3. Configuring evaluation gating with safety thresholds...")
    eval_gate = configure_evaluation_gate()

    # 4. Configure key management
    print("4. Configuring enterprise key management...")
    kms = configure_key_management()

    # 5. Print production status
    print("\n5. Production components initialized:")
    tg.print_production_status()

    # 6. Show observability metrics location
    if client.observability:
        print(f"\nObservability metrics: {client.observability.metrics_file}")

    # 7. Show key audit log location
    print(f"Key audit log: {kms.audit_log_path}")

    print("\n" + "=" * 70)
    print("Production deployment ready!")
    print("=" * 70)

    return {
        "client": client,
        "pipeline": pipeline,
        "eval_gate": eval_gate,
        "kms": kms
    }


if __name__ == "__main__":
    # Run complete production deployment
    deployment = complete_production_deployment()

    print("\nDeployment components:")
    for key, value in deployment.items():
        print(f"  - {key}: {type(value).__name__}")
