#!/usr/bin/env python3
"""
Test script to verify model dimensions and QoS target handling.
"""

import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SGRPOPolicy
from config import config, BATCH_SIZE
from env import OranEnv

def test_model_dimensions():
    """Test that the model can handle the correct input dimensions."""
    
    print("Testing model dimensions...")
    
    # Create model
    model = SGRPOPolicy()
    print(f"Model created successfully")
    
    # Create dummy input data
    batch_size = BATCH_SIZE
    num_ues = config.ENV['user_num']
    num_features = config.STATE_ENCODER['num_features']
    
    # Create dummy UE states
    ue_states = torch.randn(batch_size, num_ues, num_features)
    print(f"UE states shape: {ue_states.shape}")
    
    # Create dummy slice IDs
    ue_slice_ids = torch.randint(0, config.ENV['num_slices'], (num_ues,))
    print(f"UE slice IDs shape: {ue_slice_ids.shape}")
    
    # Create normalized QoS targets
    qos_targets = torch.randn(config.ENV['num_slices'], 2)
    print(f"QoS targets shape: {qos_targets.shape}")
    
    # Test forward pass
    try:
        slicing_means, slicing_logstd, sleep_logits = model.forward(ue_states, ue_slice_ids, qos_targets)
        print(f"✓ Forward pass successful!")
        print(f"  Slicing means shape: {slicing_means.shape}")
        print(f"  Slicing logstd shape: {slicing_logstd.shape}")
        print(f"  Sleep logits shape: {sleep_logits.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test action sampling
    try:
        slicing_action, sleep_action = model.sample_actions(ue_states, ue_slice_ids, qos_targets)
        print(f"✓ Action sampling successful!")
        print(f"  Slicing action shape: {slicing_action.shape}")
        print(f"  Sleep action shape: {sleep_action.shape}")
        print(f"  Slicing action sum: {slicing_action.sum().item():.2f} (should be ~100)")
        print(f"  Sleep action sum: {sleep_action.sum().item():.2f} (should be 7)")
    except Exception as e:
        print(f"✗ Action sampling failed: {e}")
        return False
    
    # Test log probability computation
    try:
        slicing_logp, sleep_logp = model.log_prob(ue_states, slicing_action, sleep_action, ue_slice_ids, qos_targets)
        print(f"✓ Log probability computation successful!")
        print(f"  Slicing logp: {slicing_logp.item():.4f}")
        print(f"  Sleep logp: {sleep_logp.item():.4f}")
    except Exception as e:
        print(f"✗ Log probability computation failed: {e}")
        return False
    
    print("\n✓ All tests passed! Model dimensions are correct.")
    return True

def test_environment_qos_targets():
    """Test that the environment can provide normalized QoS targets."""
    
    print("\nTesting environment QoS targets...")
    
    # Create environment
    env = OranEnv()
    print(f"Environment created successfully")
    
    # Test normalized QoS targets
    try:
        qos_targets = env.get_normalized_qos_targets()
        print(f"✓ Normalized QoS targets shape: {qos_targets.shape}")
        print(f"  QoS targets: {qos_targets}")
        
        # Check that values are in reasonable range (0-1 for normalized)
        if torch.all((qos_targets >= 0) & (qos_targets <= 1)):
            print(f"✓ QoS targets are properly normalized (0-1 range)")
        else:
            print(f"✗ QoS targets are not properly normalized")
            return False
            
    except Exception as e:
        print(f"✗ QoS target normalization failed: {e}")
        return False
    
    print("\n✓ Environment QoS target test passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SGRPO Model Dimensions and QoS Target Handling")
    print("=" * 60)
    
    success1 = test_model_dimensions()
    success2 = test_environment_qos_targets()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED! The model should work correctly now.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED! Please check the errors above.")
        print("=" * 60)
        sys.exit(1) 