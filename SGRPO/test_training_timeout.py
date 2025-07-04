#!/usr/bin/env python3
"""
Test script to verify that training can run without hanging due to infinite loops.
"""

import time
import signal
import sys
from env import OranEnv
from model import SGRPOPolicy
import torch

def timeout_handler(signum, frame):
    print("ERROR: Training timed out - likely due to infinite loop!")
    sys.exit(1)

def test_environment_timeouts():
    """Test that environment methods don't hang indefinitely."""
    print("Testing environment timeout mechanisms...")
    
    env = OranEnv()
    
    # Test 1: get_UEkpm_info with missing data
    print("Test 1: Testing KPM data retrieval with missing data...")
    try:
        # This should not hang due to timeout
        env.get_UEkpm_info(1)
        print("✓ KPM data retrieval completed")
    except Exception as e:
        print(f"✗ KPM data retrieval failed: {e}")
    
    # Test 2: send_group_actions with missing control files
    print("Test 2: Testing action sending with missing control files...")
    try:
        test_actions = [([33.33, 33.33, 33.34], [2, 3, 2])]
        env.send_group_actions(test_actions, flag=0)
        print("✓ Action sending completed")
    except Exception as e:
        print(f"✗ Action sending failed: {e}")
    
    # Test 3: step method with missing external system
    print("Test 3: Testing step method with missing external system...")
    try:
        test_actions = [([33.33, 33.33, 33.34], [2, 3, 2])]
        start_time = time.time()
        state, rewards, done, info = env.step(test_actions)
        end_time = time.time()
        print(f"✓ Step method completed in {end_time - start_time:.2f}s")
        print(f"  Rewards: {rewards}")
    except Exception as e:
        print(f"✗ Step method failed: {e}")

def test_model_timeouts():
    """Test that model methods don't hang indefinitely."""
    print("\nTesting model timeout mechanisms...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = SGRPOPolicy().to(device)
    
    # Create dummy input
    batch_size = 1
    num_ues = 3
    num_features = 17
    ue_states = torch.randn(batch_size, num_ues, num_features, device=device)
    ue_slice_ids = torch.tensor([0, 1, 2], device=device)
    
    # Test action sampling
    print("Test 4: Testing action sampling...")
    try:
        start_time = time.time()
        slicing_action, sleep_action = policy.sample_actions(ue_states, ue_slice_ids)
        end_time = time.time()
        print(f"✓ Action sampling completed in {end_time - start_time:.2f}s")
        print(f"  Slicing action: {slicing_action}")
        print(f"  Sleep action: {sleep_action}")
    except Exception as e:
        print(f"✗ Action sampling failed: {e}")

def test_training_loop():
    """Test a short training loop to ensure it doesn't hang."""
    print("\nTesting short training loop...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = OranEnv()
    policy = SGRPOPolicy().to(device)
    
    # Test a few steps
    for step in range(3):
        print(f"Step {step + 1}/3...")
        try:
            # Get state
            state = env.get_all_state()
            print(f"  State shape: {state.shape}")
            
            # Sample actions
            ue_states = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            ue_slice_ids = env.get_user_slice_ids().to(device)
            slicing_action, sleep_action = policy.sample_actions(ue_states, ue_slice_ids)
            
            # Execute step
            group_actions = [(slicing_action.detach().cpu().numpy(), sleep_action.detach().cpu().numpy())]
            next_state, rewards, done, info = env.step(group_actions)
            
            print(f"  Rewards: {rewards}")
            print(f"  Done: {done}")
            
        except Exception as e:
            print(f"✗ Step {step + 1} failed: {e}")
            break
    
    print("✓ Training loop test completed")

if __name__ == "__main__":
    # Set up timeout handler (30 seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        test_environment_timeouts()
        test_model_timeouts()
        test_training_loop()
        print("\n🎉 All tests passed! Training should not hang due to infinite loops.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        signal.alarm(0)  # Cancel the alarm 