#!/usr/bin/env python3
"""
Test script to verify that flag checking and state reading works correctly.
"""

import numpy as np
import struct
import os
import time
from env import OranEnv
from config import config

def test_flag_checking():
    """Test that the environment properly checks flags and reads state."""
    print("Testing flag checking and state reading...")
    
    env = OranEnv()
    
    # Test group actions
    group_actions = [
        ([33.33, 33.33, 33.34], [2, 3, 2]),  # Balanced
        ([80, 10, 10], [1, 5, 1]),           # Extreme slicing
        ([10, 80, 10], [3, 1, 3]),           # Another extreme
    ]
    
    print(f"Testing with {len(group_actions)} group actions")
    
    # Get initial state
    initial_state = env.get_all_state()
    print(f"Initial state shape: {initial_state.shape}")
    
    # Test group step
    print("Starting group step...")
    start_time = time.time()
    
    last_state, group_rewards, _, _ = env.step(group_actions)
    
    end_time = time.time()
    print(f"Group step completed in {end_time - start_time:.2f} seconds")
    
    print(f"Group rewards: {[f'{r:.3f}' for r in group_rewards]}")
    
    # Check if rewards are different
    unique_rewards = len(set([round(r, 2) for r in group_rewards]))
    print(f"Unique group rewards: {unique_rewards}/{len(group_rewards)}")
    
    if unique_rewards > 1:
        print("✅ SUCCESS: Group actions produce different rewards!")
    else:
        print("❌ FAILURE: All group actions still produce the same reward.")
    
    return group_rewards

def test_individual_flag_checking():
    """Test individual action flag checking."""
    print("\nTesting individual action flag checking...")
    
    env = OranEnv()
    
    # Test single action
    single_action = ([50, 30, 20], [3, 1, 3])
    
    print("Testing single action...")
    last_state, single_rewards, _, _ = env.step([single_action])
    
    print(f"Single action reward: {single_rewards[0]:.3f}")
    
    return single_rewards[0]

def test_control_file_creation():
    """Test that control files are created properly."""
    print("\nTesting control file creation...")
    
    # Check if control files exist
    for g in range(3):
        file_name = f'../trandata/slice_ctrl_{g}.bin'
        if os.path.exists(file_name):
            print(f"Control file {g} exists")
            
            # Read the file
            with open(file_name, 'rb') as f:
                data = f.read(28)
                if len(data) == 28:
                    numbers = struct.unpack('iiiiiii', data)
                    print(f"  Slicing: {numbers[:3]}, Sleep: {numbers[3:6]}, Flag: {numbers[6]}")
                else:
                    print(f"  File incomplete: {len(data)} bytes")
        else:
            print(f"Control file {g} does not exist")

def test_state_consistency():
    """Test that state reading is consistent."""
    print("\nTesting state consistency...")
    
    env = OranEnv()
    
    # Read state multiple times
    states = []
    for i in range(3):
        state = env.get_all_state()
        states.append(state)
        print(f"State {i} shape: {state.shape}")
        time.sleep(0.1)
    
    # Check if states are consistent
    state_diff = np.max(np.abs(states[0] - states[1]))
    print(f"State difference between reads: {state_diff:.6f}")
    
    if state_diff < 1e-6:
        print("✅ States are consistent")
    else:
        print("⚠️  States are changing between reads")

if __name__ == "__main__":
    print("Testing flag checking and state reading in SGRPO environment...")
    print("=" * 60)
    
    try:
        test_control_file_creation()
        test_state_consistency()
        single_reward = test_individual_flag_checking()
        group_rewards = test_flag_checking()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"Single action reward: {single_reward:.3f}")
        print(f"Group rewards: {[f'{r:.3f}' for r in group_rewards]}")
        print(f"Group rewards range: {min(group_rewards):.3f} to {max(group_rewards):.3f}")
        
        if len(set([round(r, 2) for r in group_rewards])) > 1:
            print("✅ SUCCESS: Flag checking and state reading works correctly!")
        else:
            print("❌ FAILURE: Actions still produce the same rewards.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 