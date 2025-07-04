#!/usr/bin/env python3
"""
Debug script to understand environment behavior and reward calculation.
"""

import numpy as np
import struct
import os
import time
from env import OranEnv
from config import config

def debug_environment_data():
    """Debug what data the environment is actually reading."""
    print("=== Environment Data Debug ===")
    
    env = OranEnv()
    
    # Read state multiple times to see if it changes
    print("Reading environment state multiple times...")
    for i in range(3):
        state = env.get_all_state()
        throughput = [env.RRU_ThpDL_UE[ue_idx] * 1000 for ue_idx in range(env.user_num)]
        delay = [env.DRB_Delay[ue_idx] * 1000 for ue_idx in range(env.user_num)]
        
        print(f"Read {i+1}:")
        print(f"  Throughput: {[f'{t:.1f}' for t in throughput]}")
        print(f"  Delay: {[f'{d:.1f}' for d in delay]}")
        print(f"  State shape: {state.shape}")
        time.sleep(1)
    
    return env

def debug_reward_calculation():
    """Debug the reward calculation with different actions."""
    print("\n=== Reward Calculation Debug ===")
    
    env = OranEnv()
    
    # Test with very different actions
    test_actions = [
        ([33.33, 33.33, 33.34], [2, 3, 2]),  # Balanced
        ([80, 10, 10], [1, 5, 1]),           # Extreme slicing
        ([10, 80, 10], [3, 1, 3]),           # Another extreme
        ([50, 30, 20], [4, 0, 3]),           # No sleep
        ([20, 50, 30], [0, 7, 0]),           # All sleep
    ]
    
    print("Testing reward calculation with different actions:")
    for i, (slicing, sleep) in enumerate(test_actions):
        print(f"\nAction {i}: slice={slicing}, sleep={sleep}")
        
        # Get current environment state
        state = env.get_all_state()
        throughput = [env.RRU_ThpDL_UE[ue_idx] * 1000 for ue_idx in range(env.user_num)]
        delay = [env.DRB_Delay[ue_idx] * 1000 for ue_idx in range(env.user_num)]
        
        print(f"  Environment throughput: {[f'{t:.1f}' for t in throughput]}")
        print(f"  Environment delay: {[f'{d:.1f}' for d in delay]}")
        
        # Calculate reward
        reward = env.calculate_system_reward(slicing, sleep)
        print(f"  Reward: {reward:.3f}")

def debug_control_files():
    """Debug control file creation and reading."""
    print("\n=== Control File Debug ===")
    
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

def debug_step_function():
    """Debug the step function with group actions."""
    print("\n=== Step Function Debug ===")
    
    env = OranEnv()
    
    # Test group actions
    group_actions = [
        ([33.33, 33.33, 33.34], [2, 3, 2]),  # Balanced
        ([80, 10, 10], [1, 5, 1]),           # Extreme slicing
        ([10, 80, 10], [3, 1, 3]),           # Another extreme
    ]
    
    print("Testing step function with group actions...")
    print("Group actions:")
    for i, (slicing, sleep) in enumerate(group_actions):
        print(f"  Action {i}: slice={slicing}, sleep={sleep}")
    
    # Execute step
    last_state, group_rewards, _, _ = env.step(group_actions)
    
    print(f"Group rewards: {[f'{r:.3f}' for r in group_rewards]}")
    print(f"Final state shape: {last_state.shape}")

def debug_qos_targets():
    """Debug QoS targets and penalty calculation."""
    print("\n=== QoS Targets Debug ===")
    
    print("QoS targets from config:")
    for i, target in enumerate(config.ENV['qos_targets']):
        print(f"  Slice {i}: throughput={target['throughput']} kbps, delay={target['delay']} ms")
    
    print(f"Lambda values: eta={config.ENV.get('lambda_eta', 1.0)}, p={config.ENV['lambda_p']}, d={config.ENV['lambda_d']}")

if __name__ == "__main__":
    print("Comprehensive Environment Debug")
    print("=" * 60)
    
    try:
        debug_control_files()
        debug_qos_targets()
        env = debug_environment_data()
        debug_reward_calculation()
        debug_step_function()
        
        print("\n" + "=" * 60)
        print("DEBUG SUMMARY:")
        print("1. Check if environment data changes between reads")
        print("2. Check if reward calculation uses real data")
        print("3. Check if control files are properly created")
        print("4. Check if step function produces different rewards")
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc() 