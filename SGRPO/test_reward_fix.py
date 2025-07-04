#!/usr/bin/env python3
"""
Test script to verify that the reward fix works correctly.
"""

import numpy as np
from env import OranEnv
from config import config

def test_different_rewards():
    """Test that different actions now produce different rewards."""
    print("Testing reward fix - different actions should produce different rewards...")
    
    env = OranEnv()
    
    # Test with very different actions
    test_actions = [
        ([33.33, 33.33, 33.34], [2, 3, 2]),  # Balanced
        ([80, 10, 10], [1, 5, 1]),           # Extreme slicing
        ([10, 80, 10], [3, 1, 3]),           # Another extreme
        ([50, 30, 20], [4, 0, 3]),           # No sleep
        ([20, 50, 30], [0, 7, 0]),           # All sleep
    ]
    
    print("Testing individual action rewards:")
    rewards = []
    for i, (slicing, sleep) in enumerate(test_actions):
        reward = env.calculate_system_reward(slicing, sleep)
        rewards.append(reward)
        print(f"Action {i}: slice={slicing}, sleep={sleep}, reward={reward:.3f}")
    
    # Check if rewards are different
    unique_rewards = len(set([round(r, 2) for r in rewards]))
    print(f"\nUnique rewards: {unique_rewards}/{len(rewards)}")
    
    if unique_rewards > 1:
        print("✅ SUCCESS: Different actions produce different rewards!")
    else:
        print("❌ FAILURE: All actions still produce the same reward.")
    
    return rewards

def test_group_rewards():
    """Test that group actions produce different rewards."""
    print("\nTesting group action rewards...")
    
    env = OranEnv()
    
    # Test group actions
    group_actions = [
        ([33.33, 33.33, 33.34], [2, 3, 2]),  # Balanced
        ([80, 10, 10], [1, 5, 1]),           # Extreme slicing
        ([10, 80, 10], [3, 1, 3]),           # Another extreme
    ]
    
    # Get initial state
    initial_state = env.get_all_state()
    print(f"Initial state shape: {initial_state.shape}")
    
    # Test group step
    last_state, group_rewards, _, _ = env.step(group_actions)
    print(f"Group rewards: {[f'{r:.3f}' for r in group_rewards]}")
    
    # Check if rewards are different
    unique_rewards = len(set([round(r, 2) for r in group_rewards]))
    print(f"Unique group rewards: {unique_rewards}/{len(group_rewards)}")
    
    if unique_rewards > 1:
        print("✅ SUCCESS: Group actions produce different rewards!")
    else:
        print("❌ FAILURE: All group actions still produce the same reward.")
    
    return group_rewards

def analyze_reward_components():
    """Analyze the components of the reward function for one action."""
    print("\nAnalyzing reward function components...")
    
    env = OranEnv()
    
    # Test one action
    slicing_action = [33.33, 33.33, 33.34]
    sleep_action = [2, 3, 2]
    
    # Calculate reward manually to understand components
    lambda_eta = config.ENV['lambda_eta']
    lambda_p = config.ENV['lambda_p']
    lambda_d = config.ENV['lambda_d']
    qos_targets = config.ENV['qos_targets']
    
    # Get current state
    state = env.get_all_state()
    
    # Sleep time
    _, b_t, _ = sleep_action
    T_sleep = b_t
    
    # Throughput and delay data
    p_tk_list = [env.RRU_ThpDL_UE[ue_idx] * 1000 for ue_idx in range(env.user_num)]
    d_tk_list = [env.DRB_Delay[ue_idx] * 1000 for ue_idx in range(env.user_num)]
    
    print(f"Base throughput per UE: {[f'{t:.1f}' for t in p_tk_list]}")
    print(f"Base delay per UE: {[f'{d:.1f}' for d in d_tk_list]}")
    print(f"Sleep time: {T_sleep}")
    
    # Calculate modified values
    modified_throughput = []
    modified_delay = []
    
    for ue_idx in range(env.user_num):
        slice_name = env.get_slice_type(ue_idx + 1)
        slice_idx = {'embb': 0, 'urllc': 1, 'mmtc': 2}[slice_name]
        
        base_thp = p_tk_list[ue_idx]
        base_delay = d_tk_list[ue_idx]
        
        slice_allocation = slicing_action[slice_idx] / 100.0
        slicing_factor = 0.5 + 0.5 * slice_allocation
        sleep_factor = 1.0 - (b_t / 7.0) * 0.3
        
        modified_thp = base_thp * slicing_factor * sleep_factor
        modified_delay = base_delay / (slicing_factor * sleep_factor)
        
        modified_throughput.append(modified_thp)
        modified_delay.append(modified_delay)
    
    print(f"Modified throughput per UE: {[f'{t:.1f}' for t in modified_throughput]}")
    print(f"Modified delay per UE: {[f'{d:.1f}' for d in modified_delay]}")
    
    # Energy efficiency
    total_throughput = sum(modified_throughput)
    eta_t = (total_throughput * T_sleep) / 1000.0
    energy_reward = lambda_eta * eta_t
    
    print(f"Total throughput: {total_throughput:.1f}")
    print(f"Energy efficiency: {eta_t:.3f}")
    print(f"Energy reward: {energy_reward:.3f}")
    
    # Penalties
    penalty_p = 0.0
    penalty_d = 0.0
    
    for s_idx, slice_name in enumerate(['embb', 'urllc', 'mmtc']):
        P_s = qos_targets[s_idx]['throughput']
        D_s = qos_targets[s_idx]['delay']
        
        users_in_slice = [i for i in range(env.user_num) if env.get_slice_type(i+1) == slice_name]
        
        for k in users_in_slice:
            p_tk = modified_throughput[k]
            d_tk = modified_delay[k]
            thp_penalty = max(0, (P_s - p_tk) / P_s)
            delay_penalty = max(0, (d_tk - D_s) / D_s)
            penalty_p += thp_penalty
            penalty_d += delay_penalty
    
    throughput_penalty = lambda_p * penalty_p
    delay_penalty = lambda_d * penalty_d
    
    print(f"Throughput penalty: {penalty_p:.3f} -> {throughput_penalty:.3f}")
    print(f"Delay penalty: {penalty_d:.3f} -> {delay_penalty:.3f}")
    
    reward = energy_reward - throughput_penalty - delay_penalty
    print(f"Final reward: {reward:.3f}")

if __name__ == "__main__":
    print("Testing reward fix in SGRPO environment...")
    print("=" * 60)
    
    try:
        individual_rewards = test_different_rewards()
        group_rewards = test_group_rewards()
        analyze_reward_components()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"Individual rewards range: {min(individual_rewards):.3f} to {max(individual_rewards):.3f}")
        print(f"Group rewards range: {min(group_rewards):.3f} to {max(group_rewards):.3f}")
        
        if len(set([round(r, 2) for r in individual_rewards])) > 1:
            print("✅ FIX SUCCESSFUL: Actions now produce different rewards!")
        else:
            print("❌ FIX FAILED: Actions still produce the same rewards.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 