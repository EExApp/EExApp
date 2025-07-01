"""
Centralized configuration for the MCMA-PPO algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    # Environment configuration
    ENV = {
        'num_slices': 3,  # Number of network slices (eMBB, URLLC, mMTC)
        'N_sf': 7,        # Number of DL slots in a frame
        'user_num': 3,    # Number of UEs to simulate
        'max_ues': 10,    # Maximum number of UEs supported
        'control_file': '../trandata/slice_ctrl.bin',  # Path to control file
        'kpm_file': '../trandata/KPM_UE.txt',         # Path to KPM file
        'lambda_p': 0.7,  # Throughput penalty weight in NS reward calculation
        'lambda_d': 0.3,  # Delay penalty weight in NS reward calculation
        'qos_targets': [  # QoS targets for each slice type
            {'throughput': 100, 'delay': 10},  # eMBB: High throughput, moderate latency
            {'throughput': 50, 'delay': 5},    # URLLC: Moderate throughput, low latency
            {'throughput': 20, 'delay': 20}    # mMTC: Low throughput, high latency tolerance
        ]
    }
    
    # State encoder configuration
    STATE_ENCODER = {
        'num_features': 17,     # Number of features per UE (10 MAC + 7 KPM)
        'hidden_dim': 64,      # Hidden dimension for GRU and attention
        'num_layers': 2,        # Number of GRU layers
        'num_heads': 4,         # Number of attention heads (not used)
        'dropout': 0.1,         # Dropout rate (not used)
    }
    
    # Actor-critic configuration
    ACTOR_CRITIC = {
        'state_dim': 64,       # Dimension of encoded state
        'ee_action_dim': 3,     # Energy efficiency actions (a_t, b_t, c_t)
        'ns_action_dim': 3,     # Network slicing actions (slice1, slice2, slice3)
        'num_slices': 3,        # Number of network slices
        'hidden_sizes': [64, 64], # Hidden layer sizes for MLPs
    }
    
    # GAT configuration
    GAT = {
        'input_dim': 1,         # Input dimension (critic values)
        'hidden_dim': 64,       # Hidden dimension for GAT layers
        'output_dim': 1,        # Output dimension (aggregated values)
        'num_heads': 4,         # Number of attention heads
        'dropout': 0.1,         # Dropout rate
    }
    
    # Reward weights for combining decomposed rewards
    REWARD = {
        'w_ee': 0.6,                # Weight for energy efficiency reward
        'w_ns': 0.4                 # Weight for QoS/network slicing reward
    }
    
    '''
    For epochs times:

        Run the policy to collect steps_per_epoch transitions.

        For train_pi_iters times:

            Optimize the policy using PPO loss.

        For train_v_iters times:

            Optimize the value function using TD or advantage-based targets.

        Enforce early stop if KL divergence exceeds target_kl.
    '''
    
    # PPO configuration
    PPO = {
        'steps_per_epoch': 100,    # Number of state-action-reward transitions per epoch
        'epochs': 100,              # Increased for longer training
        'gamma': 0.99,              # Discount factor
        'clip_ratio': 0.2,          # PPO clip ratio
        'pi_lr': 3e-4,              # Policy learning rate
        'vf_lr': 1e-3,              # Value function learning rate (often slightly higher)
        'train_pi_iters': 10,       # Policy updates per epoch
        'train_v_iters': 10,        # Value function updates per epoch
        'lam': 0.97,                # GAE-Lambda, closer to 1 for long horizons
        'max_ep_len': 100,         # Maximum episode length
        'target_kl': 0.015,         # Target KL divergence, slightly relaxed
        'save_freq': 10,            # Model save frequency (epochs)
    }
    
    # Action space configuration
    ACTION_SPACE = {
        'ee': {
            'min': 0,           # Minimum value for energy efficiency actions
            'max': 7,           # Maximum value for energy efficiency actions
            'sum': 7,           # Sum constraint for energy efficiency actions
        },
        'ns': {
            'min': 20,          # Minimum percentage for network slicing
            'max': 80,          # Maximum percentage for network slicing
            'sum': 100,         # Sum constraint for network slicing
        }
    }
    
    # Normalization ranges
    NORMALIZATION = {
        'MAC': {
            'dl_curr_tbs': (0, 3000),    # 0-3000 bytes
            'dl_sched_rb': (0, 106),     # 0-106 RBs
            'pusch_snr': (0, 70),        # 0-70 dB
            'pucch_snr': (0, 50),        # 0-50 dB
            'wb_cqi': (0, 15),           # 0-15
            'dl_mcs1': (0, 28),          # 0-28
            'ul_mcs1': (0, 28),          # 0-28
            'phr': (20, 70),             # 20-70 dB
            'dl_bler': (0, 0.5),         # 0-0.5
            'ul_bler': (0, 0.5)          # 0-0.5
        },
        'KPM': {
            'pdcp_sdu_volume_dl': (0, 20000),  # 0-20000 bytes
            'pdcp_sdu_volume_ul': (0, 20000),  # 0-20000 bytes
            'rlc_sdu_delay_dl': (0, 1000),     # 0-1000 ms
            'ue_thp_dl': (0, 1000),            # 0-1000 kbps
            'ue_thp_ul': (0, 1000),            # 0-1000 kbps
            'prb_tot_dl': (0, 1000),           # 0-1000 PRBs
            'prb_tot_ul': (0, 1000)            # 0-1000 PRBs
        }
    }

    # Activation functions (centralized)
    ACTIVATIONS = {
        'mlp_hidden': nn.ReLU,           # For hidden layers in MLPs (Critic, Actor)
        'mlp_output': nn.Identity,       # For output layer in MLPs (default)
        'gat': nn.ReLU,                  # For GAT layers
        'gat_leakyrelu': nn.LeakyReLU,   # For GAT attention
        'actor_softmax': F.softmax,      # For discrete actor output
    }

# Create global config instance
config = Config() 
