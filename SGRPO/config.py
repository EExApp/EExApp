"""
Centralized configuration for the SGRPO algorithm (Spatial Group Relative Policy Optimization).
"""

BATCH_SIZE = 1

class Config:
    # Environment configuration
    ENV = {
        'num_slices': 3,  # Number of network slices (eMBB, URLLC, mMTC)
        'N_sf': 7,        # Number of DL slots in a frame
        'user_num': 3,    # Number of UEs to simulate (can be dynamic)
        'max_ues': 10,    # Maximum number of UEs supported
        'control_file': '../trandata/slice_ctrl.bin',  # Path to control file
        'kpm_file': '../trandata/KPM_UE.txt',         # Path to KPM file
        'lambda_p': 0.7,  # Much reduced throughput penalty weight
        'lambda_d': 0.3,  # Much reduced delay penalty weight
        'lambda_eta': 1.0, # Much increased energy efficiency reward weight
        'lambda_sleep': 1.0, # Added for decoupled energy reward
        'lambda_thp': 1.0,   # Added for decoupled energy reward
        'qos_targets': [  # QoS targets - throughput: Mbps, delay: ms
            {'throughput': 8e-3, 'delay': 1000},  # eMBB: High throughput, moderate latency
            {'throughput': 18e-3, 'delay': 2000},    # URLLC: Moderate throughput, low latency
            {'throughput': 80e-3, 'delay': 500}    # mMTC: Low throughput, high latency tolerance
        ]
    }
    
    # State encoder configuration (Transformer)
    STATE_ENCODER = {
        'num_features': 17,     # Number of features per UE (10 MAC + 7 KPM)
        'hidden_dim': 64,      # Hidden dimension for transformer
        'num_layers': 2,        # Number of transformer layers
        'num_heads': 4,         # Number of attention heads
        'dropout': 0.1,         # Dropout rate
    }
    
    # Joint Policy Network configuration
    POLICY = {
        'state_dim': 64,       # Dimension of encoded state
        'slicing_action_dim': 3,     # Slicing actions (slice1, slice2, slice3)
        'sleep_action_dim': 3,       # Sleep actions (a_t, b_t, c_t)
        'hidden_sizes': [64, 64],    # Hidden layer sizes for MLPs
        'transformer': True,         # Use transformer encoder
    }
    
    # SGRPO training configuration
    SGRPO = {
        'steps_per_epoch': 50,    # Number of state-action-reward transitions per epoch
        'epochs': 100,              # Number of epochs
        'gamma': 0.99,              # Discount factor (if needed for future extensions)
        'clip_ratio': 0.2,          # PPO/GRPO clip ratio
        'pi_lr': 1e-5,              # Reduced learning rate for stability
        'train_pi_iters': 5,        # More frequent policy updates
        'max_ep_len': 100,         # Maximum episode length
        'target_kl': 1,           # Much higher target KL to allow more updates
        'save_freq': 10,            # Model save frequency (epochs)
        'epsilon': 0.1,             # Clipping parameter for GRPO
        'beta_kl': 0.1,          # Increased KL penalty coefficient for stability
        'group_size': 6,            # Number of actions to sample per group (G)
    }
    
    # Action space configuration
    ACTION_SPACE = {
        'sleep': {
            'min': 0,           # Minimum value for sleep actions
            'max': 7,           # Maximum value for sleep actions
            'sum': 7,           # Sum constraint for sleep actions
        },
        'slicing': {
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

# Create global config instance
config = Config() 
