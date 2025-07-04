import gym
from gym import spaces
import numpy as np
import sqlite3
import struct
import os
import copy
from typing import List, Tuple, Dict
import time
from config import config
import torch
import random
import csv


class OranEnv(gym.Env):
    """
    Custom Gym environment for O-RAN energy efficiency and network slicing
    
    This environment simulates the interaction between the DRL agent and the O-RAN system,
    handling both energy efficiency (time scheduling) and network slicing control.
    
    Slice Configuration:
    - eMBB (Slice 1): High throughput, moderate latency
    - uRLLC (Slice 2): Low latency, high reliability
    - mMTC (Slice 3): High connection density, low data rate
    
    UE Allocation:
    - Fixed allocation using modulo operation
    - UE 1,4,7,10 -> eMBB
    - UE 2,5,8 -> uRLLC
    - UE 3,6,9 -> mMTC
    """
    def __init__(self, num_slices=None, N_sf=None, user_num=None):
        super(OranEnv, self).__init__()
        
        # Store config as instance attribute
        self.config = config
        
        # Use config values if not specified
        self.num_slices = num_slices or config.ENV['num_slices']
        self.N_sf = N_sf or config.ENV['N_sf']
        self.user_num = user_num or config.ENV['user_num']
        
        # Map UE IDs to their respective slices using modulo operation
        self.ue_slice_map = {}
        for ue_id in range(1, self.user_num + 1):
            if ue_id % 3 == 1:  # eMBB slice
                self.ue_slice_map[ue_id] = 'embb'
            elif ue_id % 3 == 2:  # uRLLC slice
                self.ue_slice_map[ue_id] = 'urllc'
            else:  # mMTC slice
                self.ue_slice_map[ue_id] = 'mmtc'
        
        # State space: 17-dimensional features (10 MAC metrics + 7 KPM metrics) - per UE
        # specification from openai gym
        # describe the shape, type and bounds of the observations that the environment returns
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(17,), dtype=np.float32
        )
        
        # Action space from config
        # EE action space: discrete actions for energy efficiency (a_t, b_t, c_t)
        # Each action can take values from 0 to max_value
        ee_max_values = [config.ACTION_SPACE['sleep']['max']] * 3  # 3 sleep actions
        self.ee_action_space = spaces.MultiDiscrete(ee_max_values)
        
        # NS action space: continuous actions for network slicing (slice1, slice2, slice3)
        self.ns_action_space = spaces.Box(
            low=np.array([config.ACTION_SPACE['slicing']['min']] * 3),
            high=np.array([config.ACTION_SPACE['slicing']['max']] * 3),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.state = None
        self.last_mac_state = [0]*10  # Single list for the last MAC state
        self.last_kpm_states = [ [0]*8 for _ in range(self.user_num) ]
        
        # Initialize control file
        self.control_file = config.ENV['control_file']
        self.kpm_file = config.ENV['kpm_file']
        
        # Initialize metrics tracking
        self.UE_MAC = []
        self.UE_KPM = []
        self.RRU_ThpDL_UE = []
        self.DRB_Delay = []
        self.BLER = []
        self.RRU_PrbTotDL = []
        
        # Initialize control file with default values
        self.init_control_file()
        
        # Add parameters from config
        self.lambda_eta = config.ENV['lambda_eta']
        self.lambda_p = config.ENV['lambda_p']
        self.lambda_d = config.ENV['lambda_d']
        self.qos_targets = config.ENV['qos_targets'][:self.num_slices]
        assert len(self.qos_targets) == self.num_slices, "Mismatch between num_slices and qos_targets!"
    
    
    def get_slice_type(self, ue_id):
        """Get slice type for a given UE ID using modulo operation"""
        return self.ue_slice_map[ue_id]
    
    def get_UEmac_layer_info(self, ue_id):
        """Get MAC layer metrics for a UE from SQLite database"""
        print(f"Getting MAC data for UE {ue_id}...")
        self.mac_state = []
        
        # Connect to database (like original code)
        db_path = '/home/crybabeblues/Projects/EExApp_project/EExApp_main/trandata/xapp_db_'
        print(f"Connecting to database: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query MAC metrics for the ONE UE  (rnti = ?)
        query = """
        SELECT dl_curr_tbs, dl_sched_rb, pusch_snr, pucch_snr, wb_cqi,
               dl_mcs1, ul_mcs1, phr, dl_bler, ul_bler
        FROM MAC_UE 
        WHERE rnti = ? 
        ORDER BY tstamp DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (ue_id,))
        #cursor.execute("SELECT dl_sched_rb,ul_sched_rb,pusch_snr,pucch_snr,wb_cqi,phr,dl_bler,ul_bler FROM MAC_UE WHERE rnti = 1 ORDER BY tstamp DESC LIMIT 1;")
        self.mac_state = cursor.fetchall()
        
        # Handle missing or incomplete data (check for 10 elements, not 8)
        if len(self.mac_state) == 0 or len(self.mac_state[0]) < 10:
            print(f"Using last known MAC state for UE {ue_id}")
            self.mac_state = [self.last_mac_state]
        else:
            print(f"Retrieved fresh MAC data for UE {ue_id}: {self.mac_state[0]}")
            self.last_mac_state = self.mac_state[0]
        
        # Check the total number of records
        
        '''
        count_query = "SELECT COUNT(*) FROM MAC_UE;"
        cursor.execute(count_query)
        record_count = cursor.fetchone()[0]

        # If the number of records exceeds 100, delete the oldest 50 records
        if record_count > 100:
            delete_query = """
            DELETE FROM MAC_UE 
            WHERE tstamp IN (
            SELECT tstamp 
            FROM MAC_UE 
            ORDER BY tstamp ASC 
            LIMIT 50
            );
            """
            cursor.execute(delete_query)
            conn.commit()    
        '''
        # 实际运行的时候读取之后清空 it does not work
        # cursor.execute("SELECT MAX(tstamp) FROM MAC_UE")
        # latest_id = cursor.fetchone()[0]
        # cursor.execute('DELETE FROM MAC_UE WHERE tstamp != ?', (latest_id,))
        conn.close()
    
    
    def get_UEkpm_info(self, ue_id):
        '''ue_id, DRB_pdcpSduVolumeDL, DRB_pdcpSduVolumeUL, DRB_RlcSduDelayDL(us), DRB_UEThpDL, DRB_UEThpUL, RRU_PrbTotDL(kbps), RRU_PrbTotUL '''
        print(f"Getting KPM data for UE {ue_id}...")
        data_ueid = []
        
        while True:
            with open(self.kpm_file, 'r') as file:
                lines = file.readlines()[-self.user_num:]  # Read last user_num lines
                
            for line in lines:
                data = [float(x) for x in line.split()]
                if int(data[0]) == ue_id:
                    data_ueid = data
                        
            if len(data_ueid) < 8:
                print(f"KPM data incomplete for UE {ue_id}, retrying...")
                continue
            else:
                if data_ueid == self.last_kpm_states[ue_id-1] and data_ueid[4] != 0:
                    print(f"KPM data unchanged for UE {ue_id}, retrying...")
                    continue
                else:
                    self.last_kpm_states[ue_id-1] = copy.deepcopy(data_ueid)
                    self.kpm_state = data_ueid
                    print(f"Retrieved KPM data for UE {ue_id}: {data_ueid}")
                break
                    
    
    def get_state(self, ue_id):
        """Get complete state vector for a UE with normalized metrics"""
        
        self.get_UEmac_layer_info(ue_id)
        self.get_UEkpm_info(ue_id)
        
        # Get raw MAC and KPM data
        raw_mac = list(self.mac_state[0])
        raw_kpm = list(self.kpm_state)
        
        # Normalize MAC features
        normalized_mac = []
        #dl_curr_tbs, dl_sched_rb, pusch_snr, pucch_snr, wb_cqi, dl_mcs1, ul_mcs1, phr, dl_bler, ul_bler
        normalized_mac.append((raw_mac[0]-0)/3000)          
        normalized_mac.append((raw_mac[1]-0)/106)                        
        # pusch_snr,pucch_snr, 0-70
        normalized_mac.append((raw_mac[2]-0)/70)           
        normalized_mac.append((raw_mac[3]-0)/50)
        # wb_cqi 0-15
        normalized_mac.append((raw_mac[4]-0)/15)
        # dl_mcs1, ul_mcs1
        normalized_mac.append(raw_mac[5]/28)           
        normalized_mac.append(raw_mac[6]/28)
        # phr -24-70
        normalized_mac.append((raw_mac[7]-20)/(70-20))
        # dl_bler,ul_bler 0-0.5
        normalized_mac.append(raw_mac[8]/0.5)           
        normalized_mac.append(raw_mac[9]/0.5)

        # Normalize KPM features
        normalized_kpm = []
        # 这里的数据针对的是100ms的kpm
        # DRB_pdcpSduVolumeDL, DRB_pdcpSduVolumeUL   0-80000
        normalized_kpm.append(raw_kpm[1]/20000)  # DRB_pdcpSduVolumeDL
        normalized_kpm.append(raw_kpm[2]/20000)  # DRB_pdcpSduVolumeUL
        #DRB_RlcSduDelayDL, 0-1000 us    #there will be a peak when states is changing 
        if raw_kpm[3] > 1000:
            raw_kpm[3] = 1000
        normalized_kpm.append(raw_kpm[3]/1000)   # DRB_RlcSduDelayDL
        #DRB_UEThpDL, DRB_UEThpUL, kbps 
        normalized_kpm.append(raw_kpm[4]/1000)   # DRB_UEThpDL (downlink)
        normalized_kpm.append(raw_kpm[5]/1000)   # DRB_UEThpUL (uplink)
        #RRU_PrbTotDL, RRU_PrbTotUL  0-70000
        normalized_kpm.append(raw_kpm[6]/1000)   # RRU_PrbTotDL
        normalized_kpm.append(raw_kpm[7]/1000)   # RRU_PrbTotUL

        # Create complete normalized state (10 MAC + 7 KPM = 17 features)
        complete_state = normalized_mac + normalized_kpm
        
        # Store for reward calculation
        self.MAC = normalized_mac
        self.KPM = normalized_kpm
        self.state = complete_state

        return complete_state, normalized_mac, normalized_kpm

    
    def get_all_state(self):
        """
        Get state vectors for all UEs as a numpy array [num_ues, num_features].
        """
        print("Starting get_all_state...")
        self.UE_MAC = []
        self.UE_KPM = []
        self.RRU_ThpDL_UE = []
        self.DRB_Delay = []
        self.BLER = []
        self.RRU_PrbTotDL = []
        for i in range(1, self.user_num + 1):
            complete_state, ue_mac_i, ue_kpm_i = self.get_state(i)
            self.UE_MAC.append(ue_mac_i)
            self.UE_KPM.append(ue_kpm_i)
            self.BLER.append(ue_mac_i[8])
            self.DRB_Delay.append(ue_kpm_i[2])
            self.RRU_ThpDL_UE.append(ue_kpm_i[3])
            self.RRU_PrbTotDL.append(ue_kpm_i[5])
        complete_states = []
        for i in range(len(self.UE_MAC)):
            state_vector = np.array(self.UE_MAC[i] + self.UE_KPM[i], dtype=np.float32)
            complete_states.append(state_vector)
        return np.stack(complete_states, axis=0)  # [num_ues, num_features]

    def init_control_file(self):
        """Initialize the control file with default values"""
        # Write initial values in binary format
        with open(self.control_file, 'wb') as f:
            # Default values: slice1=40, slice2=30, slice3=30, a_t=4, b_t=0, c_t=3, flag=0
            numbers = (40, 30, 30, 4, 0, 3, 1)
            f.write(struct.pack('iiiiiii', *numbers))

    def send_group_actions(self, group_actions, flag):
        """
        Write a group of actions to separate files (slice_ctrl_{g}.bin) in ../trandata/.
        Each action is a tuple: (slicing_action, sleep_action), both arrays/lists of 3 values.
        Only write if the file's flag is 1 (ready), and set flag to the provided value (control needed or ready).
        """
        for g, (slicing_action, sleep_action) in enumerate(group_actions):
            file_name = f'../trandata/slice_ctrl_{g}.bin'
            # Convert to ints
            slicing_ints = [int(round(x)) for x in slicing_action]
            sleep_ints = [int(round(x)) for x in sleep_action]
            # Ensure slicing sum to 100
            total_slice = sum(slicing_ints)
            if total_slice != 100:
                slicing_ints = [int(round(x * 100 / total_slice)) for x in slicing_ints]
                slicing_ints[2] = 100 - slicing_ints[0] - slicing_ints[1]
            # Ensure sleep sum to 7
            total_sleep = sum(sleep_ints)
            if total_sleep != 7:
                sleep_ints = [int(round(x * 7 / total_sleep)) for x in sleep_ints]
                sleep_ints[2] = 7 - sleep_ints[0] - sleep_ints[1]
            while True:
                try:
                    with open(file_name, 'rb+') as file:
                        data = file.read(28)
                        if len(data) < 28:
                            continue
                        numbers = struct.unpack('iiiiiii', data)
                        if numbers[6] == 1:
                            file.seek(-28, os.SEEK_CUR)
                            new_numbers = tuple(slicing_ints + sleep_ints + [flag])
                            file.write(struct.pack('iiiiiii', *new_numbers))
                            break
                        else:
                            continue
                except FileNotFoundError:
                    # If file does not exist, initialize it as ready (flag=1)
                    with open(file_name, 'wb') as file:
                        init_numbers = (40, 30, 30, 4, 0, 3, 1)
                        file.write(struct.pack('iiiiiii', *init_numbers))
                    continue

    def step(self, group_actions):
        """
        For each group action: (1) wait for slice_ctrl_{g}.bin flag==1, (2) after flag==1, 
        read the current KPM and MAC data from files, (3) calculate the reward for that action. 
        After all actions, return the last state and the group of rewards.
        """
        self.send_group_actions(group_actions, flag=0)
        group_rewards = []
        
        # Get initial state before any actions
        initial_state = self.get_all_state()
        initial_throughput = [self.RRU_ThpDL_UE[ue_idx] for ue_idx in range(self.user_num)] # kbps
        initial_delay = [self.DRB_Delay[ue_idx]/1000 for ue_idx in range(self.user_num)] # us -> ms
        
        print(f"DEBUG: Initial throughput: {[f'{t:.1f}' for t in initial_throughput]}")
        print(f"DEBUG: Initial delay: {[f'{d:.1f}' for d in initial_delay]}")
        
        for g, (slicing_action, sleep_action) in enumerate(group_actions):
            file_name = f'../trandata/slice_ctrl_{g}.bin'
            
            # Wait for this specific action to be processed (flag check)
            while True:
                try:
                    with open(file_name, 'rb') as file:
                        data = file.read(28)
                        if len(data) < 28:
                            continue
                        numbers = struct.unpack('iiiiiii', data)
                        if numbers[6] == 1:  # Action has been processed
                            break
                        else:
                            time.sleep(0.1)  # Wait a bit before checking again
                except FileNotFoundError:
                    time.sleep(0.1)
                    continue
            
            # Now read the current environment state after this action was applied
            current_state = self.get_all_state()
            current_throughput = [self.RRU_ThpDL_UE[ue_idx] for ue_idx in range(self.user_num)]
            current_delay = [self.DRB_Delay[ue_idx] for ue_idx in range(self.user_num)]
            
            print(f"DEBUG: Action {g} - Slicing: {slicing_action}, Sleep: {sleep_action}")
            print(f"DEBUG: Action {g} - Current throughput: {[f'{t:.1f}' for t in current_throughput]}")
            print(f"DEBUG: Action {g} - Current delay: {[f'{d:.1f}' for d in current_delay]}")
            
            # Check if environment data actually changed
            throughput_changed = not np.allclose(initial_throughput, current_throughput, atol=0.1)
            delay_changed = not np.allclose(initial_delay, current_delay, atol=0.1)
            print(f"DEBUG: Action {g} - Throughput changed: {throughput_changed}, Delay changed: {delay_changed}")
            
            # Calculate reward based on the actual environment state after this action
            reward = self.calculate_reward(sleep_action, current_throughput, current_delay)
            
            group_rewards.append(reward)
        
        # Get the final state after all actions
        final_state = self.get_all_state()
        
        return final_state, group_rewards, False, {}

    def calculate_reward(self, sleep_action, current_throughput, current_delay):
        """
        Calculate reward based on decoupled energy reward and penalties:
        r_t = energy_reward - throughput_penalty - delay_penalty
        energy_reward = lambda_sleep * (T_sleep / T_max) + lambda_thp * (sum_k UEThpDL_k / K)
        """
        # Get lambdas from config
        lambda_sleep = self.config.ENV.get('lambda_sleep', 1.0)
        lambda_thp = self.config.ENV.get('lambda_thp', 1.0)
        lambda_p = self.lambda_p
        lambda_d = self.lambda_d
        qos_targets = self.qos_targets
        # Sleep action
        a_t, b_t, c_t = sleep_action
        T_sleep = b_t
        K = len(current_throughput)
        # Energy reward (decoupled)
        sleep_term = lambda_sleep * T_sleep
        throughput_term = lambda_thp * (sum(current_throughput) / K if K > 0 else 0.0)
        energy_reward = sleep_term + throughput_term
        # Throughput penalty (QoS violations, only if DRB_pdcpSduVolumeDL > 0)
        penalty_p = 0.0
        for s_idx, slice_name in enumerate(['embb', 'urllc', 'mmtc']):
            P_s = qos_targets[s_idx]['throughput']
            users_in_slice = [i for i in range(self.user_num) if self.get_slice_type(i+1) == slice_name]
            for k in users_in_slice:
                p_tk = current_throughput[k]
                drb_pdcp_volume_dl = self.UE_KPM[k][0] 
                if drb_pdcp_volume_dl > 0:
                    if p_tk < P_s:  # QoS violation
                        violation = (P_s - p_tk) / P_s
                        penalty_p += violation
                        print(f"DEBUG: UE {k} ({slice_name}) throughput violation: {p_tk:.1f} < {P_s:.1f}, penalty: {violation:.3f} (DRB_pdcpSduVolumeDL: {drb_pdcp_volume_dl:.1f})")
                else:
                    print(f"DEBUG: UE {k} ({slice_name}) no throughput penalty - DRB_pdcpSduVolumeDL: {drb_pdcp_volume_dl:.1f} <= 0")
        
        throughput_penalty = lambda_p * penalty_p
        # Delay penalty (QoS violations)
        penalty_d = 0.0
        for s_idx, slice_name in enumerate(['embb', 'urllc', 'mmtc']):
            D_s = qos_targets[s_idx]['delay']  # Required delay for this slice
            users_in_slice = [i for i in range(self.user_num) if self.get_slice_type(i+1) == slice_name]
            
            for k in users_in_slice:
                d_tk = current_delay[k]
                if d_tk > D_s:
                    violation = (d_tk - D_s) / D_s
                    penalty_d += violation
                    print(f"DEBUG: UE {k} ({slice_name}) delay violation: {d_tk:.1f} > {D_s:.1f}, penalty: {violation:.3f}")
        
        delay_penalty = lambda_d * penalty_d
        
        # Calculate Final Reward
        reward = energy_reward - throughput_penalty - delay_penalty 
        
        print(f"DEBUG: Reward components - energy: {energy_reward:.3f}, thp_penalty: {throughput_penalty:.3f}, delay_penalty: {delay_penalty:.3f}")
        print(f"DEBUG: Final reward: {reward:.3f}")
        
        return reward

    
    def reset(self, group_size=1):
        """
        Reset the environment by writing random valid actions to all group control files.
        Slicing actions sum to 100, sleep actions sum to 7. All files set to flag=1 (ready).
        Only call this once at the beginning of training.
        Returns the initial state.
        """

        for g in range(group_size):
            slicing = np.random.dirichlet(np.ones(3)) * 100
            slicing_ints = [int(round(x)) for x in slicing]
            slicing_ints[2] = 100 - slicing_ints[0] - slicing_ints[1]
            sleep = np.random.dirichlet(np.ones(3)) * 7
            sleep_ints = [int(round(x)) for x in sleep]
            sleep_ints[2] = 7 - sleep_ints[0] - sleep_ints[1]
            flag = 1
            file_name = f'../trandata/slice_ctrl_{g}.bin'
            with open(file_name, 'wb') as f:
                f.write(struct.pack('iiiiiii', *(slicing_ints + sleep_ints + [flag])))
        
        # Return initial state
        return self.get_all_state()

    def init_control_files(self, group_size=1):
        """
        Initialize all group control files as ready (flag=1).
        """
        import struct
        for g in range(group_size):
            file_name = f'../trandata/slice_ctrl_{g}.bin'
            init_numbers = (40, 30, 30, 4, 0, 3, 1)
            with open(file_name, 'wb') as f:
                f.write(struct.pack('iiiiiii', *init_numbers))

    def close(self):
        """Clean up resources"""
        # No database connections to close since we create/close per call
        pass
    
    def get_user_slice_ids(self):
        """
        Return a torch tensor of slice ids (0,1,2,...) for each UE, matching the order of get_all_state().
        """
        # Map slice names to indices
        slice_type_to_idx = {'embb': 0, 'urllc': 1, 'mmtc': 2}
        user_slice_ids = []
        for ue_id in range(1, self.user_num + 1):
            slice_name = self.get_slice_type(ue_id)
            user_slice_ids.append(slice_type_to_idx[slice_name])
        return torch.tensor(user_slice_ids, dtype=torch.long)

    def get_slice_id_map(self):
        """
        Return a dict mapping slice names to indices.
        """
        return {'embb': 0, 'urllc': 1, 'mmtc': 2}

    def get_normalized_qos_targets(self):
        """
        Return QoS targets directly from config without normalization.
        Returns: [num_slices, 2] tensor with [throughput, delay] for each slice.
        """
        qos_targets = []
        for slice_target in self.qos_targets:
            # Use QoS targets directly from config without normalization
            qos_targets.append([slice_target['throughput'], slice_target['delay']])
        
        return torch.tensor(qos_targets, dtype=torch.float32)

    def save_normalized_metrics(self, all_states, epoch, save_dir="sgrpo_training_plots"):
        """
        Save all normalized MAC and KPM metrics for all UEs to a CSV file for the given epoch.
        all_states: list of [num_ues, num_features] normalized state vectors
        epoch: current epoch number
        """
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"normalized_metrics_epoch_{epoch}.csv")
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            header = [f"feature_{i+1}" for i in range(len(all_states[0]))]
            writer.writerow(["UE_ID"] + header)
            # Write data
            for ue_id, state in enumerate(all_states, 1):
                writer.writerow([ue_id] + list(state))
        print(f"Saved normalized metrics for epoch {epoch} to {filename}")