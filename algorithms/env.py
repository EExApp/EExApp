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
        ee_max_values = [config.ACTION_SPACE['ee']['max']] * config.ACTOR_CRITIC['ee_action_dim']
        self.ee_action_space = spaces.MultiDiscrete(ee_max_values)
        
        # NS action space: continuous actions for network slicing (slice1, slice2, slice3)
        self.ns_action_space = spaces.Box(
            low=np.array([config.ACTION_SPACE['ns']['min']] * config.ACTOR_CRITIC['ns_action_dim']),
            high=np.array([config.ACTION_SPACE['ns']['max']] * config.ACTOR_CRITIC['ns_action_dim']),
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
        '''ue_id, DRB_pdcpSduVolumeDL, DRB_pdcpSduVolumeUL, DRB_RlcSduDelayDL, DRB_UEThpDL, DRB_UEThpUL, RRU_PrbTotDL, RRU_PrbTotUL '''
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
        #DRB_RlcSduDelayDL, 0-1000    #there will be a peak when states is changing 
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
        """Get state vectors for all UEs"""
        print("Starting get_all_state...")
        self.UE_MAC = []
        self.UE_KPM = []
        self.RRU_ThpDL_UE = []
        self.DRB_Delay = []
        self.BLER = []
        self.RRU_PrbTotDL = []
        
        # Check if control file exists, if not initialize it
        if not os.path.exists(self.control_file):
            print(f"Control file not found, initializing: {self.control_file}")
            self.init_control_file()
        
        print("Waiting for control file to be ready...")

        while True:
            with open(self.control_file, 'rb+') as file:
                data = file.read(28)  # Read 7 integers (4 bytes each)
                if len(data) < 28:
                    print(f"Control file too short: {len(data)} bytes")
                    continue
                
                numbers = struct.unpack('iiiiiii', data)
                print(f"Control file numbers: {numbers}")
                
                if numbers[6] == 1:  # Check control flag - 1 means control has been applied
                    print("Control file ready, collecting state data...")
                    print(f"Processing {self.user_num} UEs...")
                    
                    for i in range(1, self.user_num + 1):
                        print(f"\n--- Processing UE {i} ---")
                        complete_state, ue_mac_i, ue_kpm_i = self.get_state(i)
                        
                        print(f"UE {i} - MAC size: {len(ue_mac_i)}, KPM size: {len(ue_kpm_i)}, Complete size: {len(complete_state)}")
                        
                        self.UE_MAC.append(ue_mac_i)
                        self.UE_KPM.append(ue_kpm_i)
                        self.BLER.append(ue_mac_i[8])  # dl_bler
                        self.DRB_Delay.append(ue_kpm_i[2])  # rlc_sdu_delay_dl (index 2 in normalized)
                        self.RRU_ThpDL_UE.append(ue_kpm_i[3])  # ue_thp_dl (index 3 in normalized)
                        self.RRU_PrbTotDL.append(ue_kpm_i[5])  # prb_tot_dl (index 5 in normalized)
                        
                        print(f"UE {i} processed successfully")
                    
                    print(f"\nAll UEs processed. Collected data for {len(self.UE_MAC)} UEs")
                    break
                    
                else:
                    print(f"Control file not ready, flag = {numbers[6]}")
                    continue
                
        # Return complete state vectors (already normalized, 17 features each)
        complete_states = []
        for i in range(len(self.UE_MAC)):
            # The complete_state is already the concatenated normalized MAC + KPM
            state_vector = np.array(self.UE_MAC[i] + self.UE_KPM[i], dtype=np.float32)
            print(f"Final state vector {i} size: {len(state_vector)}")
            complete_states.append(state_vector)
        
        print(f"Successfully collected state data for {len(complete_states)} UEs")
        return complete_states
    

    def init_control_file(self):
        """Initialize the control file with default values"""
        # Write initial values in binary format
        with open(self.control_file, 'wb') as f:
            # Default values: slice1=40, slice2=30, slice3=30, a_t=4, b_t=0, c_t=3, flag=0
            numbers = (40, 30, 30, 4, 0, 3, 1)
            f.write(struct.pack('iiiiiii', *numbers))


    def send_action(self, slice1, slice2, slice3, a_t, b_t, c_t, flag):
        """Send action to the control file in binary format"""
        while True:
            # NS actions (slice1, slice2, slice3) are continuous floats (percentages)
            # Convert to integers for binary storage (round to nearest integer)
            slice1 = int(round(slice1))
            slice2 = int(round(slice2))
            slice3 = int(round(slice3))
            
            # EE actions (a_t, b_t, c_t) are discrete integers
            a_t = int(a_t)
            b_t = int(b_t)
            c_t = int(c_t)
            flag = int(flag)
            
            # Validate slice percentages sum to 100
            total_slice = slice1 + slice2 + slice3
            if total_slice != 100:
                # Normalize to sum to 100
                slice1 = int(round(slice1 * 100 / total_slice))
                slice2 = int(round(slice2 * 100 / total_slice))
                slice3 = 100 - slice1 - slice2  # Ensure exact sum
            
            # Write action with flag=0 to indicate control needed
            with open(self.control_file, 'rb+') as file:
                data = file.read(28)    
                if len(data) < 28:
                    continue
                
                numbers = struct.unpack('iiiiiii', data)
                if numbers[6] == 1:
                    file.seek(-28, os.SEEK_CUR)
                    new_numbers = (slice1, slice2, slice3, a_t, b_t, c_t, flag)
                    file.write(struct.pack('iiiiiii', *new_numbers))
                    break
                else:
                    continue

    
    def calculate_decomposed_rewards(self, action, user_slice_ids):
        """
        Calculate decomposed rewards for EE and NS actors based on the provided formulas.
        Args:
            action: The action taken (should include b_t for EE reward)
            user_slice_ids: List of slice types for each UE (e.g., ['embb', 'urllc', ...])
        Returns:
            reward_ee: Energy efficiency reward
            reward_ns: QoS penalty-based reward
        """
        # --- Energy Efficiency Reward ---
        # b_t is the second value in the action (assuming [slice1, slice2, slice3, a_t, b_t, c_t])
        b_t = action[4]  # index 4 is b_t
        N_sf = self.N_sf
        reward_ee = b_t / N_sf

        # --- QoS Reward (Penalty-based) ---
        lambda_p = self.lambda_p
        lambda_d = self.lambda_d
        qos_targets = self.qos_targets  # List of dicts with 'throughput' and 'delay'

        # Map slice names to indices
        slice_type_to_idx = {'embb': 0, 'urllc': 1, 'mmtc': 2}

        penalty_p = 0.0
        penalty_d = 0.0

        # For each slice
        for s_idx, slice_name in enumerate(['embb', 'urllc', 'mmtc']):
            P_s = qos_targets[s_idx]['throughput']
            D_s = qos_targets[s_idx]['delay']
            # Find users in this slice
            users_in_slice = [i for i, st in enumerate(user_slice_ids) if st == slice_name]
            
            # Check if there are users in this slice
            if len(users_in_slice) == 0:
                # No users in this slice, skip penalty calculation
                continue
                
            for k in users_in_slice:
                # Validate index bounds
                if k >= len(self.RRU_ThpDL_UE) or k >= len(self.DRB_Delay):
                    print(f"Warning: User index {k} out of bounds for {slice_name} slice")
                    continue
                    
                # Throughput and delay from KPM data
                p_tk = self.RRU_ThpDL_UE[k]  # throughput of user k
                d_tk = self.DRB_Delay[k]     # delay of user k
                penalty_p += max(0, (P_s - p_tk) / P_s)
                penalty_d += max(0, (d_tk - D_s) / D_s)

        reward_ns = -lambda_p * penalty_p - lambda_d * penalty_d

        return reward_ee, reward_ns

    def reset(self):
        """Reset the environment to initial state"""
        # Initialize last known states
        self.last_mac_state = [0]*10  # Single list for the last MAC state
        self.last_kpm_states = [ [0]*8 for _ in range(self.user_num) ]
        
        # Get initial state
        self.state = self.get_all_state()
        
        return self.state
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Parse action
        slice1, slice2, slice3 = action[:3]  # NS actions (continuous floats)
        a_t, b_t, c_t = action[3:6]          # EE actions (discrete integers)
        
        # Debug: Print action details - ensure EE actions are shown as integers
        print(f"Action received: NS=[{slice1:.2f}, {slice2:.2f}, {slice3:.2f}], EE=[{int(a_t)}, {int(b_t)}, {int(c_t)}]")
        print(f"NS sum: {slice1 + slice2 + slice3:.2f}")
        
        # Send control actions
        self.send_action(slice1, slice2, slice3, a_t, b_t, c_t, 0)
        
        # Get new state
        self.state = self.get_all_state()
        
        # Calculate decomposed rewards
        reward_ee, reward_ns = self.calculate_decomposed_rewards(action, [self.get_slice_type(ue_id) for ue_id in range(1, self.user_num + 1)])
        
        # Check if episode is done (e.g., after certain number of steps)
        done = False
        
        return self.state, (reward_ee, reward_ns), done, {}
    
    def close(self):
        """Clean up resources"""
        # No database connections to close since we create/close per call
        pass
    
    def get_user_slice_ids(self):
        """
        Returns a list of slice indices (0, 1, 2, ...) for all users, mapping from the string returned by get_slice_type(ue_id).
        """
        slice_name_to_idx = {'embb': 0, 'urllc': 1, 'mmtc': 2}
        return [slice_name_to_idx[self.get_slice_type(ue_id)] for ue_id in range(1, self.user_num + 1)]