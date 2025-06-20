import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any
from config import config


class StateEncoder(nn.Module):
    """
    GRU-based state encoder for processing multi-UE states
    
    Already-normalized 17-dim vectors per UE (10 MAC + 7 KPM). 
    Pads input to max_ues, projects to hidden_dim, processes with GRU, and returns final hidden state.

    - MAC metrics (10):
        - dl_curr_tbs: Downlink current TBS
        - dl_sched_rb: Downlink scheduled RBs
        - pusch_snr: PUSCH SNR
        - pucch_snr: PUCCH SNR
        - wb_cqi: Wideband CQI
        - dl_mcs1: Downlink MCS
        - ul_mcs1: Uplink MCS
        - phr: Power Headroom
        - dl_bler: Downlink BLER
        - ul_bler: Uplink BLER
    - KPM metrics (7):
        - PdcpSduVolumeDL: PDCP SDU Volume DL
        - PdcpSduVolumeUL: PDCP SDU Volume UL
        - RlcSduDelayDl: RLC SDU Delay DL    - delay calculation
        - UEThpDl: UE Throughput DL          - throughput calculation   
        - UEThpUL: UE Throughput UL
        - PrbTotDl: Total PRB DL
        - PrbTotUl: Total PRB UL
    This module processes the state information from multiple UEs using a GRU
    to capture temporal dependencies and produce a fixed-size representation.
    """
    def __init__(self):
        super().__init__()
        self.max_ues = config.ENV['max_ues']
        self.num_features = config.STATE_ENCODER['num_features']
        self.hidden_dim = config.STATE_ENCODER['hidden_dim']
        self.num_layers = config.STATE_ENCODER['num_layers']
        self.input_proj = nn.Linear(self.num_features, self.hidden_dim)
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
    def forward(self, ue_list: list) -> torch.Tensor:
        """
        Args:
            ue_list: list of user_num UE feature vectors, each of shape [num_features]
        Returns:
            Encoded state tensor of shape [hidden_dim]
        """
        try:
            user_num = len(ue_list)
            
            # Handle empty list case
            if user_num == 0:
                # Return zero tensor with correct shape
                return torch.zeros(self.hidden_dim, dtype=torch.float32)
            
            # Ensure all elements are tensors and get device safely
            device = 'cpu'  # Default device
            if user_num > 0:
                first_element = ue_list[0]
                if isinstance(first_element, torch.Tensor):
                    device = first_element.device
                elif hasattr(first_element, 'device'):
                    device = first_element.device
            
            # Pad to max_ues
            if user_num < self.max_ues:
                pad = [torch.zeros(self.num_features, device=device) for _ in range(self.max_ues - user_num)]
                ue_list = ue_list + pad
            elif user_num > self.max_ues:
                ue_list = ue_list[:self.max_ues]
            
            # Stack to tensor [max_ues, num_features]
            x = torch.stack(ue_list, dim=0)
            
            # Linear projection [max_ues, hidden_dim]
            x_proj = self.input_proj(x)
            
            # Add batch dimension for GRU: [1, max_ues, hidden_dim]
            x_proj = x_proj.unsqueeze(0)
            
            # GRU processing
            _, h_n = self.gru(x_proj)  # h_n: [num_layers, 1, hidden_dim]
            
            # Use the last layer's final hidden state
            encoded = h_n[-1, 0, :]  # [hidden_dim]
            return encoded
            
        except Exception as e:
            print(f"Error in StateEncoder.forward: {e}")
            print(f"ue_list type: {type(ue_list)}, length: {len(ue_list) if isinstance(ue_list, list) else 'not list'}")
            if isinstance(ue_list, list) and len(ue_list) > 0:
                print(f"First element type: {type(ue_list[0])}")
                if hasattr(ue_list[0], 'shape'):
                    print(f"First element shape: {ue_list[0].shape}")
            # Return zero tensor as fallback
            return torch.zeros(self.hidden_dim, dtype=torch.float32)



    