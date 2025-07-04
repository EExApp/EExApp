import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config, BATCH_SIZE
from state_encoder import TransformerStateEncoder
import random

def project_to_simplex(x, s=1.0):
    # Projects x onto the simplex {x | x >= 0, sum(x) = s}
    # x: [num_slices]
    sorted_x, _ = torch.sort(x, descending=True)
    tmpsum = 0.0
    for i in range(len(x)):
        tmpsum += sorted_x[i]
        t = (tmpsum - s) / (i + 1)
        if i == len(x) - 1 or sorted_x[i+1] <= t:
            break
    theta = t
    return torch.clamp(x - theta, min=0.0)

class SGRPOPolicy(nn.Module):
    """
    Joint policy network for SGRPO: Transformer encoder + slicing/sleep heads.
    Handles dynamic number of UEs and slices.
    Slicing head: 3 continuous actions (sum=100, Gaussian)
    Sleep head: 3 discrete actions (sum=7, categorical)
    """
    def __init__(self):
        super().__init__()
        self.encoder = TransformerStateEncoder()
        self.hidden_dim = config.STATE_ENCODER['hidden_dim']
        self.num_slices = config.ENV['num_slices']
        self.slicing_action_dim = config.POLICY['slicing_action_dim']
        self.sleep_action_dim = config.POLICY['sleep_action_dim']
        self.qos_dim = 2  # throughput, delay per slice

        # Learnable queries for attention pooling per slice
        self.slice_queries = nn.Parameter(torch.randn(self.num_slices, self.hidden_dim))
        self.slice_attn = nn.MultiheadAttention(self.hidden_dim, config.STATE_ENCODER['num_heads'], batch_first=True)

        # Slicing head: MLP after attention pooling + QoS targets
        # Input dimension: hidden_dim + qos_dim (64 + 2 = 66)
        self.slicing_head_mean = nn.Sequential(
            nn.Linear(self.hidden_dim + self.qos_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.slicing_head_logstd = nn.Parameter(torch.zeros(self.num_slices))

        # Sleep control head: global pooling + MLP with categorical output
        self.sleep_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.sleep_action_dim)
        )

    def forward(self, ue_states, ue_slice_ids=None, qos_targets=None):
        """
        ue_states: [BATCH_SIZE, num_ues, num_features]
        ue_slice_ids: [num_ues] (ints 0..num_slices-1)
        qos_targets: [num_slices, 2] (throughput, delay for each slice)
        Returns:
            slicing_means: [num_slices]
            slicing_logstd: [num_slices]
            sleep_logits: [sleep_action_dim]
        """
        batch_size, num_ues, _ = ue_states.shape
        assert batch_size == BATCH_SIZE, f"Batch size must be {BATCH_SIZE}"
        encoded = self.encoder(ue_states)  # [BATCH_SIZE, num_ues, hidden_dim]
        encoded = encoded.squeeze(0)  # [num_ues, hidden_dim]
        
        # Always use attention pooling for each slice
        slice_reps = []
        for s in range(self.num_slices):
            query = self.slice_queries[s].unsqueeze(0).unsqueeze(0)  # [1,1,hidden_dim]
            attn_out, _ = self.slice_attn(query, encoded.unsqueeze(0), encoded.unsqueeze(0))
            slice_context = attn_out.squeeze(0).squeeze(0)  # [hidden_dim]
            
            # Concatenate normalized QoS targets for this slice
            if qos_targets is not None:
                # Ensure QoS targets are on the same device as slice_context
                qos_target_slice = qos_targets[s].to(device=slice_context.device, dtype=slice_context.dtype)
                slice_input = torch.cat([slice_context, qos_target_slice], dim=-1)  # [hidden_dim + qos_dim]
            else:
                # If no QoS targets provided, use zeros
                qos_zeros = torch.zeros(2, device=slice_context.device, dtype=slice_context.dtype)
                slice_input = torch.cat([slice_context, qos_zeros], dim=-1)  # [hidden_dim + qos_dim]
            
            slice_reps.append(slice_input)
        
        slice_reps = torch.stack(slice_reps, dim=0)  # [num_slices, hidden_dim + qos_dim]
        slicing_means = self.slicing_head_mean(slice_reps).squeeze(-1)  # [num_slices]
        slicing_logstd = self.slicing_head_logstd  # [num_slices]
        
        # Sleep head: global context pooling
        global_context = encoded.mean(dim=0)
        sleep_logits = self.sleep_head(global_context)  # [sleep_action_dim]
        
        return slicing_means, slicing_logstd, sleep_logits

    def sample_actions(self, ue_states, ue_slice_ids=None, qos_targets=None):
        slicing_means, slicing_logstd, sleep_logits = self.forward(ue_states, ue_slice_ids, qos_targets)
        slicing_std = torch.exp(slicing_logstd)
        slicing_dist = torch.distributions.Normal(slicing_means, slicing_std)
        slicing_action = slicing_dist.rsample()  # [num_slices]
        slicing_action = project_to_simplex(slicing_action, s=100.0)  # sum=100
        # Sleep head: categorical distribution for 3 discrete actions (sum=7)
        temperature = 1.2
        sleep_logits_scaled = sleep_logits / temperature
        sleep_probs = F.softmax(sleep_logits_scaled, dim=-1)
        # Sample 7 slots according to categorical distribution
        sleep_action = torch.multinomial(sleep_probs, 7, replacement=True)
        # Count occurrences for each slot type
        sleep_counts = torch.bincount(sleep_action, minlength=3)
        sleep_action = sleep_counts  # [a_t, b_t, c_t], sum=7
        return slicing_action, sleep_action

    def log_prob(self, ue_states, slicing_action, sleep_action, ue_slice_ids=None, qos_targets=None):
        slicing_means, slicing_logstd, sleep_logits = self.forward(ue_states, ue_slice_ids, qos_targets)
        slicing_std = torch.exp(slicing_logstd)
        slicing_dist = torch.distributions.Normal(slicing_means, slicing_std)
        slicing_logp = slicing_dist.log_prob(slicing_action).sum()
        # Sleep: categorical log-prob
        temperature = 1.2
        sleep_logits_scaled = sleep_logits / temperature
        sleep_probs = F.softmax(sleep_logits_scaled, dim=-1)
        sleep_dist = torch.distributions.Categorical(probs=sleep_probs)
        # Log-prob for the sleep action (multinomial)
        sleep_logp = torch.lgamma(torch.tensor(7+1.0)) - torch.lgamma(sleep_action.float()+1.0).sum() + (sleep_action * torch.log(sleep_probs)).sum()
        return slicing_logp, sleep_logp 