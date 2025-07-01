import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config, BATCH_SIZE
from state_encoder import TransformerStateEncoder

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
    """
    def __init__(self):
        super().__init__()
        self.encoder = TransformerStateEncoder()
        self.hidden_dim = config.STATE_ENCODER['hidden_dim']
        self.num_slices = config.ENV['num_slices']
        self.slicing_action_dim = config.POLICY['slicing_action_dim']
        self.sleep_action_dim = config.POLICY['sleep_action_dim']

        # Slice-aware attention pooling: learnable query per slice
        self.slice_queries = nn.Parameter(torch.randn(self.num_slices, self.hidden_dim))
        self.slice_attn = nn.MultiheadAttention(self.hidden_dim, config.STATE_ENCODER['num_heads'], batch_first=True)
        # Slicing head: outputs mean and log_std for each slice
        self.slicing_head_mean = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.slicing_head_logstd = nn.Parameter(torch.zeros(self.num_slices))

        # Global pooling for sleep head
        self.sleep_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.sleep_action_dim)
        )

    def forward(self, ue_states, ue_slice_ids=None):
        """
        ue_states: [BATCH_SIZE, num_ues, num_features]
        ue_slice_ids: [num_ues] (ints 0..num_slices-1)
        Returns:
            slicing_means: [num_slices]
            slicing_logstd: [num_slices]
            sleep_logits: [sleep_action_dim]
        """
        batch_size, num_ues, _ = ue_states.shape
        assert batch_size == BATCH_SIZE, f"Batch size must be {BATCH_SIZE}"
        encoded = self.encoder(ue_states)  # [BATCH_SIZE, num_ues, hidden_dim]
        encoded = encoded.squeeze(0)  # [num_ues, hidden_dim]
        # Slice-aware pooling: for each slice, pool UEs belonging to that slice
        slice_reps = []
        for s in range(self.num_slices):
            if ue_slice_ids is not None:
                mask = (ue_slice_ids == s)
                if mask.sum() == 0:
                    pooled = torch.zeros(self.hidden_dim, device=ue_states.device)
                else:
                    pooled = encoded[mask].mean(dim=0)
            else:
                # fallback: use attention pooling
                query = self.slice_queries[s].unsqueeze(0).unsqueeze(0)  # [1,1,hidden_dim]
                attn_out, _ = self.slice_attn(query, encoded.unsqueeze(0), encoded.unsqueeze(0))
                pooled = attn_out.squeeze(0).squeeze(0)
            slice_reps.append(pooled)
        slice_reps = torch.stack(slice_reps, dim=0)  # [num_slices, hidden_dim]
        slicing_means = self.slicing_head_mean(slice_reps).squeeze(-1)  # [num_slices]
        slicing_logstd = self.slicing_head_logstd  # [num_slices]
        # Sleep head: global context pooling
        global_context = encoded.mean(dim=0)
        sleep_logits = self.sleep_head(global_context)  # [sleep_action_dim]
        return slicing_means, slicing_logstd, sleep_logits

    def sample_actions(self, ue_states, ue_slice_ids=None):
        slicing_means, slicing_logstd, sleep_logits = self.forward(ue_states, ue_slice_ids)
        slicing_std = torch.exp(slicing_logstd)
        slicing_dist = torch.distributions.Normal(slicing_means, slicing_std)
        slicing_action = slicing_dist.rsample()  # [num_slices]
        slicing_action = project_to_simplex(slicing_action, s=100.0)  # sum=100
        # Sleep: categorical, project to sum=7
        sleep_logits = sleep_logits
        sleep_probs = F.softmax(sleep_logits, dim=-1)
        sleep_action = (sleep_probs * 7).round().long()  # [sleep_action_dim], sum may not be exactly 7
        # Project/adjust to sum=7
        diff = 7 - sleep_action.sum().item()
        if diff != 0:
            # Adjust the largest/smallest element
            idx = torch.argmax(sleep_action) if diff < 0 else torch.argmin(sleep_action)
            sleep_action[idx] += diff
        return slicing_action, sleep_action

    def log_prob(self, ue_states, slicing_action, sleep_action, ue_slice_ids=None):
        slicing_means, slicing_logstd, sleep_logits = self.forward(ue_states, ue_slice_ids)
        slicing_std = torch.exp(slicing_logstd)
        slicing_dist = torch.distributions.Normal(slicing_means, slicing_std)
        slicing_logp = slicing_dist.log_prob(slicing_action).sum()
        # Sleep: categorical log-prob
        sleep_probs = F.softmax(sleep_logits, dim=-1)
        sleep_dist = torch.distributions.Categorical(probs=sleep_probs)
        # For multi-slot, sum log-probs (approximate)
        sleep_logp = torch.sum(torch.log(sleep_probs + 1e-8) * sleep_action.float())
        return slicing_logp, sleep_logp 