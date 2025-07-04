import torch
import numpy as np

def group_normalize(rewards):
    """
    Compute group mean and std for rewards, return normalized advantages.
    Args:
        rewards: [num_ues] tensor or np.array
    Returns:
        mean, std, normalized_advantages: all tensors
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    
    # Clip extreme rewards to prevent numerical instability
    rewards = torch.clamp(rewards, min=-100.0, max=100.0)
    
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    
    # Handle zero variance case to prevent NaN advantages
    if std < 1e-8:
        # Add small noise to break symmetry and enable learning
        noise_std = 0.1 * torch.abs(mean) if torch.abs(mean) > 1e-8 else 0.1
        noise = torch.randn_like(rewards) * noise_std
        rewards = rewards + noise
        mean = rewards.mean()
        std = rewards.std(unbiased=False) + 1e-8
        print(f"Warning: Zero variance detected, added noise (std={noise_std:.6f})")
    
    # Ensure minimum std to prevent division by very small numbers
    std = torch.clamp(std, min=1e-6)
    
    advantages = (rewards - mean) / std
    return mean, std, advantages

def log_prob_ratio(new_logp, old_logp):
    """
    Compute log-prob ratio for PPO/GRPO style update.
    Args:
        new_logp: [num_ues] tensor
        old_logp: [num_ues] tensor
    Returns:
        ratio: [num_ues] tensor
    """
    return torch.exp(new_logp - old_logp)

def kl_divergence(pi_ref_logp, pi_logp):
    """
    KL divergence estimator as in GRPO (Schulman 2020):
    D_KL[pi||pi_ref] = exp(pi_ref_logp - pi_logp) - (pi_ref_logp - pi_logp) - 1
    Args:
        pi_ref_logp: [num_ues] tensor (log prob under reference policy)
        pi_logp: [num_ues] tensor (log prob under current policy)
    Returns:
        kl: [num_ues] tensor
    """
    ratio = torch.exp(pi_ref_logp - pi_logp)
    kl = ratio - (pi_ref_logp - pi_logp) - 1
    return kl 