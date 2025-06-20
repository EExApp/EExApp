import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from state_encoder import StateEncoder
from config import config

'''
Helper functions
'''
def combined_shape(length, shape=None):
    """Helper function for buffer initialization"""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    Magic from rllab for computing discounted cumulative sums of vectors.
    
    input: 
        vector x: [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Build a multi-layer perceptron in PyTorch.
    Args:
        sizes: List of layer sizes
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError
    
    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class ConstrainedCategoricalActor(Actor):
    """
    Categorical actor for EE actions with sum constraint (a_t + b_t + c_t = 7)
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # Output logits for each action dimension
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        # Create categorical distributions for each action
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
    def sample_constrained(self, obs):
        """
        Sample actions that satisfy the sum constraint a_t + b_t + c_t = 7
        """
        logits = self.logits_net(obs)
        
        # Sample actions sequentially to satisfy constraint
        batch_size = logits.shape[0]
        actions = torch.zeros(batch_size, 3, dtype=torch.long, device=logits.device)
        
        for b in range(batch_size):
            remaining = config.ACTION_SPACE['ee']['sum']  # 7
            
            # Sample a_t (0 to remaining)
            a_logits = logits[b, 0:remaining+1]  # Only allow 0 to remaining
            a_dist = Categorical(logits=a_logits)
            a_t = a_dist.sample()
            actions[b, 0] = a_t
            remaining -= a_t
            
            # Sample b_t (0 to remaining)
            b_logits = logits[b, 1:remaining+2]  # Shift by 1, allow 0 to remaining
            b_dist = Categorical(logits=b_logits)
            b_t = b_dist.sample()
            actions[b, 1] = b_t
            remaining -= b_t
            
            # c_t is determined by constraint
            actions[b, 2] = remaining
        
        return actions
    
    def log_prob_constrained(self, obs, actions):
        """
        Compute log probability of constrained actions
        """
        logits = self.logits_net(obs)
        batch_size = logits.shape[0]
        log_probs = torch.zeros(batch_size, device=logits.device)
        
        for b in range(batch_size):
            remaining = config.ACTION_SPACE['ee']['sum']
            
            # Log prob for a_t
            a_logits = logits[b, 0:remaining+1]
            a_dist = Categorical(logits=a_logits)
            a_t = actions[b, 0]  # Get first action element
            # Validate a_t is within valid range
            if a_t.item() >= 0 and a_t.item() <= remaining:
                log_probs[b] += a_dist.log_prob(a_t)
            else:
                # Invalid action, assign very low probability
                log_probs[b] += torch.tensor(-1e6, device=logits.device)
            remaining -= a_t.item()
            
            # Log prob for b_t
            if remaining >= 0:
                b_logits = logits[b, 1:remaining+2]
                b_dist = Categorical(logits=b_logits)
                b_t = actions[b, 1]  # Get second action element
                # Validate b_t is within valid range
                if b_t.item() >= 0 and b_t.item() <= remaining:
                    log_probs[b] += b_dist.log_prob(b_t)
                else:
                    # Invalid action, assign very low probability
                    log_probs[b] += torch.tensor(-1e6, device=logits.device)
                remaining -= b_t.item()
            else:
                # Invalid constraint state, assign very low probability
                log_probs[b] += torch.tensor(-1e6, device=logits.device)
            
            # c_t is deterministic given constraint
            # Validate c_t matches the constraint
            c_t = actions[b, 2]  # Get third action element
            if c_t.item() != remaining:
                # Constraint violation, assign very low probability
                log_probs[b] += torch.tensor(-1e6, device=logits.device)
        
        return log_probs

class ConstrainedGaussianActor(Actor):
    """
    Gaussian actor for NS actions with sum constraint (slice1 + slice2 + slice3 = 100)
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # Output means for each action dimension
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        # Learnable standard deviations
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
    
    def sample_constrained(self, obs):
        """
        Sample actions that satisfy the sum constraint slice1 + slice2 + slice3 = 100
        Using Dirichlet-like sampling approach
        """
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        
        # Sample from normal distributions
        normal_dist = Normal(mu, std)
        raw_samples = normal_dist.sample()
        
        # Apply softmax to get proportions that sum to 1
        proportions = F.softmax(raw_samples, dim=-1)
        
        # Scale to sum to 100
        actions = proportions * config.ACTION_SPACE['ns']['sum']
        
        # Clip to bounds
        actions = torch.clamp(actions, 
                             config.ACTION_SPACE['ns']['min'], 
                             config.ACTION_SPACE['ns']['max'])
        
        # Renormalize to ensure sum = 100
        action_sum = torch.sum(actions, dim=-1, keepdim=True)
        actions = actions / action_sum * config.ACTION_SPACE['ns']['sum']
        
        return actions
    
    def log_prob_constrained(self, obs, actions):
        """
        Compute log probability of constrained actions
        Note: This is an approximation since the exact distribution is complex
        """
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        
        # Compute log prob of the unconstrained samples
        normal_dist = Normal(mu, std)
        log_probs = normal_dist.log_prob(actions).sum(axis=-1)
        
        # Add correction for the constraint (simplified)
        # In practice, you might want to use more sophisticated methods
        return log_probs

class MLPCritic(nn.Module):
    
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)   # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=config.ACTOR_CRITIC['hidden_sizes'], activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
            
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = ConstrainedGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = ConstrainedCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            if isinstance(self.pi, ConstrainedCategoricalActor):
                a = self.pi.sample_constrained(obs)
                logp_a = self.pi.log_prob_constrained(obs, a)
            else:
                a = self.pi.sample_constrained(obs)
                logp_a = self.pi.log_prob_constrained(obs, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

class MCMA_ActorCritic(nn.Module):
    """
    Multi-Critic Multi-Actor Actor-Critic module for O-RAN PPO
    - Uses GRU-based StateEncoder for all input
    - ee_actor: constrained categorical actor for energy efficiency (a_t, b_t, c_t)
    - ns_actor: constrained gaussian actor for network slicing (slice1, slice2, slice3)
    - ee_critic: value function for EE
    - ns_critic: value function for NS
    """
    def __init__(self, observation_space, action_space, hidden_sizes=None, activation=nn.Tanh):
        super().__init__()
        obs_dim = config.STATE_ENCODER['hidden_dim']
        if hidden_sizes is None:
            hidden_sizes = config.ACTOR_CRITIC['hidden_sizes']
        # GRU-based state encoder
        self.state_encoder = StateEncoder()
        # EE actor: constrained discrete (categorical)
        self.ee_actor = ConstrainedCategoricalActor(obs_dim, config.ACTOR_CRITIC['ee_action_dim'], hidden_sizes, activation)
        # NS actor: constrained continuous (gaussian)
        self.ns_actor = ConstrainedGaussianActor(obs_dim, config.ACTOR_CRITIC['ns_action_dim'], hidden_sizes, activation)
        # EE critic
        self.ee_critic = MLPCritic(obs_dim, hidden_sizes, activation)
        # NS critic
        self.ns_critic = MLPCritic(obs_dim, hidden_sizes, activation)

    def encode_state(self, ue_state_list):
        # ue_state_list: list of [num_features] tensors for all UEs
        # Returns: [hidden_dim] tensor
        if isinstance(ue_state_list, np.ndarray):
            ue_state_list = [torch.tensor(x, dtype=torch.float32) for x in ue_state_list]
        elif isinstance(ue_state_list, list) and isinstance(ue_state_list[0], np.ndarray):
            ue_state_list = [torch.tensor(x, dtype=torch.float32) for x in ue_state_list]
        return self.state_encoder(ue_state_list)

    def step(self, ue_state_list):
        # Accepts raw multi-UE state, encodes, then acts/criticizes
        with torch.no_grad():
            encoded = self.encode_state(ue_state_list)
            # Add batch dimension
            encoded = encoded.unsqueeze(0)  # [1, hidden_dim]
            
            # EE actor (constrained discrete) - samples 3 actions (a_t, b_t, c_t)
            ee_a = self.ee_actor.sample_constrained(encoded)  # Shape: [1, 3] - 3 discrete actions
            ee_logp_a = self.ee_actor.log_prob_constrained(encoded, ee_a)
            ee_v = self.ee_critic(encoded)
            
            # NS actor (constrained continuous) - samples 3 actions (slice1, slice2, slice3)
            ns_a = self.ns_actor.sample_constrained(encoded)  # Shape: [1, 3] - 3 continuous actions
            ns_logp_a = self.ns_actor.log_prob_constrained(encoded, ns_a)
            ns_v = self.ns_critic(encoded)
            
        return {
            'encoded': encoded.squeeze(0),
            'ee_action': ee_a.squeeze(0).to(torch.long),    # Shape: [3] - (a_t, b_t, c_t) - discrete, ensure long dtype
            'ns_action': ns_a.squeeze(0),    # Shape: [3] - (slice1, slice2, slice3) - continuous
            'ee_value': ee_v.squeeze(0),
            'ns_value': ns_v.squeeze(0),
            'ee_logp': ee_logp_a.squeeze(0),
            'ns_logp': ns_logp_a.squeeze(0)
        }

    def get_log_prob(self, encoded, ee_action, ns_action):
        # For PPO update: get log-prob for both actors
        # encoded: [batch_size, hidden_dim]
        # ee_action: [batch_size, 3] - already in correct shape
        # ns_action: [batch_size, 3] - already in correct shape
        
        # No need to unsqueeze since actions are already in batch format
        ee_logp = self.ee_actor.log_prob_constrained(encoded, ee_action)
        ns_logp = self.ns_actor.log_prob_constrained(encoded, ns_action)
        return ee_logp, ns_logp

    def get_entropy(self, encoded):
        # For entropy regularization (approximate)
        ee_entropy = self.ee_actor._distribution(encoded).entropy().mean()
        ns_entropy = self.ns_actor._distribution(encoded).entropy().mean()
        return ee_entropy, ns_entropy

    def get_values(self, encoded):
        # For value loss
        ee_v = self.ee_critic(encoded)
        ns_v = self.ns_critic(encoded)
        return ee_v, ns_v

    def sample_actions(self, encoded):
        """
        Sample actions from both actors with proper constraint handling
        Args:
            encoded: Encoded state tensor
        Returns:
            Dictionary with actions, log probabilities, and values
        """
        # EE actor sampling (constrained discrete)
        ee_a = self.ee_actor.sample_constrained(encoded)
        ee_logp_a = self.ee_actor.log_prob_constrained(encoded, ee_a)
        ee_v = self.ee_critic(encoded)
        
        # NS actor sampling (constrained continuous)
        ns_a = self.ns_actor.sample_constrained(encoded)
        ns_logp_a = self.ns_actor.log_prob_constrained(encoded, ns_a)
        ns_v = self.ns_critic(encoded)
        
        return {
            'ee_action': ee_a,
            'ns_action': ns_a,
            'ee_value': ee_v,
            'ns_value': ns_v,
            'ee_logp': ee_logp_a,
            'ns_logp': ns_logp_a
        }




