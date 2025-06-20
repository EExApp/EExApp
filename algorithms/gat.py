import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from config import config


class MultiHeadGATLayer(nn.Module):
    """
    Multi-Head Graph Attention Network Layer
    
    This layer implements proper multi-head attention mechanism where:
    - Each head computes attention independently
    - Outputs from all heads are concatenated
    - Final output dimension is num_heads * out_features_per_head
    """
    def __init__(self, in_features: int, out_features_per_head: int, num_heads: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_head = out_features_per_head
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation for each head
        self.W = nn.Linear(in_features, out_features_per_head * num_heads)
        
        # Attention mechanism for each head
        self.attention = nn.Linear(2 * out_features_per_head, 1)
        
        # LeakyReLU for attention
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Input features of shape [num_nodes, in_features]
            adj: Adjacency matrix of shape [num_nodes, num_nodes]
        Returns:
            Output features of shape [num_nodes, num_heads * out_features_per_head]
        """
        num_nodes = h.shape[0]
        
        # Linear transformation for all heads
        Wh = self.W(h)  # [num_nodes, out_features_per_head * num_heads]
        Wh = Wh.view(num_nodes, self.num_heads, self.out_features_per_head)  # [num_nodes, num_heads, out_features_per_head]
        
        # Compute attention for each head
        attention_outputs = []
        
        for head in range(self.num_heads):
            # Get features for this head
            Wh_head = Wh[:, head, :]  # [num_nodes, out_features_per_head]
            
            # Compute attention scores for this head
            attention_scores = torch.zeros(num_nodes, num_nodes, device=h.device)
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj[i, j] > 0:  # Only compute attention for connected nodes
                        # Concatenate source and target features for this head
                        combined = torch.cat([Wh_head[i], Wh_head[j]], dim=0)  # [2 * out_features_per_head]
                        attention_scores[i, j] = self.leakyrelu(self.attention(combined))
            
            # Apply softmax to get attention weights for this head
            attention_weights = F.softmax(attention_scores, dim=1)
            attention_weights = F.dropout(attention_weights, self.dropout, training=self.training)
            
            # Apply attention to features for this head
            h_prime_head = torch.matmul(attention_weights, Wh_head)  # [num_nodes, out_features_per_head]
            attention_outputs.append(h_prime_head)
        
        # Concatenate outputs from all heads
        h_prime = torch.cat(attention_outputs, dim=1)  # [num_nodes, num_heads * out_features_per_head]
        
        return h_prime


class BipartiteGAT(nn.Module):
    """
    Bipartite Graph Attention Network for critic value aggregation
    
    Network Structure:
    - Input: Values from 2 critics (QoS-aware, Energy)
    - Output: Two aggregated values for 2 actors (EE, NS)
    
    Attention Mechanism:
    - QoS-aware critic → Both Actors: Focus on QoS requirements
    - Energy critic → Both Actors: Focus on power optimization
    
    Cross-influence is enabled through the adjacency matrix, allowing each critic
    to influence both actors' policies based on learned attention weights.
    """
    def __init__(self, 
                 critic_features, 
                 actor_features, 
                 hidden_dim=64, 
                 num_heads=4, 
                 num_critics=2,  # [QoS-aware, Energy]
                 num_actors=2,   # [EE Actor, NS Actor]
                 dropout=0.1):
        super(BipartiteGAT, self).__init__()
        self.critic_features = critic_features
        self.actor_features = actor_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_critics = num_critics
        self.num_actors = num_actors
        self.dropout = dropout
        
        # Feature transformation for critics
        self.critic_transform = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Each critic value is a scalar
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Feature transformation for actors
        self.actor_transform = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Each actor value is also a scalar
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # GAT layer for critic-to-actor attention
        # MultiHeadGATLayer outputs [num_nodes, num_heads * out_features_per_head]
        # We want final output to be [num_nodes, hidden_dim], so out_features_per_head = hidden_dim // num_heads
        out_features_per_head = hidden_dim // num_heads  # 64 // 4 = 16
        self.gat = MultiHeadGATLayer(
            in_features=hidden_dim,
            out_features_per_head=out_features_per_head,  # 16 per head
            num_heads=num_heads,  # 4 heads
            dropout=dropout
        )
        
        # Separate output projections for each actor
        self.ee_actor_output = nn.Linear(hidden_dim, 1)  # Output single value
        self.ns_actor_output = nn.Linear(hidden_dim, 1)  # Output single value
        
        # Attention weights for critic-actor relationships
        self.attention_weights = nn.Parameter(torch.zeros(num_critics, num_actors))
        nn.init.xavier_uniform_(self.attention_weights)
    
    def forward(self, critic_values):
        """
        Aggregate critic values using graph attention
        
        Args:
            critic_values: List of 2 critic values, each of shape [1]
                         [qos_value, energy_value]
        
        Returns:
            Tuple of (ee_actor_value, ns_actor_value), each of shape [1]
        """
        # Stack critic values [num_critics, 1]
        stacked_values = torch.stack(critic_values, dim=0)  # [num_critics, 1]
        
        # Transform critic features [num_critics, hidden_dim]
        critic_features = self.critic_transform(stacked_values)
        
        # Initialize actor features [num_actors, hidden_dim]
        actor_features = torch.zeros(
            self.num_actors, 
            self.hidden_dim,
            device=critic_values[0].device
        )
        
        # Create bipartite adjacency matrix with learned attention weights
        # [num_critics + num_actors, num_critics + num_actors]
        adj = torch.zeros(
            self.num_critics + self.num_actors,
            self.num_critics + self.num_actors,
            device=critic_values[0].device
        )
        
        # Apply learned attention weights to create connections
        attention_weights = torch.sigmoid(self.attention_weights)  # [num_critics, num_actors]
        for i in range(self.num_critics):
            for j in range(self.num_actors):
                adj[i, self.num_critics + j] = attention_weights[i, j]
        
        # Combine features for GAT [num_critics + num_actors, hidden_dim]
        all_features = torch.cat([critic_features, actor_features], dim=0)
        
        # Apply GAT [num_critics + num_actors, num_heads * out_features_per_head] = [4, 4*16] = [4, 64]
        updated_features = self.gat(all_features, adj)
        
        # Extract actor features [num_actors, hidden_dim] = [2, 64]
        actor_features = updated_features[self.num_critics:]
        
        # Project to actor-specific outputs
        # [1] for each actor
        ee_actor_value = self.ee_actor_output(actor_features[0])
        ns_actor_value = self.ns_actor_output(actor_features[1])
        
        return ee_actor_value, ns_actor_value


class CriticAggregator(nn.Module):
    """
    Aggregates critic values for actor policy updates
    
    This module uses a bipartite GAT to determine how each critic's evaluation
    should influence both the EE Actor (time scheduling) and NS Actor (slice allocation).
    
    Value Flow:
    - QoS-aware critic influences both actors based on QoS requirements
    - Energy critic influences both actors based on power optimization
    - Cross-influences are learned through attention mechanism
    """
    def __init__(self):
        super().__init__()
        
        # Calculate out_features_per_head to ensure proper dimension handling
        out_features_per_head1 = config.GAT['hidden_dim'] // config.GAT['num_heads']
        
        # GAT layers
        self.gat1 = MultiHeadGATLayer(
            in_features=config.GAT['input_dim'],
            out_features_per_head=out_features_per_head1,  # hidden_dim // num_heads
            num_heads=config.GAT['num_heads'],
            dropout=config.GAT['dropout']
        )
        
        # Second GAT layer: input is the output of first GAT (hidden_dim * num_heads)
        # We want final output to be hidden_dim, so out_features_per_head = hidden_dim // num_heads
        out_features_per_head2 = config.GAT['hidden_dim'] // config.GAT['num_heads']
        
        self.gat2 = MultiHeadGATLayer(
            in_features=config.GAT['hidden_dim'],  # Output from first GAT
            out_features_per_head=out_features_per_head2,  # hidden_dim // num_heads
            num_heads=config.GAT['num_heads'],
            dropout=config.GAT['dropout']
        )
        
        # Output projection
        self.proj = nn.Linear(
            config.GAT['hidden_dim'],  # Output from second GAT
            config.GAT['output_dim']
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape [num_nodes, input_dim]
            adj: Adjacency matrix of shape [num_nodes, num_nodes]
        Returns:
            Aggregated features of shape [num_nodes, output_dim]
        """
        x = F.elu(self.gat1(x, adj))
        x = F.dropout(x, config.GAT['dropout'], training=self.training)
        x = self.gat2(x, adj)
        x = self.proj(x)
        return x
