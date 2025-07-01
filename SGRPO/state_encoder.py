import torch
import torch.nn as nn
from config import config

class TransformerStateEncoder(nn.Module):
    """
    Transformer-based encoder for dynamic number of UEs (spatial/contextual encoding).
    Input: [batch_size, num_ues, num_features]
    Output: [batch_size, num_ues, hidden_dim]
    """
    def __init__(self, num_features=None, hidden_dim=None, num_layers=None, num_heads=None, dropout=None):
        super().__init__()
        self.num_features = num_features or config.STATE_ENCODER['num_features']
        self.hidden_dim = hidden_dim or config.STATE_ENCODER['hidden_dim']
        self.num_layers = num_layers or config.STATE_ENCODER['num_layers']
        self.num_heads = num_heads or config.STATE_ENCODER['num_heads']
        self.dropout = dropout or config.STATE_ENCODER['dropout']

        self.input_proj = nn.Linear(self.num_features, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, x):
        """
        x: [batch_size, num_ues, num_features]
        returns: [batch_size, num_ues, hidden_dim]
        """
        x_proj = self.input_proj(x)
        # No mask: all UEs are valid
        encoded = self.transformer_encoder(x_proj)
        return encoded 