import torch
import torch.nn as nn
from config import config

class TransformerStateEncoder(nn.Module):
    """
    Transformer-based encoder for dynamic number of UEs (spatial/contextual encoding).
    Input: [batch_size, num_ues, num_features]
    Output: [batch_size, num_ues, hidden_dim]
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.input_dim = config.STATE_ENCODER['num_features']
        self.hidden_dim = config.STATE_ENCODER['hidden_dim']
        self.num_layers = config.STATE_ENCODER['num_layers']
        self.num_heads = config.STATE_ENCODER['num_heads']
        self.dropout = config.STATE_ENCODER['dropout']
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
            device=self.device
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim, device=self.device)

    def forward(self, x):
        """
        x: [batch_size, num_ues, num_features]
        returns: [batch_size, num_ues, hidden_dim]
        """
        x = x.to(self.device)
        x_proj = self.input_proj(x)
        # No mask: all UEs are valid
        encoded = self.transformer_encoder(x_proj)
        return encoded 