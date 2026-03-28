"""
PINN model definition for fatigue life prediction.
"""

import torch
import torch.nn as nn


class EnhancedPINNFatigueModel(nn.Module):
    """Physics-Informed Neural Network for fatigue life prediction."""

    def __init__(self, input_dim=5, hidden_dim=32, use_dropout=False, dropout_rate=0.1):
        super(EnhancedPINNFatigueModel, self).__init__()
        self.use_dropout = use_dropout

        # Physical parameters
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.beta1 = nn.Parameter(torch.tensor(0.005, dtype=torch.float32))
        self.beta2 = nn.Parameter(torch.tensor(0.012, dtype=torch.float32))
        self.beta3 = nn.Parameter(torch.tensor(-0.008, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))

        # Neural networks
        self.fiber_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU()
        )

        self.other_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU()
        )

        self.combine_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        fiber_features = x[:, :3]
        other_features = x[:, 3:]
        S = other_features[:, 1:2]
        S = torch.clamp(S, min=1e-6)
        log_S = torch.log10(S + 1e-6)

        phys_output = (self.alpha +
                       self.beta1 * fiber_features[:, 0:1] +
                       self.beta2 * fiber_features[:, 1:2] +
                       self.beta3 * fiber_features[:, 2:3] -
                       self.gamma * log_S)

        fiber_encoded = self.fiber_net(fiber_features)
        other_encoded = self.other_net(other_features)

        if self.use_dropout:
            fiber_encoded = self.dropout(fiber_encoded)
            other_encoded = self.dropout(other_encoded)

        combined = torch.cat([fiber_encoded, other_encoded], dim=1)
        residual = self.combine_net(combined)

        return phys_output + residual

    def get_physical_params(self):
        return {
            'alpha': self.alpha.item(),
            'beta1': self.beta1.item(),
            'beta2': self.beta2.item(),
            'beta3': self.beta3.item(),
            'gamma': self.gamma.item()
        }