# model.py

"""
Contains the definition for the Multi-Layer Perceptron (MLP) model.
This allows the model architecture to be imported by other scripts like
training.py and predict.py.
"""

import torch.nn as nn


class BalancedMLP(nn.Module):
    """A Multi-Layer Perceptron with BatchNorm, LeakyReLU, and Dropout."""

    def __init__(self, input_dim: int, dropout_rate: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Kaiming normal initialization to linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: nn.Module) -> nn.Module:
        """Define the forward pass of the model."""
        return self.model(x)
