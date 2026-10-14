"""Fully connected autoencoder used for the LHC anomaly-detection study."""

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """39 -> 32 -> 8 -> 32 -> 39 autoencoder."""

    def __init__(self, input_dim: int = 39, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
