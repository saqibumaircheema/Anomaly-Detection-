"""Training loop used for the compact SM-background autoencoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim


def train_model(
    train_loader,
    validation_loader,
    model: nn.Module,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    save_path: str = "best_autoencoder.pt",
):
    """Train the model and save the checkpoint with the lowest validation loss."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_validation_loss = float("inf")
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for data, _ in train_loader:
            reconstruction = model(data)
            loss = criterion(reconstruction, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for data, _ in validation_loader:
                reconstruction = model(data)
                validation_loss += criterion(reconstruction, data).item() * data.size(0)

        validation_loss /= len(validation_loader.dataset)
        history["validation_loss"].append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train={train_loss:.6f} | validation={validation_loss:.6f}"
        )

    return history, best_validation_loss
