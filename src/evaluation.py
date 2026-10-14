"""Evaluation helpers for reconstruction-error anomaly detection."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn


def anomaly_scores(model: nn.Module, data: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Return mean-squared reconstruction error for each event."""
    model.eval()
    tensor = torch.as_tensor(data, dtype=torch.float32)
    scores: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size]
            reconstruction = model(batch)
            score = torch.mean((reconstruction - batch) ** 2, dim=1)
            scores.append(score.cpu().numpy())

    return np.concatenate(scores)


def roc_metrics(sm_scores: np.ndarray, bsm_scores: np.ndarray):
    """Return false-positive rate, true-positive rate and ROC AUC."""
    labels = np.concatenate(
        [np.zeros(len(sm_scores), dtype=int), np.ones(len(bsm_scores), dtype=int)]
    )
    scores = np.concatenate([sm_scores, bsm_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    return fpr, tpr, thresholds, auc
