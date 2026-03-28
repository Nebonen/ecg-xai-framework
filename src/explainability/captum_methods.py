"""Gradient-based XAI methods via Captum.

Covers:
  - GradientSHAP
"""

import numpy as np
import torch
import torch.nn as nn
from captum.attr import GradientShap


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().detach().cpu().numpy()


def compute_gradient_shap(
    model: nn.Module,
    signal: torch.Tensor,
    target_class: int,
    background: torch.Tensor,
    n_samples: int = 50,
) -> np.ndarray:
    """GradientSHAP attribution.

    Args:
        model:        Trained model in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        target_class: Class index to explain.
        background:   Background distribution tensor of shape [N, leads, timesteps].
        n_samples:    Number of noise samples per input.

    Returns:
        Attribution array of shape [leads, timesteps].
    """
    gs = GradientShap(model)
    attrs = gs.attribute(
        signal,
        baselines=background,
        target=target_class,
        n_samples=n_samples,
    )
    return _to_numpy(attrs)
