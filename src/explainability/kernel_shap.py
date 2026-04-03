"""KernelSHAP for 12-lead ECG signals.

Uses lead × temporal segmentation to make KernelSHAP tractable on [12, 5000]
inputs while preserving per-lead attribution differences.
"""

import numpy as np
import shap
import torch
import torch.nn as nn


def compute_kernel_shap(
    model: nn.Module,
    signal: torch.Tensor,
    target_class: int,
    background: torch.Tensor,
    n_segments: int = 100,
    nsamples: int = 1000,
) -> np.ndarray:
    """KernelSHAP attribution via lead × temporal segmentation.

    Each feature corresponds to a (lead, time_segment) pair, so that each
    lead receives independent SHAP values.  The total number of features is
    ``n_leads × n_segments`` (e.g. 12 × 100 = 1200).

    Args:
        model:        Trained model in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        target_class: Class index to explain.
        background:   Background distribution tensor of shape [N, leads, timesteps].
        n_segments:   Number of temporal segments per lead.
        nsamples:     Number of coalitions sampled by KernelExplainer.

    Returns:
        Attribution array of shape [leads, timesteps].
    """
    device = signal.device
    signal_np = signal.squeeze(0).detach().cpu().numpy()  # [leads, timesteps]
    bg_np = background.detach().cpu().numpy()              # [N, leads, timesteps]

    n_leads, n_timesteps = signal_np.shape
    seg_size = n_timesteps // n_segments
    # Segment boundaries — last segment absorbs remainder
    seg_bounds = [(i * seg_size, (i + 1) * seg_size) for i in range(n_segments)]
    seg_bounds[-1] = (seg_bounds[-1][0], n_timesteps)

    n_features = n_leads * n_segments  # one feature per (lead, segment)

    # Background mean signal for masking
    bg_mean = bg_np.mean(axis=0)  # [leads, timesteps]

    def predict_fn(masks: np.ndarray) -> np.ndarray:
        """Map binary masks [batch, n_leads * n_segments] → model outputs."""
        batch_size = masks.shape[0]
        batch = np.tile(bg_mean, (batch_size, 1, 1))  # start from background

        for i in range(batch_size):
            for lead_idx in range(n_leads):
                for seg_idx in range(n_segments):
                    feat_idx = lead_idx * n_segments + seg_idx
                    if masks[i, feat_idx] == 1:
                        start, end = seg_bounds[seg_idx]
                        batch[i, lead_idx, start:end] = signal_np[lead_idx, start:end]

        with torch.no_grad():
            tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            logits = model(tensor)
            probs = torch.sigmoid(logits)[:, target_class].cpu().numpy()
        return probs

    # Reference: all features "on" = original signal; background = all "off"
    fg_mask = np.ones((1, n_features))
    bg_mask = np.zeros((1, n_features))

    np.random.seed(42)
    explainer = shap.KernelExplainer(predict_fn, bg_mask)
    shap_values = explainer.shap_values(fg_mask, nsamples=nsamples, silent=True)  # [1, n_features]

    # Reshape and upsample → [leads, timesteps]
    seg_shap = shap_values.flatten().reshape(n_leads, n_segments)  # [leads, n_segments]
    attr = np.zeros((n_leads, n_timesteps), dtype=np.float32)
    for seg_idx in range(n_segments):
        start, end = seg_bounds[seg_idx]
        attr[:, start:end] = seg_shap[:, seg_idx : seg_idx + 1]

    return attr
