"""KernelSHAP for 12-lead ECG signals.

Uses temporal segmentation to make KernelSHAP tractable on [12, 5000] inputs.
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
    nsamples: int = 500,
) -> np.ndarray:
    """KernelSHAP attribution via temporal segmentation.

    Segments the time axis into ``n_segments`` groups, computes SHAP values
    at the segment level using ``shap.KernelExplainer``, then upsamples back
    to the original resolution.

    Args:
        model:        Trained model in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        target_class: Class index to explain.
        background:   Background distribution tensor of shape [N, leads, timesteps].
        n_segments:   Number of temporal segments (features for KernelSHAP).
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

    # Background mean signal for masking
    bg_mean = bg_np.mean(axis=0)  # [leads, timesteps]

    def predict_fn(masks: np.ndarray) -> np.ndarray:
        """Map binary segment masks [batch, n_segments] → model outputs."""
        batch_size = masks.shape[0]
        batch = np.tile(bg_mean, (batch_size, 1, 1))  # start from background

        for i in range(batch_size):
            for seg_idx in range(n_segments):
                if masks[i, seg_idx] == 1:
                    start, end = seg_bounds[seg_idx]
                    batch[i, :, start:end] = signal_np[:, start:end]

        with torch.no_grad():
            tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            logits = model(tensor)
            probs = torch.sigmoid(logits)[:, target_class].cpu().numpy()
        return probs

    # Reference: all segments "on" = original signal; background = all "off"
    fg_mask = np.ones((1, n_segments))
    bg_mask = np.zeros((1, n_segments))

    explainer = shap.KernelExplainer(predict_fn, bg_mask)
    shap_values = explainer.shap_values(fg_mask, nsamples=nsamples, silent=True)  # [1, n_segments]

    # Upsample segment-level SHAP values → [leads, timesteps]
    seg_shap = shap_values.flatten()  # [n_segments]
    attr = np.zeros((n_leads, n_timesteps), dtype=np.float32)
    for seg_idx in range(n_segments):
        start, end = seg_bounds[seg_idx]
        attr[:, start:end] = seg_shap[seg_idx]

    return attr
