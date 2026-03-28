"""HiResCAM and GradCAM for 1D ECG signals via signal_grad_cam.

Install the library first:
    pip install git+https://github.com/bmi-labmedinfo/signal_grad_cam.git

Uses TorchCamBuilder.get_cam() with stdout/figure suppression to extract
only the raw CAM arrays.
"""

import contextlib
import io
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from signal_grad_cam import TorchCamBuilder


def _run_cam(
    model: nn.Module,
    signal: torch.Tensor,
    target_layer: nn.Module,
    target_class: int,
    method: str,
) -> np.ndarray:
    """Run a CAM method and return the attribution map.

    Args:
        model:        Trained 1D CNN in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        target_layer: Convolutional layer to target.
        target_class: Class index to explain.
        method:       "HiResCAM" or "Grad-CAM".

    Returns:
        Attribution map of shape [timesteps].
    """
    # Find the target layer name in the model
    layer_name = None
    for name, module in model.named_modules():
        if module is target_layer:
            layer_name = name
            break
    if layer_name is None:
        raise ValueError("target_layer not found in model")

    # class_names is required by the display logic inside get_cam
    n_out = list(model.parameters())[-1].shape[0]
    class_names = [str(i) for i in range(n_out)]

    # Suppress the verbose stdout from TorchCamBuilder constructor and get_cam,
    # and close any stray matplotlib figures the library creates internally.
    figs_before = set(plt.get_fignums())

    with contextlib.redirect_stdout(io.StringIO()):
        builder = TorchCamBuilder(model=model, class_names=class_names)

        data_np = signal.squeeze(0).detach().cpu().numpy()  # [leads, timesteps]

        with tempfile.TemporaryDirectory() as tmpdir:
            cams_dict, _, _ = builder.get_cam(
                data_list=[data_np],
                data_labels=[target_class],
                target_classes=target_class,
                explainer_types=method,
                target_layers=layer_name,
                softmax_final=False,
                results_dir_path=tmpdir,
            )

    # Close any figures the library opened
    for num in set(plt.get_fignums()) - figs_before:
        plt.close(num)

    # Remove any leftover forward/backward hooks that signal_grad_cam
    # registered on model layers — they cause RuntimeError when the model
    # is later called with tensors that don't require gradients.
    for mod in model.modules():
        mod._forward_hooks.clear()
        mod._backward_hooks.clear()

    # cams_dict keys are like "HiResCAM_layername_classN"
    key = list(cams_dict.keys())[0]
    cam = cams_dict[key][0]  # first (only) sample

    # Flatten to 1D if needed (may be [1, timesteps] or [timesteps])
    cam = cam.squeeze()
    return cam


def compute_hirescam(
    model: nn.Module,
    signal: torch.Tensor,
    target_layer: nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """HiResCAM attribution for a 1D ECG signal.

    Args:
        model:        Trained 1D CNN in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        target_layer: Convolutional layer to target (use models.cnn.get_layer).
        target_class: Class index to explain. Defaults to the predicted class.

    Returns:
        Attribution map of shape [timesteps].
    """
    if target_class is None:
        with torch.no_grad():
            target_class = model(signal).argmax(dim=1).item()
    return _run_cam(model, signal, target_layer, target_class, "HiResCAM")


def compute_gradcam(
    model: nn.Module,
    signal: torch.Tensor,
    target_layer: nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """GradCAM attribution for a 1D ECG signal.

    Args:
        model:        Trained 1D CNN in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        target_layer: Convolutional layer to target.
        target_class: Class index to explain. Defaults to the predicted class.

    Returns:
        Attribution map of shape [timesteps].
    """
    if target_class is None:
        with torch.no_grad():
            target_class = model(signal).argmax(dim=1).item()
    return _run_cam(model, signal, target_layer, target_class, "Grad-CAM")
