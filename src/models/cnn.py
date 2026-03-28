import torch
import torch.nn as nn

# Supported xresnet variants and their fastai constructors
_XRESNET_VARIANTS = {
    "xresnet1d34": "xresnet34",
    "xresnet1d50": "xresnet50",
    "xresnet1d101": "xresnet101",
}


def build_xresnet1d(
    variant: str = "xresnet1d34",
    n_leads: int = 12,
    n_classes: int = 5,
) -> nn.Module:
    """Instantiate an xresnet 1D model randomly initialised.

    Uses fastai's xresnet with ndim=1 to produce a 1D network compatible
    with [batch, 12, timesteps] ECG inputs.

    The model is an nn.Sequential with this index layout:
      0, 1, 2  — stem ConvLayer blocks
      3        — MaxPool
      4, 5, 6, 7 — residual block groups
      8        — AdaptiveAvgPool
      9, 10, 11 — Flatten, Dropout, Linear

    Target layer for GradCAM/HiResCAM = "7" (last residual block group).

    Args:
        variant:   One of 'xresnet1d34', 'xresnet1d50', 'xresnet1d101'.
        n_leads:   Number of input channels (leads). Default 12.
        n_classes: Number of output classes. Default 5 (SB, SR, AF, ST, TWC).

    Returns:
        The nn.Module set to eval mode.
    """
    if variant not in _XRESNET_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(_XRESNET_VARIANTS)}")

    import fastai.vision.models.xresnet as xresnet_module
    constructor = getattr(xresnet_module, _XRESNET_VARIANTS[variant])
    model = constructor(c_in=n_leads, n_out=n_classes, ndim=1)
    model.eval()
    return model


def load_model(
    weights_path: str,
    n_leads: int = 12,
    n_classes: int = 5,
    variant: str = "xresnet1d34",
) -> nn.Module:
    """Load an xresnet1d model from a .pt checkpoint or a fastai .pkl learner.

    For .pt / .pth files (trained from scratch via train_from_scratch()):
        Builds the architecture with build_xresnet1d() then loads state_dict.

    For .pkl files (fastai learner):
        Loads the learner and extracts the underlying nn.Module.

    Args:
        weights_path: Path to a .pt / .pth state-dict file or a fastai .pkl learner.
        n_leads:      Input channels — only used for .pt format. Default 12.
        n_classes:    Output classes — only used for .pt format. Default 5.
        variant:      Model variant — only used for .pt format. Default 'xresnet1d34'.

    Returns:
        The underlying nn.Module set to eval mode.
    """
    if weights_path.endswith('.pt') or weights_path.endswith('.pth'):
        model = build_xresnet1d(variant, n_leads=n_leads, n_classes=n_classes)
        ckpt = torch.load(weights_path, map_location='cpu')
        state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state)
    else:  # fastai .pkl learner
        from fastai.learner import load_learner
        model = load_learner(weights_path).model
    model.eval()
    return model


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Retrieve a submodule by dotted name (e.g. '7' or '7.2').

    Useful for pointing GradCAM / HiResCAM at a specific convolutional layer.
    Run print_layers() to discover available names.
    """
    return dict(model.named_modules())[layer_name]


def print_layers(model: nn.Module) -> None:
    """Print all named modules — helps identify the right GradCAM target layer."""
    for name, module in model.named_modules():
        print(f"{name:60s}  {module.__class__.__name__}")
