"""Faithfulness evaluation metrics for XAI attribution methods.

Faithfulness measures whether high-attribution regions actually drive
the model's prediction. The deletion curve (AOPC) is the standard metric:
mask the most important timesteps first and measure how fast the model's
confidence drops. The insertion curve is the complementary metric: start
from a fully-masked baseline and progressively restore highest-attribution
timesteps, measuring how fast confidence recovers.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def deletion_curve(
    model: nn.Module,
    signal: torch.Tensor,
    attributions: np.ndarray,
    target_class: int,
    steps: int = 100,
    mask_value: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute the deletion curve and AOPC score.

    Progressively masks the most important timesteps (highest attribution)
    and records model confidence at each step.

    Args:
        model:        Trained model in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        attributions: 1D attribution array [timesteps] (already aggregated
                      across leads if multi-lead).
        target_class: Class index being explained.
        steps:        Number of masking steps.
        mask_value:   Value used to replace masked timesteps.

    Returns:
        scores: Confidence at each deletion step, shape [steps].
        aopc:   Area-over-the-perturbation-curve — drop from original score.
                Higher = more faithful attribution.
    """
    model.eval()
    n_timesteps = attributions.shape[-1]
    step_size = max(1, n_timesteps // steps)
    sorted_idx = np.argsort(attributions)[::-1]  # most important first

    with torch.no_grad():
        original_score = torch.sigmoid(model(signal))[0, target_class].item()

    signal_np = signal.squeeze().cpu().numpy().copy()
    scores = []

    for i in range(0, n_timesteps, step_size):
        to_mask = sorted_idx[: i + step_size]
        masked = signal_np.copy()
        masked[..., to_mask] = mask_value
        masked_tensor = torch.tensor(masked, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(model(masked_tensor))[0, target_class].item()
        scores.append(score)

    scores_arr = np.array(scores)
    aopc = original_score - scores_arr.mean()
    return scores_arr, aopc


def insertion_curve(
    model: nn.Module,
    signal: torch.Tensor,
    attributions: np.ndarray,
    target_class: int,
    steps: int = 100,
    mask_value: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute the insertion curve and AOPC score.

    Starts from a fully-masked baseline and progressively restores the most
    important timesteps (highest attribution), recording model confidence at
    each step.

    Args:
        model:        Trained model in eval mode.
        signal:       Input tensor of shape [1, leads, timesteps].
        attributions: 1D attribution array [timesteps] (already aggregated
                      across leads if multi-lead).
        target_class: Class index being explained.
        steps:        Number of insertion steps.
        mask_value:   Value used for the masked baseline.

    Returns:
        scores: Confidence at each insertion step, shape [steps].
        aopc:   scores.mean() - baseline_score. Higher = more faithful.
    """
    model.eval()
    n_timesteps = attributions.shape[-1]
    step_size = max(1, n_timesteps // steps)
    sorted_idx = np.argsort(attributions)[::-1]  # most important first

    signal_np = signal.squeeze().cpu().numpy().copy()
    baseline = np.full_like(signal_np, mask_value)

    with torch.no_grad():
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)
        baseline_score = torch.sigmoid(model(baseline_tensor))[0, target_class].item()

    scores = []

    for i in range(0, n_timesteps, step_size):
        to_restore = sorted_idx[: i + step_size]
        restored = baseline.copy()
        restored[..., to_restore] = signal_np[..., to_restore]
        restored_tensor = torch.tensor(restored, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = torch.sigmoid(model(restored_tensor))[0, target_class].item()
        scores.append(score)

    scores_arr = np.array(scores)
    aopc = scores_arr.mean() - baseline_score
    return scores_arr, aopc


def _random_attributions(n_timesteps: int, seed: int = 0) -> np.ndarray:
    """Return reproducible random attribution values for baseline comparison."""
    return np.random.RandomState(seed).rand(n_timesteps)


def compare_methods(
    model: nn.Module,
    signal: torch.Tensor,
    attributions_dict: dict[str, np.ndarray],
    target_class: int,
    steps: int = 100,
    include_random: bool = True,
    include_insertion: bool = False,
) -> dict[str, dict[str, float]]:
    """Run deletion (and optionally insertion) curves for multiple XAI methods.

    Args:
        attributions_dict: Mapping of method name → attribution array.
        include_random:    If True, adds a "Random" baseline entry.
        include_insertion: If True, also computes insertion AOPC.

    Returns:
        Dict of method name → {"deletion_aopc": float, "insertion_aopc": float}.
    """
    methods = dict(attributions_dict)
    if include_random:
        n_timesteps = next(iter(methods.values())).shape[-1]
        methods["Random"] = _random_attributions(n_timesteps)

    results = {}
    for name, attrs in methods.items():
        _, del_aopc = deletion_curve(model, signal, attrs, target_class, steps)
        entry = {"deletion_aopc": del_aopc}
        if include_insertion:
            _, ins_aopc = insertion_curve(model, signal, attrs, target_class, steps)
            entry["insertion_aopc"] = ins_aopc
        results[name] = entry

    return results


def aggregate_aopc(
    model: nn.Module,
    dataset,
    attribution_fn,
    target_class: int,
    n_samples: int = 20,
    steps: int = 100,
    include_insertion: bool = False,
    include_random: bool = True,
    seed: int = 42,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compute AOPC scores aggregated across multiple test samples.

    Args:
        model:          Trained model in eval mode.
        dataset:        Dataset returning (signal, label) tuples.
        attribution_fn: Callback ``fn(model, signal, target_class) -> dict[str, ndarray]``
                        returning attributions dict for a single sample.
        target_class:   Class index being explained.
        n_samples:      Number of samples to aggregate over.
        steps:          Number of masking steps per curve.
        include_insertion: If True, also computes insertion AOPC.
        include_random: If True, adds a "Random" baseline entry.
        seed:           Random seed for sample selection.

    Returns:
        Dict of method → metric → (mean, std) across samples.
    """
    # Find candidate samples with target class active
    candidates = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label[target_class] == 1.0:
            candidates.append(i)

    rng = np.random.RandomState(seed)
    selected = rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False)

    # Accumulate per-sample results
    all_results: dict[str, dict[str, list[float]]] = {}

    for idx in tqdm(selected, desc="Aggregating AOPC"):
        sig, _ = dataset[idx]
        signal = sig.unsqueeze(0)
        attrs_dict = attribution_fn(model, signal, target_class)
        sample_results = compare_methods(
            model, signal, attrs_dict, target_class,
            steps=steps, include_random=include_random,
            include_insertion=include_insertion,
        )
        for method, metrics in sample_results.items():
            if method not in all_results:
                all_results[method] = {}
            for metric, value in metrics.items():
                all_results[method].setdefault(metric, []).append(value)

    # Compute mean and std
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for method, metrics in all_results.items():
        summary[method] = {}
        for metric, values in metrics.items():
            arr = np.array(values)
            summary[method][metric] = (float(arr.mean()), float(arr.std()))

    return summary
