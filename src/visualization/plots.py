"""Visualization utilities for ECG signals with attribution overlays."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_ecg_with_attribution(
    signal: np.ndarray,
    attributions: np.ndarray,
    fs: float = 300.0,
    method_name: str = "Attribution",
    label: str = None,
    save_path: str = None,
) -> Figure:
    """Overlay an attribution heatmap on an ECG signal.

    Args:
        signal:       1D ECG signal array [timesteps].
        attributions: 1D attribution array [timesteps], same length as signal.
        fs:           Sampling frequency in Hz.
        method_name:  XAI method name for the plot title.
        label:        Ground-truth label string (optional).
        save_path:    If given, saves the figure here.

    Returns:
        Matplotlib Figure.
    """
    time = np.arange(len(signal)) / fs
    norm_attr = _normalize(attributions)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    axes[0].plot(time, signal, color="black", linewidth=0.8)
    axes[0].set_ylabel("Amplitude (mV)")
    title = f"ECG — {method_name}"
    if label:
        title += f"  |  Label: {label}"
    axes[0].set_title(title)

    axes[1].plot(time, signal, color="black", linewidth=0.8, alpha=0.35)
    sc = axes[1].scatter(time, signal, c=norm_attr, cmap="RdYlGn_r", s=3, zorder=3)
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_xlabel("Time (s)")
    plt.colorbar(sc, ax=axes[1], label="Attribution (normalised)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_method_comparison(
    signal: np.ndarray,
    attributions_dict: dict[str, np.ndarray],
    fs: float = 300.0,
    save_path: str = None,
) -> Figure:
    """Plot multiple XAI methods stacked for side-by-side comparison.

    Args:
        signal:            1D ECG signal array [timesteps].
        attributions_dict: Dict mapping method name → attribution array.
        fs:                Sampling frequency in Hz.
        save_path:         If given, saves the figure here.

    Returns:
        Matplotlib Figure.
    """
    n = len(attributions_dict)
    time = np.arange(len(signal)) / fs

    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 3 * (n + 1)), sharex=True)

    axes[0].plot(time, signal, color="black", linewidth=0.8)
    axes[0].set_title("Original ECG Signal")
    axes[0].set_ylabel("Amplitude (mV)")

    for ax, (name, attrs) in zip(axes[1:], attributions_dict.items()):
        # Normalize each method independently so all are visible
        norm_attrs = _normalize(attrs)
        ax.plot(time, signal, color="black", linewidth=0.8, alpha=0.35)
        sc = ax.scatter(time, signal, c=norm_attrs, cmap="RdYlGn_r", s=3, zorder=3,
                        vmin=0.0, vmax=1.0)
        ax.set_title(name)
        ax.set_ylabel("Amplitude (mV)")
        plt.colorbar(sc, ax=ax, label="Attribution (normalised)")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_deletion_curves(
    scores_dict: dict[str, np.ndarray],
    save_path: str = None,
) -> Figure:
    """Plot deletion curves for multiple XAI methods.

    Args:
        scores_dict: Dict mapping method name → deletion scores array.
        save_path:   If given, saves the figure here.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, scores in scores_dict.items():
        ax.plot(scores, label=name)
    ax.set_xlabel("Deletion step")
    ax.set_ylabel("Model confidence")
    ax.set_title("Deletion Curves (lower drop = less faithful)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_faithfulness_curves(
    deletion_scores: dict[str, np.ndarray],
    insertion_scores: dict[str, np.ndarray],
    save_path: str = None,
) -> Figure:
    """Plot deletion and insertion curves side by side.

    Args:
        deletion_scores:  Dict mapping method name → deletion scores array.
        insertion_scores: Dict mapping method name → insertion scores array.
        save_path:        If given, saves the figure here.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax_del, ax_ins) = plt.subplots(1, 2, figsize=(14, 5))

    for name, scores in deletion_scores.items():
        ax_del.plot(scores, label=name)
    ax_del.set_xlabel("Deletion step")
    ax_del.set_ylabel("Model confidence")
    ax_del.set_title("Deletion Curves (steeper drop = more faithful)")
    ax_del.legend()

    for name, scores in insertion_scores.items():
        ax_ins.plot(scores, label=name)
    ax_ins.set_xlabel("Insertion step")
    ax_ins.set_ylabel("Model confidence")
    ax_ins.set_title("Insertion Curves (steeper rise = more faithful)")
    ax_ins.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_aggregate_aopc(
    summary: dict[str, dict[str, tuple[float, float]]],
    save_path: str = None,
) -> Figure:
    """Grouped bar chart of aggregate AOPC scores with error bars.

    Args:
        summary:   Dict from ``aggregate_aopc()`` — method → metric → (mean, std).
        save_path: If given, saves the figure here.

    Returns:
        Matplotlib Figure.
    """
    methods = list(summary.keys())
    metrics = list(next(iter(summary.values())).keys())
    n_metrics = len(metrics)
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(metrics):
        means = [summary[m][metric][0] for m in methods]
        stds = [summary[m][metric][1] for m in methods]
        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=metric, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("AOPC")
    ax.set_title("Aggregate AOPC across test samples (mean ± std)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _normalize(arr: np.ndarray) -> np.ndarray:
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)
