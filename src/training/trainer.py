"""Training utilities for xresnet1d on the ECG Arrhythmia Database.

train_from_scratch() — full training run with cosine LR, checkpointing, early stopping.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_from_scratch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    patience: int = 10,
    checkpoint_path: str = "outputs/models/xresnet1d101_trained.pt",
    device: str = "mps",
    scheduler_type: str = "reduce_on_plateau",
) -> nn.Module:
    """Train xresnet1d from random initialisation on the ECG Arrhythmia Database.

    Args:
        model:           Randomly-initialised model from build_xresnet1d().
        train_loader:    DataLoader for the training split.
        val_loader:      DataLoader for the validation split.
        epochs:          Maximum number of epochs.
        lr:              Initial learning rate for Adam.
        weight_decay:    L2 regularisation coefficient.
        grad_clip:       Maximum gradient norm (0 disables clipping).
        patience:        Early-stopping patience (epochs without val AUROC improvement).
        checkpoint_path: Where to save the best model state dict.
        device:          'cpu', 'cuda', or 'mps'.
        scheduler_type:  'cosine' or 'reduce_on_plateau'.

    Returns:
        Model loaded with the best checkpoint weights.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:  # reduce_on_plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6,
        )
    criterion = nn.BCEWithLogitsLoss()

    # If a checkpoint already exists, use its AUROC as the baseline
    # so we never overwrite a better model from a previous run.
    best_auroc = -1.0
    if os.path.exists(checkpoint_path):
        existing = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(existing, dict) and 'val_auroc' in existing:
            best_auroc = existing['val_auroc']
            print(f"Existing checkpoint has val AUROC {best_auroc:.4f} — will only overwrite if beaten.")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            if torch.isnan(loss):
                continue  # skip batch — can happen on MPS in early epochs
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_auroc = _evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_auroc)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            epochs_without_improvement = 0
            torch.save({'state_dict': model.state_dict(), 'val_auroc': val_auroc}, checkpoint_path)
            print(f"  -> New best AUROC {best_auroc:.4f} — checkpoint saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no improvement for {patience} consecutive epochs.")
                break

    # Reload best weights before returning
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"\nTraining finished. Best val AUROC: {best_auroc:.4f}")
    return model


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            total_loss += criterion(outputs, labels).item()
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Guard against NaN outputs (can happen on MPS with randomly-initialised
    # weights producing very large activations in early epochs).
    if np.isnan(all_probs).any():
        nan_frac = np.isnan(all_probs).mean()
        print(f"  Warning: {nan_frac:.1%} of predictions are NaN — replacing with 0.5")
        np.nan_to_num(all_probs, copy=False, nan=0.5)

    mean_auroc = roc_auc_score(all_labels, all_probs, average="macro")
    return total_loss / len(loader), mean_auroc
