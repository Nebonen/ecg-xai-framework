"""Train xresnet1d from scratch on the ECG Arrhythmia Database.

Usage:
    conda activate ecg-xai
    python train.py

Reads hyperparameters from configs/training.yaml and configs/model.yaml.
Saves the best checkpoint (by val AUROC) to the path in training.yaml.
"""

import os

import yaml
from torch.utils.data import DataLoader

from src.models.cnn import build_xresnet1d
from src.data.dataset import ArrhythmiaDataset
from src.training.trainer import train_from_scratch


def main():
    # Load configs
    with open('configs/data.yaml') as f:
        data_cfg = yaml.safe_load(f)
    with open('configs/training.yaml') as f:
        train_cfg = yaml.safe_load(f)['training']
    with open('configs/model.yaml') as f:
        model_cfg = yaml.safe_load(f)['model']

    DATA_DIR = data_cfg['dataset']['raw_dir']
    FS       = data_cfg['signal']['sample_rate']
    CLASSES  = data_cfg['labels']['classes']
    SEED     = data_cfg['splits']['seed']

    # Use preprocessed cache if available (run preprocess_dataset.py first)
    cache_dir = os.path.join('data', 'preprocessed', str(FS))
    if not os.path.isdir(cache_dir):
        print(f"No cache found at {cache_dir} — loading from WFDB (slow).")
        print("Run 'python preprocess_dataset.py' first for faster training.")
        cache_dir = None

    # Datasets
    ds_kwargs = dict(
        data_dir=DATA_DIR,
        classes=CLASSES,
        seed=SEED,
        train_ratio=data_cfg['splits']['train_ratio'],
        val_ratio=data_cfg['splits']['val_ratio'],
        cache_dir=cache_dir,
        sampling_rate=FS,
    )
    train_ds = ArrhythmiaDataset(split='train', augment=True, **ds_kwargs)
    val_ds   = ArrhythmiaDataset(split='val',   augment=False, **ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
    print(f"Classes: {CLASSES}")
    print(f"Device: {train_cfg['device']}  |  Epochs: {train_cfg['epochs']}  |  LR: {train_cfg['learning_rate']}")

    # Build model
    variant = model_cfg['name']
    model = build_xresnet1d(
        variant=variant,
        n_leads=model_cfg['n_leads'],
        n_classes=model_cfg['n_classes'],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {variant}  |  Parameters: {total_params:,}")

    # Train
    trained_model = train_from_scratch(
        model,
        train_loader,
        val_loader,
        epochs=train_cfg['epochs'],
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        grad_clip=train_cfg['gradient_clip'],
        patience=train_cfg['early_stopping_patience'],
        checkpoint_path=train_cfg['checkpoint_path'],
        device=train_cfg['device'],
        scheduler_type=train_cfg.get('scheduler', 'reduce_on_plateau'),
    )

    print("Training complete. Best checkpoint saved to", train_cfg['checkpoint_path'])


if __name__ == '__main__':
    main()
