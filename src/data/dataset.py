import os

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess

# Default top-5 arrhythmia classes (SNOMED-CT abbreviations, by frequency)
ARRHYTHMIA_CLASSES = ["SB", "SR", "AF", "ST", "TWC"]


class ArrhythmiaDataset(Dataset):
    """PyTorch Dataset for the ECG Arrhythmia Database (v1.0.0).

    Loads 12-lead ECG signals from WFDB files. Labels are multi-hot float
    vectors for the configured arrhythmia classes.

    Args:
        data_dir:      Path to the downloaded dataset root directory.
        split:         One of 'train', 'val', or 'test'.
        classes:       List of class abbreviations (e.g. ['SB', 'SR', 'AF', 'ST', 'TWC']).
        seed:          Random seed for reproducible train/val/test splits.
        train_ratio:   Fraction of data for training (default 0.8).
        val_ratio:     Fraction of data for validation (default 0.1).
        augment:       Apply random noise + amplitude scaling (training only).
        cache_dir:     Path to preprocessed .npy cache (from preprocess_dataset.py).
        metadata_path: Path to arrhythmia_metadata.csv.
        sampling_rate: Signal sampling rate in Hz (default 500).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        classes: list[str] | None = None,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        augment: bool = False,
        cache_dir: str | None = None,
        metadata_path: str = "data/processed/arrhythmia_metadata.csv",
        sampling_rate: int = 500,
    ):
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.cache_dir = cache_dir
        self.classes = classes or ARRHYTHMIA_CLASSES

        # Load metadata
        meta = pd.read_csv(metadata_path)

        # Parse abbreviations into sets for fast lookup
        meta['abbrev_set'] = meta['abbreviations'].fillna('').apply(
            lambda x: set(x.split(',')) if x else set()
        )

        # Filter to records that have at least one class in our class list
        class_set = set(self.classes)
        mask = meta['abbrev_set'].apply(lambda s: bool(s & class_set))
        meta = meta[mask].reset_index(drop=True)

        # Drop records whose .npy cache is missing (e.g. malformed .hea files)
        if cache_dir is not None:
            has_cache = meta['record_id'].apply(
                lambda rid: os.path.exists(os.path.join(cache_dir, f"{rid}.npy"))
            )
            n_missing = (~has_cache).sum()
            if n_missing > 0:
                print(f"Skipping {n_missing} records with missing .npy cache files")
                meta = meta[has_cache].reset_index(drop=True)

        # Build multi-hot labels
        labels = np.zeros((len(meta), len(self.classes)), dtype=np.float32)
        for i, abbrevs in enumerate(meta['abbrev_set']):
            for j, cls in enumerate(self.classes):
                if cls in abbrevs:
                    labels[i, j] = 1.0

        # Split into train/val/test
        indices = np.arange(len(meta))
        test_ratio = 1.0 - train_ratio - val_ratio

        # First split: train+val vs test
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=seed
        )
        # Second split: train vs val
        relative_val = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=relative_val, random_state=seed
        )

        if split == 'train':
            selected = train_idx
        elif split == 'val':
            selected = val_idx
        elif split == 'test':
            selected = test_idx
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.meta = meta.iloc[selected].reset_index(drop=True)
        self.labels = labels[selected]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]

        if self.cache_dir is not None:
            # Fast path: load preprocessed signal from .npy cache
            npy_path = os.path.join(self.cache_dir, f"{row['record_id']}.npy")
            signal = np.load(npy_path)
        else:
            # Slow path: read from WFDB + preprocess on the fly
            record_path = os.path.join(self.data_dir, row['record_path'])
            signal, _ = wfdb.rdsamp(record_path)
            signal = signal.T.astype(np.float32)
            signal = np.stack([
                preprocess(signal[i], fs=float(self.sampling_rate), duration=10.0)
                for i in range(signal.shape[0])
            ])

        if self.augment:
            # Gaussian noise
            signal += np.random.normal(0, 0.02, signal.shape).astype(np.float32)
            # Random amplitude scaling
            signal *= np.random.uniform(0.8, 1.2)

        return (
            torch.tensor(signal, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
