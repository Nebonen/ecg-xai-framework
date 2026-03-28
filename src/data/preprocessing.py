import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 300.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to an ECG signal."""
    # Constant or near-constant signals cause filtfilt to produce NaN
    if signal.std() < 1e-10:
        return np.zeros_like(signal)
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = filtfilt(b, a, signal)
    # Replace any remaining NaN/Inf with zeros
    np.nan_to_num(filtered, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return filtered


def normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize a signal."""
    std = signal.std()
    return (signal - signal.mean()) / std if std > 0 else signal - signal.mean()


def segment(signal: np.ndarray, fs: float, duration: float) -> np.ndarray:
    """Crop or zero-pad signal to a fixed duration."""
    target_len = int(fs * duration)
    if len(signal) >= target_len:
        return signal[:target_len]
    return np.pad(signal, (0, target_len - len(signal)), mode="constant")


def preprocess(
    signal: np.ndarray,
    fs: float = 300.0,
    duration: float = 30.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> np.ndarray:
    """Full preprocessing pipeline: bandpass filter → z-score normalize → segment."""
    signal = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs)
    signal = normalize(signal)
    signal = segment(signal, fs=fs, duration=duration)
    return signal
