import numpy as np

# Standard EMG feature extraction for a window of shape [win_len, n_channels]
def extract_emg_features(window):
    # window: [win_len, n_channels]
    features = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(x))
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(x ** 2))
        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(x)))
        # Zero Crossings (ZC)
        zc = np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) > 0.01))
        # Slope Sign Changes (SSC)
        dx = np.diff(x)
        ssc = np.sum(((dx[:-1] * dx[1:]) < 0) & (np.abs(dx[:-1]) > 0.01) & (np.abs(dx[1:]) > 0.01))
        # Willison Amplitude (WAMP)
        wamp = np.sum(np.abs(np.diff(x)) > 0.01)
        features.extend([mav, rms, wl, zc, ssc, wamp])
    return np.array(features, dtype=np.float32)

# Feature vector length: n_channels * 6
# Features: [MAV, RMS, WL, ZC, SSC, WAMP] for each channel

# Extract temporal features for RNN/GRU: split window into n_steps blocks, extract features from each
def extract_temporal_features(window, fs=2000, n_steps=4):
    # window: [win_len, 10] (e.g., [100, 10] for 100ms at 2kHz)
    block_len = window.shape[0] // n_steps  # e.g., 25 samples for 25ms blocks
    temporal_features = []
    for i in range(n_steps):
        start = i * block_len
        end = start + block_len
        block = window[start:end, :]
        feats = extract_emg_features(block)  # [60] for 10 channels
        temporal_features.append(feats)
    return np.stack(temporal_features, axis=0)  # [n_steps, 60]
