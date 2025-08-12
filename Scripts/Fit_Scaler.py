import os
import json
import numpy as np
from Scripts.dataset_windows import load_windows_streaming

# Fit_Scaler.py: Computes per-channel median & IQR for training subjects only, saves to disk
# RobustScaler helps EMG by reducing outlier impact and normalizing subject variability
# CLI: python Scripts/Fit_Scaler.py --csv data/your.csv --train_ids train_ids.txt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='Path to CSV file')
parser.add_argument('--train_ids', required=True, help='Path to train_ids.txt')
parser.add_argument('--out', default='artifacts/scaler.json', help='Output scaler path')
args = parser.parse_args()

# Read train subject IDs
with open(args.train_ids) as f:
    train_ids = set(int(line.strip()) for line in f if line.strip())

# First pass: collect all windows for training subjects
channel_data = [[] for _ in range(4)]  # Ch1..Ch4
for subject_id, gesture, window, start, end in load_windows_streaming(args.csv):
    if subject_id in train_ids:
        # Accumulate per-channel data block-wise (bounded memory)
        for ch in range(4):
            channel_data[ch].append(window[:, ch].ravel())

# Concatenate blocks for each channel
channel_data = [np.concatenate(ch_blocks) for ch_blocks in channel_data]

# Second pass: compute robust stats
medians = [float(np.median(ch)) for ch in channel_data]
iqrs = [float(np.subtract(*np.percentile(ch, [75, 25]))) for ch in channel_data]

# Enforce min IQR epsilon to avoid division by zero
min_epsilon = 1e-6
iqrs = [max(iqr, min_epsilon) for iqr in iqrs]

# Save scaler
scaler = {'median': medians, 'iqr': iqrs}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(scaler, f, indent=2)

print(f"Scaler saved to {args.out}:\nMedian: {medians}\nIQR: {iqrs}")

# Why RobustScaler? EMG signals have frequent outliers and subject-to-subject amplitude shifts.
# Median/IQR scaling is less sensitive to extreme values than mean/std, improving model robustness.
