# EMG All-CNN Training Pipeline (Step 1): Data Loading and Preprocessing
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# ------------------------------
# Configurable parameters
# ------------------------------
SAMPLE_RATE = 2000  # Hz
WINDOW_SIZE = 200   # 100 ms windows @ 2000 Hz
STRIDE = 100        # 50% overlap
N_CHANNELS = 10      # 4 real + 6 virtual sensors
N_CLASSES = 10       # Number of gestures to classify
GESTURE_FILES = {
    '0_REST': 0,
    '1_EXTENSION': 1,
    '2_FLEXION': 2,
    '3_ULNAR_DEVIATION': 3,
    '4_RADIAL_DEVIATION': 4,
    '5_GRIP': 5,
    '6_ABDUCTION': 6,
    '7_ADDUCTION': 7,
    '8_SUPPINATION': 8,
    '9_PRONATION': 9
}

# ------------------------------
# Function: Compute virtual channels
# ------------------------------
def compute_virtual_channels(data):
    """
    Given a [n_samples x 4] EMG array, compute 6 virtual channels
    as pairwise differences: (0-1, 0-2, 0-3, 1-2, 1-3, 2-3)
    Returns [n_samples x 10] array
    """
    ch0, ch1, ch2, ch3 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    virtuals = np.stack([
        ch0 - ch1,
        ch0 - ch2,
        ch0 - ch3,
        ch1 - ch2,
        ch1 - ch3,
        ch2 - ch3
    ], axis=1)
    return np.hstack([data, virtuals])  # shape: [n_samples, 10]


# ------------------------------
# Function: Window data
# ------------------------------
def window_data(data, label):
    """
    Converts continuous EMG data to overlapping windows with labels.
    Returns:
        - X_windows: [n_windows, 200, 10]
        - y_windows: [n_windows]
    """
    windows = []
    labels = []
    for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
        window = data[start:start + WINDOW_SIZE]
        if window.shape[0] == WINDOW_SIZE:
            windows.append(window)
            labels.append(label)
    return np.array(windows), np.array(labels)


# ------------------------------
# Function: Load and process all gesture CSVs
# ------------------------------
def load_dataset_from_csv(folder_path = os.path.join("Segregated_Data")):
    """
    Reads all gesture files, computes virtual sensors, windows the data,
    and returns the full dataset.
    """
    X_all, y_all = [], []

    for gesture_name, label in GESTURE_FILES.items():
        file_path = os.path.join(folder_path, f"{gesture_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data for gesture {gesture_name!r} at {file_path}")
        print(f"Loading {file_path}...")

        df = pd.read_csv(file_path)
        raw_data = df.values  # shape: [samples, 4]

        data_10ch = compute_virtual_channels(raw_data)  # shape: [samples, 10]
        X, y = window_data(data_10ch, label)  # shapes: [n_windows, 200, 10], [n_windows]

        X_all.append(X)
        y_all.append(y)

    # Concatenate all gestures
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y