# EMG All-CNN Training Pipeline (Step 1): Data Loading and Preprocessing
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from keras.models import load_model

# Progress bars
from keras.callbacks import Callback
from tqdm import tqdm

# Early stopping - avoiding overfitting
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
    '8_SUPINATION': 8,
    '9_PRONATION': 9
}

# Loading bar for each epoch for QoL
class TQDMProgressBar(Callback):
    """
    Custom Keras Callback that wraps each epoch with a tqdm progress bar.
    """
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.epochs}")

    def on_train_batch_begin(self, batch, logs=None):
        self.progress_bar = tqdm(total=self.params['steps'], unit='batch', leave=False)

    def on_train_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()

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

# ------------------------------
# Function: Normalize and split dataset
# ------------------------------
def normalize_and_split(X, y, test_size=0.2):
    """
    Normalize dataset using global per-channel z-score, then split into train/test sets.
    Returns:
        - X_train, X_test: normalized inputs
        - y_train_cat, y_test_cat: one-hot encoded targets
        - scaler_stats: (mean, std) tuple for deployment
    """
    # Flatten time dimension to normalize per channel across all windows
    X_flat = X.reshape(-1, X.shape[-1])  # shape: [N * 200, 10]
    
    # Compute mean and std
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8  # epsilon to avoid division by zero

    # Normalize each window
    X_norm = (X - mean) / std

    # Train/test split (stratified by gesture)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=test_size, stratify=y, random_state=42
    )

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=N_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=N_CLASSES)

    return X_train, X_test, y_train_cat, y_test_cat, (mean, std)

# ------------------------------
# Save normalization parameters
# ------------------------------
def save_normalization_stats(mean, std, output_file="normalization_params.npz"):
    """
    Save mean and std as .npz file to use in deployment (e.g. STM32 firmware).
    """
    np.savez(output_file, mean=mean, std=std)
    print(f"Saved normalization parameters to {output_file}")

# ------------------------------
# Function: Build CNN model
# ------------------------------

def build_cnn_model(input_shape=(200, 10), num_classes=10):
    """
    Builds a compact CNN for EMG gesture classification.
    Input: [200 timesteps, 10 channels]
    """
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model

if __name__ == "__main__":
    # Load data
    X, y = load_dataset_from_csv()

    # Normalize and split
    X_train, X_test, y_train_cat, y_test_cat, (mean, std) = normalize_and_split(X, y)

    print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")

    # Saving mean and std for embedded deployment:
    save_normalization_stats(mean, std)

    # Training Model
    model = build_cnn_model(input_shape=(200, 10), num_classes=10)

    # Directory to save best model - early stopping
    checkpoint_path = "best_model.h5"
    callbacks = [TQDMProgressBar(),  
                 # Save model with lowest validation loss
                ModelCheckpoint(filepath=checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1),
    
                # Stop if no improvement for 5 epochs
                EarlyStopping(monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1)]
    
    model.fit(X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=20,
            batch_size=64,
            callbacks=callbacks,
            verbose=0)
    
    # Evaluation of models
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # Load model
    model = load_model("best_model.h5")

    model.save("final_cnn_model.h5")

