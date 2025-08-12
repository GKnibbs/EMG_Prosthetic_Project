import numpy as np
import pandas as pd

# Configuration
gesture_files = {
    0: '0_REST.csv',
    1: '1_EXTENSION.csv',
    2: '2_FLEXION.csv',
    3: '3_ULNAR_DEVIATION.csv',
    4: '4_RADIAL_DEVIATION.csv',
    5: '5_GRIP.csv',
    6: '6_ABDUCTION.csv',
    7: '7_ADDUCTION.csv',
    8: '8_SUPINATION.csv',
    9: '9_PRONATION.csv'
}

fs = 2000
segment_length_sec = 6
segment_size = fs * segment_length_sec  # 12000

X = []
y = []
ids = []

for gesture_label, filename in gesture_files.items():
    print(f'Processing {filename}...')
    df = pd.read_csv(filename)

    # Extract only the channels and ID
    signal = df[['ch1', 'ch2', 'ch3', 'ch4']].values
    iD_col = df['iD'].values

    total_rows = len(signal)
    num_segments = total_rows // segment_size

    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size

        segment = signal[start:end]           # shape: (12000, 4)
        iD_segment = iD_col[start:end]

        if len(segment) == segment_size:
            X.append(segment)
            y.append(gesture_label)
            ids.append(iD_segment[0])  # assumes all 12000 rows in a block have same iD

# Convert to numpy arrays
X = np.stack(X)            # shape: (N_samples, 12000, 4)
y = np.array(y)            # shape: (N_samples,)
ids = np.array(ids)        # shape: (N_samples,)

# Optional: transpose to (N_samples, 4, 12000) if needed
# X = np.transpose(X, (0, 2, 1))

# Save to disk
np.save('X_raw.npy', X)
np.save('y_labels.npy', y)
np.save('iD_labels.npy', ids)

print(f"Saved {X.shape[0]} samples.")
