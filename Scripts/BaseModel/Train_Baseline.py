import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Train_Baseline.py: Compact 1D CNN for EMG gesture classification, designed for embedded constraints
# Model: Input [win, 4] → Conv1D(16, 7) → BN → ReLU → Conv1D(32, 5) → BN → ReLU → GlobalAvgPool1D → Dense(64) → Dropout(0.2) → Dense(10, softmax)

# --- Load data ---
# For demo, use tf.data pipeline from Make_TFRecords.py (update path as needed)
def get_dataset(split, win_len=200, batch_size=64):
    tfrecord_dir = os.path.join('artifacts', 'tfrecords', split)
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    from Scripts.BaseModel.Make_TFRecords import parse_fn
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(lambda x: parse_fn(x, win_len), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda w, l, sid, st: (w, tf.one_hot(l, 10)), num_parallel_calls=tf.data.AUTOTUNE)
    if split == 'train':
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Build model ---
win_len = 200  # Update if needed
inputs = keras.Input(shape=(win_len, 4), name='emg_window')
x = layers.Conv1D(16, 7, padding='same')(inputs)
# BatchNorm helps stabilize training
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv1D(32, 5, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64)(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# --- Compile ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Class weights ---
# Compute class weights from training label histogram
train_ds = get_dataset('train', win_len)
labels = []
for _, y in train_ds.unbatch().take(100000):  # Limit for speed
    labels.append(np.argmax(y.numpy()))
labels = np.array(labels)
class_counts = np.bincount(labels, minlength=10)
class_weights = {i: float(len(labels)) / (10 * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
# Class weights: higher loss weight for minority classes to counter imbalance
print('Class weights:', class_weights)

# --- Training ---
val_ds = get_dataset('val', win_len)
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
]
print('Model params:', model.count_params())

# Save dir
save_dir = os.path.join('artifacts', 'models', 'baseline')
os.makedirs(save_dir, exist_ok=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save model
model.save(os.path.join(save_dir, 'model.keras'))
print(f'Model saved to {save_dir}')

# Line-by-line comments:
# - Model is compact for embedded deployment, but expressive enough to learn EMG features
# - Class weights help address class imbalance by upweighting rare gestures
# - EarlyStopping prevents overfitting and restores best weights
# - All params and results are logged for reproducibility

# Windows command to run (CPU or GPU):
# python Scripts\Train_Baseline.py
