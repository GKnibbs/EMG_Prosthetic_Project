import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from Scripts.Make_TFRecords_Features import get_dataset

# Train_Baseline_Features.py: Simple MLP for EMG gesture classification using feature vectors
# Model: Input [n_features] → Dense(64) → ReLU → Dropout → Dense(32) → ReLU → Dense(10, softmax)

n_features = 24  # 4 channels × 6 features each
batch_size = 64

# --- Load data ---
def get_feature_dataset(split, n_features=24, batch_size=64):
    tfrecord_dir = os.path.join('artifacts', 'tfrecords_features', split)
    ds = get_dataset(tfrecord_dir, n_features=n_features, batch_size=batch_size)
    ds = ds.map(lambda x, l, sid, st: (x, tf.one_hot(l, 10)), num_parallel_calls=tf.data.AUTOTUNE)
    if split == 'train':
        ds = ds.shuffle(10000)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- Build model ---
inputs = keras.Input(shape=(n_features,), name='feature_vector')
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Class weights ---
train_ds = get_feature_dataset('train', n_features, batch_size)
labels = []
for _, y in train_ds.unbatch().take(100000):
    labels.append(np.argmax(y.numpy()))
labels = np.array(labels)
class_counts = np.bincount(labels, minlength=10)
class_weights = {i: float(len(labels)) / (10 * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
print('Class weights:', class_weights)

# --- Training ---
val_ds = get_feature_dataset('val', n_features, batch_size)
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]
print('Model params:', model.count_params())

save_dir = os.path.join('artifacts', 'models', 'baseline_features')
os.makedirs(save_dir, exist_ok=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(os.path.join(save_dir, 'model'))
print(f'Model (features) saved to {save_dir}')

# This script mirrors Train_Baseline.py but uses feature vectors as input.
# To run: python Scripts\Train_Baseline_Features.py
