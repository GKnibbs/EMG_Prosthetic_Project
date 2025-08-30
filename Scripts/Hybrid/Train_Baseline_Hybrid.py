
import os
import numpy as np
import tensorflow as tf
from Scripts.Hybrid.HybridModel import build_hybrid_model
from Scripts.Hybrid.Dataset_Hybrid import get_hybrid_dataset

# --- Config ---
n_features = 60  # amplitude features
n_steps = 4
n_temporal_features = 60  # temporal features per step
num_classes = 10
batch_size = 64
epochs = 100
learning_rate = 1e-3

train_tfrecord_dir = os.path.join('artifacts', 'tfrecords_features_hybrid', 'train')
val_tfrecord_dir = os.path.join('artifacts', 'tfrecords_features_hybrid', 'val')
save_dir = os.path.join('artifacts', 'models', 'hybrid')
os.makedirs(save_dir, exist_ok=True)

# --- Data pipeline ---
train_ds = get_hybrid_dataset(train_tfrecord_dir, n_features, n_steps, n_temporal_features, batch_size, shuffle=True)
val_ds = get_hybrid_dataset(val_tfrecord_dir, n_features, n_steps, n_temporal_features, batch_size, shuffle=False)

# --- Compute class weights ---
labels = []
for (_, _), y, _, _ in train_ds.unbatch().take(100000):
    labels.append(y.numpy())
labels = np.array(labels)
class_counts = np.bincount(labels, minlength=num_classes)
class_weights = {i: float(len(labels)) / (num_classes * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
print('Class weights:', class_weights)

# --- Build model ---
model = build_hybrid_model(n_features=n_features, n_steps=n_steps, n_temporal_features=n_temporal_features, n_classes=num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print('Model params:', model.count_params())

# --- Callbacks ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'model.keras'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

# --- Training ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# --- Save final model ---
model.save(os.path.join(save_dir, 'final_model.keras'))
print(f'Training complete. Model saved to {save_dir}')
