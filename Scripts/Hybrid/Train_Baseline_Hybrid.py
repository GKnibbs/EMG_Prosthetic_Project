import os
import tensorflow as tf
from Scripts.Hybrid.HybridModel import build_hybrid_model
from Scripts.Hybrid.Dataset_Hybrid import get_dataset

# Configurations
TRAIN_TFRECORD = 'path/to/train_hybrid.tfrecord'  # Update with actual path
VAL_TFRECORD = 'path/to/val_hybrid.tfrecord'      # Update with actual path
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = 'checkpoints/hybrid_model'
NUM_CLASSES = 10
N_AMPLITUDE_FEATURES = 24
N_TEMPORAL_FEATURES = 60

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Data pipeline
dtrain = get_dataset(TRAIN_TFRECORD, batch_size=BATCH_SIZE, shuffle=True)
dval = get_dataset(VAL_TFRECORD, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = build_hybrid_model(n_amplitude_features=N_AMPLITUDE_FEATURES, n_temporal_features=N_TEMPORAL_FEATURES, num_classes=NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# Training
history = model.fit(
    dtrain,
    validation_data=dval,
    epochs=EPOCHS,
    callbacks=[ckpt_cb, earlystop_cb]
)

# Save final model
model.save(os.path.join(CHECKPOINT_DIR, 'final_model.h5'))
print('Training complete. Model saved to', CHECKPOINT_DIR)
