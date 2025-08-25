
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Train_Baseline_VirtualChannels.py: 1D CNN for EMG gesture classification with 10 channels (4 real + 6 virtual)

def get_dataset(split, win_len=200, n_channels=10, batch_size=64):
	tfrecord_dir = os.path.join('artifacts', 'tfrecords_virtualchannels', split)
	files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
	from Scripts.Make_TFRecords_VirtualChannels import parse_fn
	ds = tf.data.TFRecordDataset(files)
	ds = ds.map(lambda x: parse_fn(x, win_len, n_channels), num_parallel_calls=tf.data.AUTOTUNE)
	ds = ds.map(lambda w, l, sid, st: (w, tf.one_hot(l, 10)), num_parallel_calls=tf.data.AUTOTUNE)
	if split == 'train':
		ds = ds.shuffle(10000)
	ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	return ds

win_len = 200
n_channels = 10
inputs = keras.Input(shape=(win_len, n_channels), name='emg_window')
x = layers.Conv1D(16, 7, padding='same')(inputs)
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

model.compile(
	optimizer=keras.optimizers.Adam(learning_rate=1e-3),
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

# --- Class weights ---
train_ds = get_dataset('train', win_len, n_channels)
labels = []
for _, y in train_ds.unbatch().take(100000):
	labels.append(np.argmax(y.numpy()))
labels = np.array(labels)
class_counts = np.bincount(labels, minlength=10)
class_weights = {i: float(len(labels)) / (10 * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
print('Class weights:', class_weights)

# --- Training ---
val_ds = get_dataset('val', win_len, n_channels)
callbacks = [
	keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]
print('Model params:', model.count_params())

save_dir = os.path.join('artifacts', 'models', 'baseline_virtualchannels')
os.makedirs(save_dir, exist_ok=True)

history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=100,
	class_weight=class_weights,
	callbacks=callbacks
)

model.save(os.path.join(save_dir, 'model'))
print(f'Model (virtual channels) saved to {save_dir}')
