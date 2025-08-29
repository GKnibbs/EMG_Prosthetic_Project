
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Scripts.Make_TFRecords_Features_VirtualChannels import parse_fn, get_dataset

# Evaluate_Features_VirtualChannels.py: Evaluate feature-based EMG gesture classifier (10 channels, 60 features)

# --- Load model ---
model_dir = os.path.join('artifacts', 'models', 'baseline_features_virtualchannels', 'model.keras')
model = tf.keras.models.load_model(model_dir)

# --- Load test data ---
n_features = 60  # 10 channels × 6 features
batch_size = 64
tfrecord_dir = os.path.join('artifacts', 'tfrecords_features_virtualchannels', 'test')
files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
ds = tf.data.TFRecordDataset(files)
ds = ds.map(lambda x: parse_fn(x, n_features), num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(batch_size)

# --- Collect predictions and true labels ---
y_true, y_pred, subject_ids = [], [], []
for batch in ds:
	features, labels, sids, _ = batch
	probs = model.predict(features)
	preds = np.argmax(probs, axis=1)
	y_true.extend(labels.numpy())
	y_pred.extend(preds)
	subject_ids.extend(sids.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
subject_ids = np.array(subject_ids)

# --- Metrics ---
acc_macro = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)

# --- Confusion matrix ---
gesture_names = [
	'Rest',
	'Extension',
	'Flexion',
	'Ulnar Deviation',
	'Radial Deviation',
	'Grip',
	'Abduction',
	'Adduction',
	'Supination',
	'Pronation'
]
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(10,8))
im = plt.imshow(cm, cmap='Blues')
plt.title('Normalized Confusion Matrix (Features, Virtual Channels)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(np.arange(10), gesture_names, rotation=45, ha='right')
plt.yticks(np.arange(10), gesture_names)

# Show numbers in each cell
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
	for j in range(cm.shape[1]):
		plt.text(j, i, f'{cm[i, j]:.2f}',
				 ha='center', va='center',
				 color='white' if cm[i, j] > thresh else 'black', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join('artifacts', 'confusion_matrix_features_virtualchannels.png'))
plt.close()

# --- Per-subject accuracy ---
subject_acc = {}
for sid in np.unique(subject_ids):
	mask = subject_ids == sid
	subject_acc[sid] = accuracy_score(y_true[mask], y_pred[mask])
pd.DataFrame({'subject_id': list(subject_acc.keys()), 'accuracy': list(subject_acc.values())}) \
	.to_csv(os.path.join('artifacts', 'subject_accuracy_features_virtualchannels.csv'), index=False)

# --- Print analysis ---
print('Macro accuracy:', acc_macro)
print('Classification report:')
for cls, stats in report.items():
	if cls.isdigit():
		print(f'Gesture {cls}: Precision={stats["precision"]:.2f}, Recall={stats["recall"]:.2f}, F1={stats["f1-score"]:.2f}')
print('Confusion matrix saved to artifacts/confusion_matrix_features_virtualchannels.png')
print('Per-subject accuracy saved to artifacts/subject_accuracy_features_virtualchannels.csv')
