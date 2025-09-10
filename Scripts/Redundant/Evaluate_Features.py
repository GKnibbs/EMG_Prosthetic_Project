import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Scripts.Redundant.Make_TFRecords_Features import parse_fn, get_dataset

# Evaluate_Features.py: Evaluate feature-based EMG gesture classifier
# Loads saved model, runs on test TFRecords (features), outputs metrics, confusion matrix PNG, per-subject accuracy CSV

# --- Load model ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'artifacts')
model_dir = os.path.join(ARTIFACTS_DIR, 'models', 'baseline_features', 'model')
model = tf.keras.models.load_model(model_dir)

# --- Load test data ---
n_features = 24  # 4 channels × 6 features
batch_size = 64
tfrecord_dir = os.path.join('artifacts', 'tfrecords_features', 'test')
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
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.title('Normalized Confusion Matrix (Features)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.savefig(os.path.join('artifacts', 'confusion_matrix_features.png'))
plt.close()

# --- Per-subject accuracy ---
subject_acc = {}
for sid in np.unique(subject_ids):
    mask = subject_ids == sid
    subject_acc[sid] = accuracy_score(y_true[mask], y_pred[mask])
pd.DataFrame({'subject_id': list(subject_acc.keys()), 'accuracy': list(subject_acc.values())}) \
    .to_csv(os.path.join(ARTIFACTS_DIR, 'subject_accuracy_features.csv'), index=False)

# --- Print analysis ---
print('Macro accuracy:', acc_macro)
print('Classification report:')
for cls, stats in report.items():
    if cls.isdigit():
        print(f'Gesture {cls}: Precision={stats["precision"]:.2f}, Recall={stats["recall"]:.2f}, F1={stats["f1-score"]:.2f}')
print('Confusion matrix saved to artifacts/confusion_matrix_features.png')
print('Per-subject accuracy saved to artifacts/subject_accuracy_features.csv')

# To run: python Scripts/Evaluate_Features.py
