import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Evaluate.py: Honest per-class and per-subject performance for EMG gesture classification
# Loads saved model, runs on test TFRecords, outputs metrics, confusion matrix PNG, per-subject accuracy CSV

# --- Load model ---
model_dir = os.path.join('artifacts', 'models', 'baseline', 'model')
model = tf.keras.models.load_model(model_dir)

# --- Load test data ---
from Scripts.BaseModel.Make_TFRecords import parse_fn
win_len = 200  # Update if needed
batch_size = 64

tfrecord_dir = os.path.join('artifacts', 'tfrecords', 'test')
files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
ds = tf.data.TFRecordDataset(files)
ds = ds.map(lambda x: parse_fn(x, win_len), num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(batch_size)

# --- Collect predictions and true labels ---
y_true, y_pred, subject_ids = [], [], []
for batch in ds:
    windows, labels, sids, _ = batch
    probs = model.predict(windows)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)
    subject_ids.extend(sids.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
subject_ids = np.array(subject_ids)

# --- Metrics ---
# Macro/micro accuracy
acc_macro = accuracy_score(y_true, y_pred)
# Per-class precision/recall/F1
report = classification_report(y_true, y_pred, output_dict=True)

# --- Confusion matrix ---
cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.savefig(os.path.join('artifacts', 'confusion_matrix.png'))
plt.close()

# --- Per-subject accuracy ---
subject_acc = {}
for sid in np.unique(subject_ids):
    mask = subject_ids == sid
    subject_acc[sid] = accuracy_score(y_true[mask], y_pred[mask])
pd.DataFrame({'subject_id': list(subject_acc.keys()), 'accuracy': list(subject_acc.values())}) \
    .to_csv(os.path.join('artifacts', 'subject_accuracy.csv'), index=False)

# --- Print analysis ---
print('Macro accuracy:', acc_macro)
print('Classification report:')
for cls, stats in report.items():
    if cls.isdigit():
        print(f'Gesture {cls}: Precision={stats["precision"]:.2f}, Recall={stats["recall"]:.2f}, F1={stats["f1-score"]:.2f}')
print('Confusion matrix saved to artifacts/confusion_matrix.png')
print('Per-subject accuracy saved to artifacts/subject_accuracy.csv')

# Short analysis: which gestures confuse, where class weights helped/hurt
# - Look for off-diagonal values in confusion matrix: high confusion means gestures are similar or hard to distinguish
# - Class weights may help rare gestures (higher recall), but can hurt precision if overcompensated
# - Review per-class F1 and confusion matrix to guide future improvements

# Thorough comments:
# - Loads model and test TFRecords for honest evaluation
# - Computes macro/micro accuracy, per-class metrics for detailed insight
# - Saves normalized confusion matrix as PNG for visual inspection
# - Per-subject accuracy table helps spot subject-specific issues
# - Prints concise analysis to guide next steps
