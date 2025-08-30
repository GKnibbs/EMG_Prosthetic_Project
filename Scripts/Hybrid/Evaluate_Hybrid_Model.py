import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Scripts.Hybrid.Dataset_Hybrid import get_hybrid_dataset
from Scripts.Hybrid.HybridModel import build_hybrid_model

# --- Load model ---
model_dir = os.path.join('artifacts', 'models', 'hybrid', 'model.keras')
model = tf.keras.models.load_model(model_dir)

# --- Load test data ---
n_features = 60
n_steps = 4
n_temporal_features = 60
batch_size = 64
tfrecord_dir = os.path.join('artifacts', 'tfrecords_features_hybrid', 'test')
ds = get_hybrid_dataset(tfrecord_dir, n_features, n_steps, n_temporal_features, batch_size, shuffle=False)

# --- Collect predictions and true labels ---
y_true, y_pred = [], []
for (amp, temp), labels in ds:
    probs = model.predict([amp, temp])
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

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
plt.title('Normalized Confusion Matrix (Hybrid Model)')
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
plt.savefig(os.path.join('artifacts', 'confusion_matrix_hybrid.png'))
plt.close()


# --- Per-subject accuracy removed: subject_id not available in new dataset format ---

# --- Print analysis ---
print('Macro accuracy:', acc_macro)
print('Classification report:')
for cls, stats in report.items():
    if cls.isdigit():
        print(f'Gesture {cls}: Precision={stats["precision"]:.2f}, Recall={stats["recall"]:.2f}, F1={stats["f1-score"]:.2f}')
print('Confusion matrix saved to artifacts/confusion_matrix_hybrid.png')
print('Per-subject accuracy saved to artifacts/subject_accuracy_hybrid.csv')
