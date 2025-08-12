import os
import numpy as np
import tensorflow as tf
import json
from Scripts.dataset_windows import load_windows_streaming
from Scripts.Utils_Scaling import apply_median_iqr

# Make_TFRecords.py: Streams windows, applies robust scaling, writes TFRecords for fast TF/Keras input
# Each TFRecord contains: window (float32, [win,4]), label (int8), subject_id (int8), start_sample (int64)
# Shards output by ~100MB files for memory safety

# --- Config ---
SPLITS = ['train', 'val', 'test']
SPLIT_IDS = {split: f'{split}_ids.txt' for split in SPLITS}
TFRECORD_DIR = os.path.join('artifacts', 'tfrecords')
CSV_DIR = 'data'  # Update if needed
SCALER_PATH = 'artifacts/scaler.json'
SHARD_SIZE_MB = 100

# --- Load scaler ---
with open(SCALER_PATH) as f:
    scaler = json.load(f)
med = np.array(scaler['median'], dtype=np.float32)
iqr = np.array(scaler['iqr'], dtype=np.float32)

# --- Load split IDs ---
split_subjects = {}
for split, path in SPLIT_IDS.items():
    with open(path) as f:
        split_subjects[split] = set(int(line.strip()) for line in f if line.strip())

# --- Helper: TFRecord serialization ---
def serialize_example(window, label, subject_id, start_sample):
    # Serialize window as raw float32 bytes
    window_bytes = window.astype(np.float32).tobytes()
    feature = {
        'window': tf.train.Feature(bytes_list=tf.train.BytesList(value=[window_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'subject_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[subject_id])),
        'start_sample': tf.train.Feature(int64_list=tf.train.Int64List(value=[start_sample]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

# --- Helper: gesture label to int (0..9) ---
def gesture_to_id(gesture):
    # Map gesture string to int (customize as needed)
    gesture_map = {f'gesture_{i}': i for i in range(10)}
    return gesture_map.get(gesture, -1)

# --- Main TFRecord writer ---
def write_tfrecords(split):
    out_dir = os.path.join(TFRECORD_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    shard_idx = 0
    n_bytes = 0
    writer = None
    # Stream all CSVs
    for fname in os.listdir(CSV_DIR):
        if not fname.lower().endswith('.csv'):
            continue
        fpath = os.path.join(CSV_DIR, fname)
        for subject_id, gesture, window, start, end in load_windows_streaming(fpath):
            if subject_id not in split_subjects[split]:
                continue
            # Apply robust scaling
            window_scaled = apply_median_iqr(window, med, iqr)
            label = gesture_to_id(gesture)
            # Serialize example
            example = serialize_example(window_scaled, label, subject_id, start)
            # Shard by ~100MB
            if writer is None or n_bytes > SHARD_SIZE_MB * 1024 * 1024:
                if writer:
                    writer.close()
                shard_path = os.path.join(out_dir, f'{split}_{shard_idx:03d}.tfrecord')
                writer = tf.io.TFRecordWriter(shard_path)
                n_bytes = 0
                shard_idx += 1
            writer.write(example)
            n_bytes += len(example)
    if writer:
        writer.close()
    print(f"TFRecords for {split} written to {out_dir}")

# --- Run for all splits ---
if __name__ == '__main__':
    for split in SPLITS:
        write_tfrecords(split)

# --- TF parse_fn and tf.data pipeline example ---
def parse_fn(example_proto, win_len=200):
    # Parse features from TFRecord
    features = {
        'window': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
        'start_sample': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    # Deserialize window
    window = tf.io.decode_raw(parsed['window'], tf.float32)
    window = tf.reshape(window, [win_len, 4])
    label = tf.cast(parsed['label'], tf.int32)
    subject_id = tf.cast(parsed['subject_id'], tf.int32)
    start_sample = tf.cast(parsed['start_sample'], tf.int64)
    return window, label, subject_id, start_sample

# Example tf.data pipeline for training:
def get_dataset(tfrecord_dir, win_len=200, batch_size=32, shuffle=True, cache=False):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(lambda x: parse_fn(x, win_len), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Serialization details:
# - window: float32, shape [win,4], serialized as raw bytes for compactness
# - label: int8 gesture id (0..9), subject_id: int8, start_sample: int64
# - Sharding by ~100MB keeps memory usage low and enables parallel loading
# - parse_fn deserializes window and metadata for model input
# - tf.data pipeline supports shuffle (train), batch, prefetch, cache
