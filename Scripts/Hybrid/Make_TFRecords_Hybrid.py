
import os
import numpy as np
import tensorflow as tf
import json
import argparse
from Scripts.Hybrid.Dataset_Windows_Hybrid import load_windows_streaming
from Scripts.Utils_Config.Utils_Scaling import apply_median_iqr
from Scripts.Utils_Config.Utils_Features import extract_emg_features, extract_temporal_features


# Make_TFRecords_Hybrid.py: Streams 10-channel windows, extracts features, writes TFRecords
# Each TFRecord: feature_vector (float32, [60]), label (int8), subject_id (int8), start_sample (int64)
# Each TFRecord contains both amplitude and temporal features
# 4 Temporal features per time step (100ms)

SPLITS = ['train', 'val', 'test']
SPLIT_IDS = {split: os.path.join('EMG_Prosthetic_Project', 'artifacts', f'{split}_ids.txt') for split in SPLITS}
TFRECORD_DIR = os.path.join('artifacts', 'tfrecords_features_hybrid')
SCALER_PATH = 'artifacts/scaler.json'
SHARD_SIZE_MB = 100

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='Directory containing per-gesture CSV files')
args, _ = parser.parse_known_args()

with open(SCALER_PATH) as f:
	scaler = json.load(f)
med = np.array(scaler['median'], dtype=np.float32)
iqr = np.array(scaler['iqr'], dtype=np.float32)

split_subjects = {}
for split, path in SPLIT_IDS.items():
	with open(path) as f:
		split_subjects[split] = set(int(line.strip()) for line in f if line.strip())

def serialize_example(amplitude_features, temporal_features, windows, label, subject_id, start_sample):
	window_bytes = windows.astype(np.float32).tobytes()
	feature = {
		'amplitude_features': tf.train.Feature(float_list=tf.train.FloatList(value=amplitude_features.flatten())),
		'temporal_features': tf.train.Feature(float_list=tf.train.FloatList(value=temporal_features.flatten())),
		'window': tf.train.Feature(bytes_list=tf.train.BytesList(value=[window_bytes])),
		'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
		'subject_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(subject_id)])),
		'start_sample': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(start_sample)]))
	}
	example = tf.train.Example(features=tf.train.Features(feature=feature))
	return example.SerializeToString()

def gesture_to_id(gesture):
	try:
		return int(gesture)
	except Exception:
		return -1

def write_tfrecords(split):
	out_dir = os.path.join(TFRECORD_DIR, split)
	os.makedirs(out_dir, exist_ok=True)
	shard_idx = 0
	n_bytes = 0
	writer = None
	for fname in os.listdir(args.data_dir):
		if not fname.lower().endswith('.csv'):
			continue
		fpath = os.path.join(args.data_dir, fname)
		for subject_id, gesture, window, start, end in load_windows_streaming(fpath):
			if subject_id not in split_subjects[split]:
				continue
			# Only scale the first 4 channels (real sensors)
			window_scaled = window.copy()
			window_scaled[:, :4] = apply_median_iqr(window[:, :4], med, iqr)
			# Extract features for all 10 channels
			amplitude_feature = extract_emg_features(window_scaled)
			# Extract temporal features (4x25ms blocks)
			temporal_features = extract_temporal_features(window_scaled, fs=2000, n_steps=4)
			label = gesture_to_id(gesture)
			example = serialize_example(amplitude_feature, temporal_features, window_scaled, label, subject_id, start)
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
	print(f"TFRecords (features, virtual channels) for {split} written to {out_dir}")

if __name__ == '__main__':
	for split in SPLITS:
		write_tfrecords(split)

def parse_fn(example_proto, n_features=60):
	features = {
		'feature_vector': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.int64),
		'subject_id': tf.io.FixedLenFeature([], tf.int64),
		'start_sample': tf.io.FixedLenFeature([], tf.int64)
	}
	parsed = tf.io.parse_single_example(example_proto, features)
	feature_vector = tf.io.decode_raw(parsed['feature_vector'], tf.float32)
	feature_vector = tf.reshape(feature_vector, [n_features])
	label = tf.cast(parsed['label'], tf.int32)
	subject_id = tf.cast(parsed['subject_id'], tf.int32)
	start_sample = tf.cast(parsed['start_sample'], tf.int64)
	return feature_vector, label, subject_id, start_sample

def get_dataset(tfrecord_dir, n_features=60, batch_size=32, shuffle=True, cache=False):
	files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
	ds = tf.data.TFRecordDataset(files)
	ds = ds.map(lambda x: parse_fn(x, n_features), num_parallel_calls=tf.data.AUTOTUNE)
	if shuffle:
		ds = ds.shuffle(10000)
	ds = ds.batch(batch_size)
	if cache:
		ds = ds.cache()
	ds = ds.prefetch(tf.data.AUTOTUNE)
	return ds
