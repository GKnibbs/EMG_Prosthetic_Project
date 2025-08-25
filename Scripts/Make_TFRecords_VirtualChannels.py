
import os
import numpy as np
import tensorflow as tf
import json
import argparse
from Scripts.Dataset_Windows_VirtualChannels import load_windows_streaming
from Scripts.Utils_Scaling import apply_median_iqr

# Make_TFRecords_VirtualChannels.py: Streams windows (10 channels), applies robust scaling, writes TFRecords
# Each TFRecord contains: window (float32, [win,10]), label (int8), subject_id (int8), start_sample (int64)

SPLITS = ['train', 'val', 'test']
SPLIT_IDS = {split: os.path.join('EMG_Prosthetic_Project', 'artifacts', f'{split}_ids.txt') for split in SPLITS}
TFRECORD_DIR = os.path.join('artifacts', 'tfrecords_virtualchannels')
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

def serialize_example(window, label, subject_id, start_sample):
	window_bytes = window.astype(np.float32).tobytes()
	feature = {
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
			label = gesture_to_id(gesture)
			example = serialize_example(window_scaled, label, subject_id, start)
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
	print(f"TFRecords (virtual channels) for {split} written to {out_dir}")

if __name__ == '__main__':
	for split in SPLITS:
		write_tfrecords(split)

def parse_fn(example_proto, win_len=200, n_channels=10):
	features = {
		'window': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.int64),
		'subject_id': tf.io.FixedLenFeature([], tf.int64),
		'start_sample': tf.io.FixedLenFeature([], tf.int64)
	}
	parsed = tf.io.parse_single_example(example_proto, features)
	window = tf.io.decode_raw(parsed['window'], tf.float32)
	window = tf.reshape(window, [win_len, n_channels])
	label = tf.cast(parsed['label'], tf.int32)
	subject_id = tf.cast(parsed['subject_id'], tf.int32)
	start_sample = tf.cast(parsed['start_sample'], tf.int64)
	return window, label, subject_id, start_sample

def get_dataset(tfrecord_dir, win_len=200, n_channels=10, batch_size=32, shuffle=True, cache=False):
	files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
	ds = tf.data.TFRecordDataset(files)
	ds = ds.map(lambda x: parse_fn(x, win_len, n_channels), num_parallel_calls=tf.data.AUTOTUNE)
	if shuffle:
		ds = ds.shuffle(10000)
	ds = ds.batch(batch_size)
	if cache:
		ds = ds.cache()
	ds = ds.prefetch(tf.data.AUTOTUNE)
	return ds
