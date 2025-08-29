import os
import numpy as np
import tensorflow as tf
from Scripts.Hybrid.Make_TFRecords_Hybrid import gesture_to_id

# Dataset_Hybrid.py: Data pipeline for hybrid amplitude-temporal model
# - For each 100ms window: 
#   amplitude_features: [n_features] (e.g., 60)
#   temporal_features: [n_steps, n_temporal_features] (e.g., 4x24)
#   label: int
#   subject_id: int
#   start_sample: int

def parse_hybrid_example(example_proto, n_features=60, n_steps=4, n_temporal_features=40):
    feature_description = {
        'amplitude_features': tf.io.FixedLenFeature([n_features], tf.float32),
        'temporal_features': tf.io.FixedLenFeature([n_steps * n_temporal_features], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
        'start_sample': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    amp = parsed['amplitude_features']
    temp = tf.reshape(parsed['temporal_features'], [n_steps, n_temporal_features])
    label = tf.cast(parsed['label'], tf.int32)
    subject_id = tf.cast(parsed['subject_id'], tf.int32)
    start_sample = tf.cast(parsed['start_sample'], tf.int64)
    return (amp, temp), label, subject_id, start_sample


def get_hybrid_dataset(tfrecord_dir, n_features=60, n_steps=4, n_temporal_features=24, batch_size=32, shuffle=True, cache=False):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(lambda x: parse_hybrid_example(x, n_features, n_steps, n_temporal_features), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Note: You will need a TFRecord writer that outputs both amplitude and temporal features for each window.
# This script provides the dataset loading and parsing logic for the hybrid model.
