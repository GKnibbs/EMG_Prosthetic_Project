import tensorflow as tf
import os

# Path to your virtual channel TFRecords (test set)
tfrecord_dir = os.path.join('artifacts', 'tfrecords_virtualchannels', 'test')
files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))

# Import the correct parse_fn for 10 channels
from Scripts.Redundant.Make_TFRecords_VirtualChannels import parse_fn

win_len = 200
n_channels = 10

ds = tf.data.TFRecordDataset(files)
ds = ds.map(lambda x: parse_fn(x, win_len, n_channels))

count = 0
for window, label, subject_id, start_sample in ds.take(5):
    print("Window shape:", window.shape)
    print("Label:", label.numpy())
    print("Subject ID:", subject_id.numpy())
    print("Start sample:", start_sample.numpy())
    count += 1

print("Total records read:", count)