
import numpy as np
import pandas as pd
from Scripts.Utils_Signal import safe_chunk
from Scripts.Config import trim_head_ms, trim_tail_ms

# Generator: streams a CSV, yields sliding windows for each subject
# Each window: [win_len, 10] (4 real + 6 virtual channels)
def load_windows_streaming(csv_path, fs=2000, win_ms=100, hop_ms=50):
	win_len = int(fs * win_ms / 1000)
	hop_len = int(fs * hop_ms / 1000)
	chunk_size = 10000
	subject_buffers = {}
	gesture = None
	for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
		if gesture is None:
			import os
			base = os.path.basename(csv_path)
			gesture = base.split('_')[0] if '_' in base else base.split('.')[0]
		for subject_id, sub_df in chunk.groupby('iD'):
			data = sub_df[['ch1', 'ch2', 'ch3', 'ch4']].values
			if subject_id not in subject_buffers:
				subject_buffers[subject_id] = np.empty((0, 4))
			subject_buffers[subject_id] = np.vstack([subject_buffers[subject_id], data])
	for subject_id, buf in subject_buffers.items():
		trim_head = int(trim_head_ms * fs / 1000)
		trim_tail = int(trim_tail_ms * fs / 1000)
		buf = buf[trim_head:buf.shape[0]-trim_tail if trim_tail > 0 else buf.shape[0]]
		start = 0
		while start + win_len <= buf.shape[0]:
			window = buf[start : start+win_len]  # [win_len, 4]
			# Compute 6 virtual channels (all pairwise differences)
			ch1 = window[:, 0]
			ch2 = window[:, 1]
			ch3 = window[:, 2]
			ch4 = window[:, 3]
			v1 = ch1 - ch2
			v2 = ch1 - ch3
			v3 = ch1 - ch4
			v4 = ch2 - ch3
			v5 = ch2 - ch4
			v6 = ch3 - ch4
			virtuals = np.stack([v1, v2, v3, v4, v5, v6], axis=1)  # [win_len, 6]
			window_10ch = np.concatenate([window, virtuals], axis=1)  # [win_len, 10]
			yield (subject_id, gesture, window_10ch, start+trim_head, start+trim_head+win_len)
			start += hop_len
		# No need to keep leftovers < win_len
