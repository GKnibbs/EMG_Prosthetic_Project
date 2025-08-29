
import numpy as np
import pandas as pd
from Scripts.Utils_Signal import safe_chunk
from Scripts.Config import trim_head_ms, trim_tail_ms

# Generator: streams a CSV, yields sliding windows for each subject
# Inputs: csv_path, fs=2000, win_ms=100, hop_ms=50
# For each subject, slide fixed windows (no trial alignment) across their signal
# Each yield: (subject_id, gesture, window, start_sample, end_sample)
def load_windows_streaming(csv_path, fs=2000, win_ms=100, hop_ms=50):
    # Optionally trim N ms from start/end of each subject's stream to reduce transition residue
    # Trimming helps remove ambiguous frames at gesture boundaries, improving training quality
    # Calculate window and hop in samples
    win_len = int(fs * win_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)
    # Stream CSV in chunks
    chunk_size = 10000  # Tune for memory
    # We'll collect rows per subject, but flush as soon as enough for a window
    subject_buffers = {}
    gesture = None
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Get gesture label from filename (e.g., 0_REST.csv -> 0)
        if gesture is None:
            import os
            base = os.path.basename(csv_path)
            gesture = base.split('_')[0] if '_' in base else base.split('.')[0]
        # Group rows by subject
        for subject_id, sub_df in chunk.groupby('iD'):
            # Only keep Ch1..Ch4 columns
            data = sub_df[['ch1', 'ch2', 'ch3', 'ch4']].values
            # Buffer for this subject
            if subject_id not in subject_buffers:
                subject_buffers[subject_id] = np.empty((0, 4))
            subject_buffers[subject_id] = np.vstack([subject_buffers[subject_id], data])
    # After all chunks, process each subject buffer
    for subject_id, buf in subject_buffers.items():
        # Apply trimming to reduce ambiguous transition frames
        trim_head = int(trim_head_ms * fs / 1000)
        trim_tail = int(trim_tail_ms * fs / 1000)
        # Only keep samples after head trim and before tail trim
        buf = buf[trim_head:buf.shape[0]-trim_tail if trim_tail > 0 else buf.shape[0]]
        # Slide windows over trimmed buffer
        start = 0
        while start + win_len <= buf.shape[0]:
            # Careful index math: window is buf[start : start+win_len]
            window = buf[start : start+win_len]
            yield (subject_id, gesture, window, start+trim_head, start+trim_head+win_len)
            start += hop_len
        # No need to keep leftovers < win_len

# Line-by-line comments:
# - Calculate window/hop in samples for precise indexing
# - Buffer rows per subject to avoid cross-subject leakage
# - After streaming, trim head/tail samples to reduce ambiguous transition frames
# - Slide windows with careful start/end math, yield only full windows
# - Window indices are offset by trim_head to reflect true sample positions
