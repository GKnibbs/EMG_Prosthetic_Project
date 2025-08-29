import os
import csv
import pandas as pd
import numpy as np

# This script scans huge CSVs in the data/ folder without ever fully loading them into memory.
# It builds a manifest.csv with: gesture label, file path, subject IDs present, total samples, estimated trials, and sampling rate.
# All validation and stats are done in a streaming fashion for efficiency.

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Segregated_Data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, 'manifest.csv')

# Make sure the artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Helper: get gesture label from filename (assumes gesture in filename)
def get_gesture_label(filename):
    # Example: 0_REST.csv -> 0
    base = os.path.basename(filename)
    return base.split('_')[0] if '_' in base else base.split('.')[0]

# Helper: estimate sampling rate (assume 2000Hz unless found in file)
def get_sampling_rate():
    # For now, hardcode 2000Hz as per project spec
    return 2000

# Helper: stream a CSV and collect stats
def scan_csv(path):
    # We'll use pandas read_csv with chunksize for streaming
    chunk_size = 10000  # Tune as needed
    ids = set()
    n_rows = 0
    n_nans = 0
    n_out_of_range = 0
    sample_ids = set()
    columns_ok = False
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        # Validate columns on first chunk
        if not columns_ok:
            required = ['iD', 'ch1', 'ch2', 'ch3', 'ch4']
            if not all(col in chunk.columns for col in required):
                print(f"[WARN] {path}: missing required columns!")
                return None
            columns_ok = True
        # Count rows
        n_rows += len(chunk)
        # Unique subject IDs
        ids.update(chunk['iD'].unique())
        # NaN check
        n_nans += chunk.isna().sum().sum()
        # Out-of-range check (±5V surrogate)
        for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
            n_out_of_range += ((chunk[ch] < -5) | (chunk[ch] > 5)).sum()
        # Sample a few IDs for manifest
        if len(sample_ids) < 5:
            sample_ids.update(chunk['iD'].unique())
            if len(sample_ids) > 5:
                sample_ids = set(list(sample_ids)[:5])
    return {
        'n_rows': n_rows,
        'n_ids': len(ids),
        'ids_sample': ';'.join(str(i) for i in list(ids)[:5]),
        'n_nans': n_nans,
        'n_out_of_range': n_out_of_range,
        'fs_hz': get_sampling_rate(),
        'approx_duration_sec': round(n_rows / get_sampling_rate(), 2),
        'ids': ids
    }

# Scan all CSVs in data/
manifest_rows = []
if not os.path.exists(DATA_DIR):
    print(f"[ERROR] Data directory not found: {DATA_DIR}")
else:
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith('.csv'):
            fpath = os.path.join(DATA_DIR, fname)
            gesture = get_gesture_label(fname)
            stats = scan_csv(fpath)
            if stats is None:
                continue
            manifest_rows.append({
                'gesture': gesture,
                'file': fpath,
                'n_rows': stats['n_rows'],
                'n_ids': stats['n_ids'],
                'ids_sample': stats['ids_sample'],
                'fs_hz': stats['fs_hz'],
                'approx_duration_sec': stats['approx_duration_sec']
            })
            # Print concise summary for each file
            print(f"{fname}: {stats['n_rows']} rows, {stats['n_ids']} subjects, NaNs: {stats['n_nans']}, out-of-range: {stats['n_out_of_range']}")

# Write manifest.csv
with open(MANIFEST_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['gesture', 'file', 'n_rows', 'n_ids', 'ids_sample', 'fs_hz', 'approx_duration_sec'])
    writer.writeheader()
    for row in manifest_rows:
        writer.writerow(row)

print(f"Manifest written to {MANIFEST_PATH} ({len(manifest_rows)} files scanned)")
