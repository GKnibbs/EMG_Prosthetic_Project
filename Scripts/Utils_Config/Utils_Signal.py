import numpy as np

# Helper: safely extract a chunk from an array (returns zeros if out-of-bounds)
def safe_chunk(arr, start, length):
    # If chunk is fully in bounds, return slice
    if start >= 0 and start + length <= arr.shape[0]:
        return arr[start : start+length]
    # If out-of-bounds, pad with zeros
    out = np.zeros((length, arr.shape[1]), dtype=arr.dtype)
    valid_len = max(0, min(length, arr.shape[0] - start))
    if valid_len > 0 and start < arr.shape[0]:
        out[:valid_len] = arr[start : start+valid_len]
    return out

# Helper: chunk an array into non-overlapping windows
def chunk_array(arr, win_len):
    # Returns a list of windows (may drop last incomplete)
    n_windows = arr.shape[0] // win_len
    return [arr[i*win_len:(i+1)*win_len] for i in range(n_windows)]

# Helper: sliding windows with hop
def sliding_windows(arr, win_len, hop_len):
    # Returns a list of windows (may drop last incomplete)
    windows = []
    start = 0
    while start + win_len <= arr.shape[0]:
        windows.append(arr[start:start+win_len])
        start += hop_len
    return windows
