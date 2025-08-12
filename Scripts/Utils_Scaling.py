import numpy as np

# apply_median_iqr: scales window (x - med) / iqr
# RobustScaler helps EMG by reducing outlier impact and normalizing subject variability
# x: [win_len, 4], med: [4], iqr: [4]
def apply_median_iqr(x, med, iqr):
    # Subtract median and divide by IQR for each channel
    # This makes features less sensitive to outliers and subject amplitude shifts
    return (x - med) / iqr

# Why RobustScaler?
# - EMG signals often have spikes/outliers due to movement artifacts or electrode noise
# - Subject amplitude varies widely (skin, muscle, placement)
# - Median/IQR scaling is robust to these, unlike mean/std
# - Improves generalization and stability for ML models
