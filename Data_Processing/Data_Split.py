
# Data_Split.py: Randomly splits subject IDs (1..40) into train/val/test and writes to txt files
# Usage: python Scripts/Data_Split.py --n_subjects 40 --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --out_dir artifacts

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_subjects', type=int, default=40, help='Total number of subject IDs (default: 40)')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Fraction of subjects for training')
parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction for validation')
parser.add_argument('--test_ratio', type=float, default=0.15, help='Fraction for test')
parser.add_argument('--out_dir', default='artifacts', help='Output directory for split files')
args = parser.parse_args()

# Generate subject IDs 1..n_subjects
all_ids = np.arange(1, args.n_subjects + 1)
np.random.seed(42)
np.random.shuffle(all_ids)
n = len(all_ids)
n_train = int(n * args.train_ratio)
n_val = int(n * args.val_ratio)
train_ids = all_ids[:n_train]
val_ids = all_ids[n_train:n_train+n_val]
test_ids = all_ids[n_train+n_val:]

# Write to files
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, 'train_ids.txt'), 'w') as f:
    for i in train_ids:
        f.write(f"{i}\n")
with open(os.path.join(args.out_dir, 'val_ids.txt'), 'w') as f:
    for i in val_ids:
        f.write(f"{i}\n")
with open(os.path.join(args.out_dir, 'test_ids.txt'), 'w') as f:
    for i in test_ids:
        f.write(f"{i}\n")

print(f"Split complete: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test IDs written to {args.out_dir}")
