#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 2

dataset_id = 'spacenet9'

profile = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'width': SIZE,
    'height': SIZE,
    'count': 1,
    'crs': CRS.from_epsg(4326),
    'transform': Affine(
        0.0001,
        0.0,
        54.3,
        0.0,
        -0.0001,
        24.5,
    ),
}

np.random.seed(0)
Z = np.random.randint(np.iinfo('uint8').max, size=(SIZE, SIZE), dtype='uint8')


def compute_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# Remove existing data if it exists
if os.path.exists(dataset_id):
    shutil.rmtree(dataset_id)

# Create train dataset
train_path = os.path.join(dataset_id, 'train')
os.makedirs(train_path, exist_ok=True)

# Generate train files matching real dataset structure:
# - AOI 02: 2 samples (02_optical_train_01.tif, 02_optical_train_02.tif)
# - AOI 03: 1 sample (03_optical_train_01.tif)
# Total: 3 samples
train_samples = [
    ('02', '01'),
    ('02', '02'),
    ('03', '01'),
]

for aoi, sample_id in train_samples:
    # Create optical image
    optical_file = os.path.join(train_path, f'{aoi}_optical_train_{sample_id}.tif')
    with rasterio.open(optical_file, 'w', **profile) as src:
        src.write(Z, 1)

    # Create SAR image
    sar_file = os.path.join(train_path, f'{aoi}_sar_train_{sample_id}.tif')
    with rasterio.open(sar_file, 'w', **profile) as src:
        src.write(Z, 1)

    # Create tiepoints CSV matching real SpaceNet9 format
    # Format: sar_row,sar_col,optical_row,optical_col
    tiepoints_file = os.path.join(train_path, f'{aoi}_tiepoints_train_{sample_id}.csv')
    tiepoints_data = pd.DataFrame({
        'sar_row': [0.4, 1.6],      # sar_y
        'sar_col': [0.6, 1.4],      # sar_x
        'optical_row': [0.5, 1.5],  # optical_y
        'optical_col': [0.5, 1.5],  # optical_x
    })
    tiepoints_data.to_csv(tiepoints_file, index=False)

# Create test dataset (note: folder is 'publictest' not 'test')
test_path = os.path.join(dataset_id, 'publictest')
os.makedirs(test_path, exist_ok=True)

# Generate test files matching real dataset structure:
# - AOI 02: 1 sample (02_optical_publictest.tif)
# - AOI 03: 1 sample (03_optical_publictest.tif)
# Total: 2 samples
for aoi in ['02', '03']:
    # Create optical image
    optical_file = os.path.join(test_path, f'{aoi}_optical_publictest.tif')
    with rasterio.open(optical_file, 'w', **profile) as src:
        src.write(Z, 1)

    # Create SAR image
    sar_file = os.path.join(test_path, f'{aoi}_sar_publictest.tif')
    with rasterio.open(sar_file, 'w', **profile) as src:
        src.write(Z, 1)

# Create zip archives
print('Creating train.zip...')
shutil.make_archive(
    os.path.join(dataset_id, 'train'),
    'zip',
    root_dir=dataset_id,
    base_dir='train',
)

print('Creating testpublic.zip...')
shutil.make_archive(
    os.path.join(dataset_id, 'testpublic'),
    'zip',
    root_dir=dataset_id,
    base_dir='publictest',
)

# Compute and print MD5 checksums
print('\nMD5 Checksums:')
train_zip = os.path.join(dataset_id, 'train.zip')
if os.path.exists(train_zip):
    print(f'train.zip: {compute_md5(train_zip)}')

testpublic_zip = os.path.join(dataset_id, 'testpublic.zip')
if os.path.exists(testpublic_zip):
    print(f'testpublic.zip: {compute_md5(testpublic_zip)}')
