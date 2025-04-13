# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 8 dataset."""

from typing import ClassVar

from torchgeo.datasets.spacenet.base import SpaceNet

import glob
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch
from fiona.errors import FionaError, FionaValueError
from fiona.transform import transform_geom
from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from torch import Tensor

from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import (
    Path,
    check_integrity,
    extract_archive,
    percentile_normalization,
    which,
)


class SpaceNet9(NonGeoDataset, ABC):
    r"""SpaceNet9: Cross-Modal Satellite Imagery Registration.

    `SpaceNet 9 <https://spacenet.ai/sn9-challenge/>`_ is a dataset focusing on
    co-registering of optical and SAR imagery.

    If you use this dataset in your research, please cite the following paper:

    * https://elib.dlr.de/209238/1/IGARSS_24_SN9%20%285%29.pdf

    .. versionadded:: 0.7
    """

    url = 's3://spacenet9-challenge/{tarball}'
    dataset_id = "spacenet9"
    #directory_glob = '{product}'
    image_glob = '*.tif'
    tiepoints_glob = '*.csv'
    tarballs: ClassVar[dict[str, str]] = {
        'train': 'train.zip',
        'test': 'publictest.zip',
    }
    md5s: ClassVar[dict[str, str]] = {
        'train': '373f1607adffb1b0ba4976ec959ebcdf',
        'test': '6755f4cb45d145723f585e3912ca0435',
    }
    valid_aois: ClassVar[dict[str, list[str]]] = {'train': ['02', '03'], 'test': ['02', '03']}
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['optical', 'sar'],
        'test': ['optical', 'sar'],
    }
    valid_tiepoints = ('tiepoints',)
    # chip_size: ClassVar[dict[str, tuple[int, int]]] = {
    #     'PRE-event': (1300, 1300),
    #     'POST-event': (1300, 1300),
    # }
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        aois: list[int] = [],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet9 Dataset instance.

        Args:
            root: Root directory where dataset can be found.
            split: 'train' or 'test' split.
            aois: areas of interest
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            AssertionError: If any invalid arguments are passed.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.aois = aois or self.valid_aois[split]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        assert self.split in {'train', 'test'}
        self._verify()

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieve a sample from the dataset."""
        # Implement logic to load optical-SAR image pairs and tiepoints
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        # Implement logic to return the dataset size
        pass

    def _list_files(self) -> tuple[list[str], list[str]]:
        """List files in the dataset directory."""
        foldername = "publictest" if self.split == "test" else self.split
        root = os.path.join(self.root, self.dataset_id, foldername)
        filename = "publictest" if self.split == "test" else f"{self.split}_"
        optical_images = sorted(glob.glob(os.path.join(root, f"*_optical_{filename}{self.image_glob}")))
        sar_images = sorted(glob.glob(os.path.join(root, f"*_sar_{filename}{self.image_glob}")))
        if self.split == "train":
            tiepoints = sorted(glob.glob(os.path.join(root, f"*_tiepoints_{filename}{self.tiepoints_glob}")))
        else:
            tiepoints = None
        return optical_images, sar_images, tiepoints

    def _verify(self) -> None:
        """Verify the dataset integrity."""
        self.optical_images = []
        self.sar_images = []
        self.tiepoints = []
        root = os.path.join(self.root, self.dataset_id)
        os.makedirs(root, exist_ok=True)

        # Check if the extracted files already exist
        optical_images, sar_images, tiepoints = self._list_files()
        if optical_images and sar_images and (self.split != "train" or tiepoints):
            self.optical_images.extend(optical_images)
            self.sar_images.extend(sar_images)
            if self.split == "train":
                self.tiepoints.extend(tiepoints)
            return

        # Check if the tarball has already been downloaded
        tarball = self.tarballs[self.split]
        tarball_path = os.path.join(root, tarball)
        if os.path.exists(tarball_path):
            self._extract_and_list_files(tarball_path, root)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(f"Dataset not found in {root}. Set `download=True` to download it.")

        # Download the dataset
        self._download_and_verify(tarball, tarball_path, root)

    def _extract_and_list_files(self, tarball_path: str, root: str) -> None:
        """Extract the tarball and list files."""
        print("Extracting data...")
        extract_archive(tarball_path, root)
        optical_images, sar_images, tiepoints = self._list_files()
        self.optical_images.extend(optical_images)
        self.sar_images.extend(sar_images)
        if self.split == "train":
            self.tiepoints.extend(tiepoints)

    def _download_and_verify(self, tarball: str, tarball_path: str, root: str) -> None:
        """Download the dataset and verify its integrity."""
        url = self.url.format(tarball=tarball)
        aws = which("aws")
        aws("s3", "cp", url, root)
        check_integrity(tarball_path, self.md5s[self.split] if self.checksum else None)
        self._extract_and_list_files(tarball_path, root)


if __name__ == "__main__":
    # Example usage
    dataset = SpaceNet9(root='/home/ash1/Projects/torchgeo/test_data', split='test', download=True, checksum=True)
    print("Optical Images:", dataset.optical_images)
    print("Sar Images:", dataset.sar_images)
    print("Tiepoints:", dataset.tiepoints)