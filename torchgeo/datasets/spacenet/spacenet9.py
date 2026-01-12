# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 9 dataset."""

import glob
import os
from collections.abc import Callable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import (
    Path,
    download_url,
    extract_archive,
    percentile_normalization,
)


class SpaceNet9(NonGeoDataset):
    r"""SpaceNet9: Cross-Modal Satellite Imagery Registration.

    `SpaceNet 9 <https://spacenet.ai/sn9-challenge/>`_ is a dataset for cross-modal
    satellite imagery registration, focusing on aligning optical and SAR imagery
    in earthquake-affected regions for disaster response applications.

    The dataset contains high-resolution imagery from:

    * Optical: Maxar Open Data Program (3-channel RGB, 0.3-0.5m resolution)
    * SAR: UMBRA satellite (single-channel, 0.3-0.5m resolution)

    Manually labeled tie-points are provided for training to evaluate registration
    quality. The challenge involves computing pixel-wise spatial transformations to
    align imagery across modalities for downstream tasks like damage assessment and
    change detection.

    Dataset details:

    * Train split: 3 samples (AOI 02: 2 images, AOI 03: 1 image)
    * Test split: 2 samples (AOI 02 and 03, one each)
    * Each sample includes optical image, SAR image, and tie-points (train only)
    * Images have width and height < 13000 pixels

    If you use this dataset in your research, please cite the following paper:

    * https://elib.dlr.de/209238/1/IGARSS_24_SN9%20%285%29.pdf

    Challenge details: https://www.topcoder.com/challenges/9620f66a-767e-40ac-81d5-5cc61274b186

    .. versionadded:: 0.9
    """

    url = 'https://spacenet-dataset.s3.us-east-1.amazonaws.com/spacenet/SN9_cross-modal/{tarball}'
    dataset_id = 'spacenet9'
    # directory_glob = '{product}'
    image_glob = '*.tif'
    tiepoints_glob = '*.csv'
    tarballs: ClassVar[dict[str, str]] = {
        'train': 'train.zip',
        'test': 'testpublic.zip',
    }
    md5s: ClassVar[dict[str, str]] = {
        'train': '373f1607adffb1b0ba4976ec959ebcdf',
        'test': '6755f4cb45d145723f585e3912ca0435',
    }
    valid_aois: ClassVar[dict[str, list[str]]] = {
        'train': ['02', '03'],
        'test': ['02', '03'],
    }
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
        """Retrieve a sample from the dataset.

        Args:
            index: index to return

        Returns:
            data dictionary containing optical and SAR images, and optionally tiepoints
        """
        # Load optical image
        optical_path = self.optical_images[index]
        with rasterio.open(optical_path) as src:
            optical = src.read()
            optical = torch.from_numpy(optical).float()

        # Load SAR image
        sar_path = self.sar_images[index]
        with rasterio.open(sar_path) as src:
            sar = src.read()
            sar = torch.from_numpy(sar).float()

        sample = {'optical': optical, 'sar': sar}

        # Load tiepoints if available (training split only)
        if self.split == 'train' and self.tiepoints:
            tiepoints_path = self.tiepoints[index]
            tiepoints_df = pd.read_csv(tiepoints_path)
            # SpaceNet9 format: sar_row,sar_col,optical_row,optical_col
            # Convert to [optical_x, optical_y, sar_x, sar_y] format
            # Note: row=y, col=x
            tiepoints = torch.zeros((len(tiepoints_df), 4), dtype=torch.float32)
            tiepoints[:, 0] = torch.from_numpy(
                tiepoints_df['optical_col'].values
            ).float()  # optical_x
            tiepoints[:, 1] = torch.from_numpy(
                tiepoints_df['optical_row'].values
            ).float()  # optical_y
            tiepoints[:, 2] = torch.from_numpy(
                tiepoints_df['sar_col'].values
            ).float()  # sar_x
            tiepoints[:, 3] = torch.from_numpy(
                tiepoints_df['sar_row'].values
            ).float()  # sar_y
            sample['tiepoints'] = tiepoints

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.optical_images)

    def _list_files(self) -> tuple[list[str], list[str], list[str] | None]:
        """List files in the dataset directory."""
        foldername = 'publictest' if self.split == 'test' else self.split
        root = os.path.join(self.root, self.dataset_id, foldername)
        filename = 'publictest' if self.split == 'test' else f'{self.split}_'
        optical_images = sorted(
            glob.glob(os.path.join(root, f'*_optical_{filename}{self.image_glob}'))
        )
        sar_images = sorted(
            glob.glob(os.path.join(root, f'*_sar_{filename}{self.image_glob}'))
        )
        if self.split == 'train':
            tiepoints = sorted(
                glob.glob(
                    os.path.join(root, f'*_tiepoints_{filename}{self.tiepoints_glob}')
                )
            )
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
        if optical_images and sar_images and (self.split != 'train' or tiepoints):
            self.optical_images.extend(optical_images)
            self.sar_images.extend(sar_images)
            if self.split == 'train':
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
            raise DatasetNotFoundError(
                f'Dataset not found in {root}. Set `download=True` to download it.'
            )

        # Download and extract the dataset
        self._download(tarball, root)
        self._extract_and_list_files(tarball_path, root)

    def _extract_and_list_files(self, tarball_path: str, root: str) -> None:
        """Extract the tarball and list files."""
        print('Extracting data...')
        extract_archive(tarball_path, root)
        optical_images, sar_images, tiepoints = self._list_files()
        self.optical_images.extend(optical_images)
        self.sar_images.extend(sar_images)
        if self.split == 'train':
            self.tiepoints.extend(tiepoints)

    def _download(self, tarball: str, root: str) -> None:
        """Download the dataset."""
        import shutil

        url = self.url.format(tarball=tarball)
        md5 = self.md5s[self.split] if self.checksum else None

        # Check if URL is a local path (for testing)
        if url.startswith(('http://', 'https://', 's3://')):
            # Remote URL - download
            download_url(url, root, md5=md5)

        elif os.path.exists(url):
            shutil.copy(url, os.path.join(root, tarball))
        else:
            raise FileNotFoundError(f'Local file not found: {url}')

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
        show_tiepoints: bool = True,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            show_tiepoints: flag indicating whether to draw tiepoints and connection lines

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # Get optical and SAR images
        optical = sample['optical'].numpy()
        sar = sample['sar'].numpy()

        # Normalize for visualization
        # For optical: use first 3 channels if multi-channel
        if optical.shape[0] >= 3:
            optical_rgb = np.transpose(optical[:3], (1, 2, 0))
        else:
            optical_rgb = np.transpose(optical, (1, 2, 0))
        optical_rgb = percentile_normalization(optical_rgb, axis=(0, 1))

        # For SAR: typically single channel, display as grayscale
        if sar.shape[0] == 1:
            sar_display = sar[0]
        else:
            # If multiple channels, use first channel
            sar_display = sar[0]

        # Normalize SAR for display
        sar_display = (sar_display - sar_display.min()) / (
            sar_display.max() - sar_display.min() + 1e-8
        )

        # Get dimensions - both images should be the same size
        optical_h, optical_w = optical_rgb.shape[:2]
        sar_h, sar_w = sar_display.shape[:2]

        # Use the maximum dimensions to ensure both images fit
        max_h = max(optical_h, sar_h)
        max_w = max(optical_w, sar_w)

        # Create figure with 2 columns: Optical (left) and SAR (right)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot optical image on the left with explicit extent
        axs[0].imshow(optical_rgb, extent=[0, max_w, max_h, 0])
        axs[0].set_xlim(0, max_w)
        axs[0].set_ylim(max_h, 0)
        axs[0].set_aspect('equal')
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title('Optical')

        # Plot SAR image on the right with explicit extent
        axs[1].imshow(sar_display, cmap='gray', extent=[0, max_w, max_h, 0])
        axs[1].set_xlim(0, max_w)
        axs[1].set_ylim(max_h, 0)
        axs[1].set_aspect('equal')
        axs[1].axis('off')
        if show_titles:
            axs[1].set_title('SAR')

        # Adjust subplot spacing to minimize gap
        plt.subplots_adjust(wspace=0.05)

        # Plot tiepoints if available and requested
        if show_tiepoints and 'tiepoints' in sample:
            tiepoints = sample['tiepoints'].numpy()

            # Plot tiepoints on Optical image (left)
            axs[0].scatter(
                tiepoints[:, 0],
                tiepoints[:, 1],
                c='red',
                s=50,
                marker='o',
                edgecolors='white',
                linewidths=1.5,
                label='Optical points',
                zorder=10,
            )

            # Plot tiepoints on SAR image (right)
            axs[1].scatter(
                tiepoints[:, 2],
                tiepoints[:, 3],
                c='cyan',
                s=50,
                marker='o',
                edgecolors='white',
                linewidths=1.5,
                label='SAR points',
                zorder=10,
            )

            # Draw lines connecting corresponding points across the two subplots
            for i in range(len(tiepoints)):
                optical_x, optical_y = tiepoints[i, 0], tiepoints[i, 1]
                sar_x, sar_y = tiepoints[i, 2], tiepoints[i, 3]

                # Use ConnectionPatch to draw lines between subplots
                con = ConnectionPatch(
                    xyA=(optical_x, optical_y),
                    xyB=(sar_x, sar_y),
                    coordsA='data',
                    coordsB='data',
                    axesA=axs[0],
                    axesB=axs[1],
                    color='yellow',
                    linewidth=2,
                    linestyle='--',
                    alpha=0.8,
                    zorder=1,
                )
                fig.add_artist(con)

            axs[0].legend(loc='upper left', fontsize=8)
            axs[1].legend(loc='upper right', fontsize=8)

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig


if __name__ == '__main__':
    # Example usage
    dataset = SpaceNet9(
        root='/home/ash1/Projects/torchgeo/test_data',
        split='train',
        download=True,
        checksum=True,
    )
    print('Optical Images:', dataset.optical_images)
    print('Sar Images:', dataset.sar_images)
    print('Tiepoints:', dataset.tiepoints)
    sample = dataset[1]
    dataset.plot(sample, show_tiepoints=True)
    plt.show()
