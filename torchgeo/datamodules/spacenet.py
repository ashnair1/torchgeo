# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet trainers."""

import abc
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch import Generator, Tensor  # type: ignore[attr-defined]
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms as T

from torchgeo.datasets import SpaceNet2


class SpaceNetDataModule(pl.LightningDataModule, abc.ABC):
    """Base LightningDataModule implementation for SpaceNet datasets."""

    def __init__(
        self,
        root_dir: str,
        image: str,
        collections: List[str],
        bands: List[int] = [4, 2, 1],  # [5, 3, 2]
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 4,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize a LightningDataModule for SpaceNet DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the respective SpaceNet Datasets
                      class
            image: The ``image`` argument to pass to the SpaceNet Dataset class
            collections: The ``collection`` argument to pass to the respective SpaceNet
                         Dataset class
            bands: Band indexes to be used if multispectral image is used.
            seed: The seed value to use when doing the train-val split
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            api_key: The RadiantEarth MLHub API key to use if the dataset needs to be
                downloaded
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.image = image
        self.collections = collections
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.api_key = api_key
        # TODO: Remove once pretrained backbones with C>3 are ready
        assert len(bands) == 3
        self.bands = bands

    @abc.abstractmethod
    def custom_transform(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Transform a single sample from the Dataset."""

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """

    @abc.abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val splits.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    # TODO: We need a test_dataloader as well


class SpaceNet2DataModule(SpaceNetDataModule):
    """LightningDataModule implementation for SpaceNet 1 dataset."""

    def __init__(
        self,
        root_dir: str,
        image: str,
        collections: List[str] = [],
        bands: List[int] = [4, 2, 1],  # [5, 3, 2]
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 4,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize a LightningDataModule for SpaceNet 1 DataLoader.

        Args:
            root_dir: The ``root`` arugment to pass to the SpaceNet2 Datasets classes
            image: The ``image`` argument to pass to the SpaceNet2 Dataset class
            collections: collection selection which must be a subset of:
                         [sn2_AOI_2_Vegas, sn2_AOI_3_Paris, sn2_AOI_4_Shanghai,
                         sn2_AOI_5_Khartoum]
            bands: Band indexes to be used if multispectral image is used.
            seed: The seed value to use when doing the train-val split
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            api_key: The RadiantEarth MLHub API key to use if the dataset needs to be
                downloaded
        """
        super().__init__(
            root_dir, image, collections, bands, seed, batch_size, num_workers, api_key
        )

    def custom_transform(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Transform a single sample from the Dataset."""
        # size = sample["image"].shape[-1]
        # n = size + 32/2
        # n = int(n - (n%32))
        # size = (n, n)

        # Size should be divisble by 32
        size = (160, 160) if self.image == "MS" else (640, 640)
        # Select three bands if multispectral image
        if self.image in {"MS", "PS-MS"}:
            sample["image"] = sample["image"][self.bands, ...]

        # scale to [0,1]
        sample["image"] = sample["image"] / 255.0

        # Resize to be divisible by 32
        rsz = T.Resize(size)
        sample["image"] = rsz(sample["image"])
        sample["mask"] = rsz(sample["mask"].unsqueeze(0)).squeeze()
        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        # Shouldn't download be true? Also adiant earth
        # can be enabled as a profile so you can download
        # w/o api key
        do_download = self.api_key is not None
        _ = SpaceNet2(
            self.root_dir,
            self.image,
            self.collections,
            self.custom_transform,
            download=do_download,
            api_key=self.api_key,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val splits.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """
        self.dataset = SpaceNet2(
            self.root_dir,
            self.image,
            self.collections,
            self.custom_transform,
            download=False,
            api_key=self.api_key,
        )

        # TODO: Choose one of these methods
        # Method 1
        train_ids, val_ids = train_test_split(np.arange(len(self.dataset)))
        self.train_dataset = Subset(self.dataset, train_ids)
        self.val_dataset = Subset(self.dataset, val_ids)

        # Method 2
        train_size = 0.75
        test_size = 1 - train_size

        split = [int(i * len(self.dataset)) for i in [train_size, test_size]]
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, split, generator=Generator().manual_seed(self.seed)
        )


if __name__ == "__main__":
    sn2 = SpaceNet2DataModule(
        root_dir="/home/ashwin/Desktop/Projects/torchgeo/data/spacenet2",
        collections=["sn2_AOI_5_Khartoum"],
        image="MS",
    )

    sn2.prepare_data()
    sn2.setup()
