# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet trainers."""

import abc
import os
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from torchgeo.datamodules.utils import dataset_split
from torchgeo.datasets import SpaceNet2
from torchgeo.datasets.utils import PredictDataset


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
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        predict_on: Optional[str] = None,
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
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            predict_on: Directory of images to run inference on
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
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.predict_on = predict_on
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

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for predicting."""
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )


class SpaceNet2DataModule(SpaceNetDataModule):
    """LightningDataModule implementation for SpaceNet2 dataset.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        root_dir: str,
        image: str,
        collections: List[str] = [],
        bands: List[int] = [4, 2, 1],  # [5, 3, 2]
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        predict_on: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize a LightningDataModule for SpaceNet2 DataLoader.

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
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            predict_on: Directory of images to run inference on
            api_key: The RadiantEarth MLHub API key to use if the dataset needs to be
                downloaded
        """
        super().__init__(
            root_dir,
            image,
            collections,
            bands,
            seed,
            batch_size,
            num_workers,
            val_split_pct,
            test_split_pct,
            predict_on,
            api_key,
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
        if "mask" in sample:
            sample["mask"] = rsz(sample["mask"].unsqueeze(0)).squeeze()

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        do_download = len(os.listdir(self.root_dir)) == 0
        _ = SpaceNet2(
            self.root_dir,
            self.image,
            self.collections,
            self.custom_transform,
            download=do_download,
            api_key=self.api_key,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        train_dataset = SpaceNet2(
            self.root_dir, self.image, self.collections, self.custom_transform
        )

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]
        self.test_dataset: Dataset[Any]
        self.predict_dataset: Dataset[Any]

        if self.val_split_pct > 0.0:
            if self.test_split_pct > 0.0:
                self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                    train_dataset,
                    val_pct=self.val_split_pct,
                    test_pct=self.test_split_pct,
                )
            else:
                self.train_dataset, self.val_dataset = dataset_split(
                    train_dataset, val_pct=self.val_split_pct
                )
                self.test_dataset = self.val_dataset

        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset
            self.test_dataset = train_dataset
        if self.predict_on:
            self.predict_dataset = PredictDataset(
                self.predict_on, self.custom_transform
            )
