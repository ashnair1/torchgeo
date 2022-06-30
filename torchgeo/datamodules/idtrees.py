# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""IDTReeS datamodule."""

from typing import Any, Dict, List, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from torchgeo.datamodules.utils import dataset_split
from torchgeo.datasets.idtrees import IDTReeS

DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.6),
    K.RandomVerticalFlip(p=0.4),
    data_keys=["input", "bbox_xyxy"],
)


def collate_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten wrapper."""
    images = torch.stack([i["image"] for i in batch])
    r_batch = {"image": images}

    if "label" in batch[0]:
        r_batch["boxes"] = [b["boxes"].reshape(0, 4) if b["boxes"].numel() == 0 else b["boxes"] for b in batch]  # type: ignore[assignment]
        r_batch["labels"] = [b["label"].long() for b in batch]  # type: ignore[assignment]

    return r_batch


class IDTReeSDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the IDTReeS dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        augmentations: K.AugmentationSequential = DEFAULT_AUGS,
        predict_on: str = "test",
        task: str = "task1",
    ) -> None:
        """Initialize a LightningDataModule for IDTReeS based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the IDTReeS
                Dataset classes
            batch_size: The batch size used in the train DataLoader
                (val_batch_size == test_batch_size == 1)
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            patch_size: Size of random patch from image and mask (height, width)
            augmentations: Default augmentations applied
            predict_on: Directory/Dataset of images to run inference on
            task: 'task1' for detection, 'task2' for detection + classification
                (only relevant for split='test')
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.augmentations = augmentations
        self.predict_on = predict_on
        self.task = task

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        # RGB is int32 so divide by 255
        sample["image"] = sample["image"] / 255.0
        sample["image"] = torch.clip(sample["image"], min=0.0, max=1.0)

        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        train_dataset = IDTReeS(
            self.root_dir, split="train", transforms=self.preprocess, download=True
        )

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]
        self.test_dataset: Dataset[Any]

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

        assert self.predict_on == "test"
        self.predict_dataset = IDTReeS(
            self.root_dir,
            self.predict_on,
            self.task,
            transforms=self.preprocess,
            download=True,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for prediction."""
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=False,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Any], dataloader_idx: int
    ) -> Dict[str, Any]:
        """Apply augmentations to batch after transferring to GPU.

        Args:
            batch (dict): A batch of data that needs to be altered or augmented.
            dataloader_idx (int): The index of the dataloader to which the batch
            belongs.

        Returns:
            dict: A batch of data
        """
        # Training
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "training")
            and self.trainer.training
            and self.augmentations is not None
        ):

            batch["image"], batch["boxes"] = self.augmentations(
                batch["image"], batch["boxes"]
            )

        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.IDTreeS.plot`."""
        try:
            return self.val_dataset.dataset.plot(*args, **kwargs)  # type: ignore[attr-defined]
        except AttributeError:
            # If val_split_pct == 0.0
            return self.val_dataset.plot(*args, **kwargs)  # type: ignore[attr-defined]
