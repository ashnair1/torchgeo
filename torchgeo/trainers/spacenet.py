# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet trainers."""

import abc
from typing import Any, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import Generator, Tensor  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, IoU
from torchvision import transforms as T

from torchgeo.datasets import SpaceNet1
from torchgeo.models import FCN


# LightningModules
class SpaceNetSegmentationTask(pl.LightningModule):
    """LightningModule for training models on the SpaceNet datasets.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    in_channels = 3
    classes = 2

    def config_task(self, kwargs: Any) -> None:
        """Configures the task based on kwargs parameters."""
        if kwargs["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=self.in_channels,
                classes=self.classes,
            )
        elif kwargs["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                encoder_output_stride=kwargs["encoder_output_stride"],
                in_channels=self.in_channels,
                classes=self.classes,
            )
        elif kwargs["segmentation_model"] == "fcn":
            self.model = FCN(self.in_channels, self.classes)
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()  # type: ignore[attr-defined]
        elif kwargs["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            encoder_output_stride: The output stride parameter in DeepLabV3+ models
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task(kwargs)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_iou = IoU(num_classes=self.classes)
        self.val_iou = IoU(num_classes=self.classes)
        self.test_iou = IoU(num_classes=self.classes)

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_accuracy(y_hat_hard, y)
        self.train_iou(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc", self.train_accuracy.compute())
        self.log("train_iou", self.train_iou.compute())
        self.train_accuracy.reset()
        self.train_iou.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss)
        self.val_accuracy(y_hat_hard, y)
        self.val_iou(y_hat_hard, y)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                batch["image"][0].cpu().numpy(), 0, 3
            )
            mask = batch["mask"][0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask, vmin=0, vmax=4)
            axs[1].axis("off")
            axs[2].imshow(pred, vmin=0, vmax=4)
            axs[2].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc", self.val_accuracy.compute())
        self.log("val_iou", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_accuracy(y_hat_hard, y)
        self.test_iou(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc", self.test_accuracy.compute())
        self.log("test_iou", self.test_iou.compute())
        self.test_accuracy.reset()
        self.test_iou.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler."""
        optimizer: torch.optim.optimizer.Optimizer
        if self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
            )
        elif self.hparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                momentum=0.9,
                weight_decay=1e-2,
            )
        else:
            raise ValueError(
                f"Optimizer choice '{self.hparams['optimizer']}' is not valid."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
                "verbose": True,
            },
        }


# DataModules
class SpaceNetDataModule(pl.LightningDataModule, abc.ABC):
    """Base LightningDataModule implementation for SpaceNet datasets."""

    def __init__(
        self,
        root_dir: str,
        image: str,
        bands: List[int] = [4, 2, 1],  # [5, 3, 2]
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 4,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize a LightningDataModule for SpaceNet DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the SpaceNet1 Datasets classes
            image: The ``image`` argument to pass to the SpaceNet 1 Dataset class
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


class SpaceNet1DataModule(SpaceNetDataModule):
    """LightningDataModule implementation for SpaceNet 1 dataset."""

    def __init__(
        self,
        root_dir: str,
        image: str,
        bands: List[int] = [4, 2, 1],  # [5, 3, 2]
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 4,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize a LightningDataModule for SpaceNet 1 DataLoader.

        Args:
            root_dir: The ``root`` arugment to pass to the SpaceNet1 Datasets classes
            image: The ``image`` argument to pass to the SpaceNet 1 Dataset class
            bands: Band indexes to be used if multispectral image is used.
            seed: The seed value to use when doing the train-val split
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            api_key: The RadiantEarth MLHub API key to use if the dataset needs to be
                downloaded
        """
        super().__init__(root_dir, image, bands, seed, batch_size, num_workers, api_key)

    def custom_transform(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Transform a single sample from the Dataset."""
        # Select three bands from multispectral image & estimate size
        if self.image == "8band":
            size = (96, 96)
            sample["image"] = sample["image"][self.bands, ...]
        else:
            size = (416, 416)

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
        _ = SpaceNet1(
            self.root_dir,
            self.image,
            self.custom_transform,
            download=do_download,
            api_key=self.api_key,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val splits.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """
        self.dataset = SpaceNet1(
            self.root_dir,
            self.image,
            self.custom_transform,
            download=False,
            api_key=self.api_key,
        )

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
    sn1 = SpaceNet1DataModule(
        root_dir="/media/ashwin/DATA2/torchgeo/data/spacenet1",
        image="rgb",
    )

    sn1.prepare_data()
    sn1.setup()
