# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""xView2 dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional, Tuple

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib import patches
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, draw_semantic_segmentation_masks, extract_archive


class XView(NonGeoDataset):
    r"""xView dataset.

    The `xView <http://xviewdataset.org/>`__
    dataset is a large scale dataset for object detection from overhead imagery. The
    dataset contains > 1 million objects across 60 classes in over 1400 km\ :sup:`2`\ of
    imagery. This dataset object uses the available training set (15.4 GB) and
    validation set (5.3 GB)

    Dataset format:

    * images are three-channel GeoTiFFs from WorldView-3
    * labels are in a single GeoJSON file

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1802.07856

    .. versionadded:: 0.4
    """

    metadata = {
        "train": {
            "filename": "train_images.tgz",
            "md5": "494854d0b1bf5cd2df9f7398cf6099c0",
            "directory": "train_images",
        },
        "labels": {
            "filename": "train_labels.tgz",
            "md5": "87bf732785da60c80053c1948ed8b0cb",
            "file": "xView_train.geojson",
        },
        "val": {
            "filename": "val_images.tgz",
            "md5": "1256d2ceaa517e411d85a90355ebbaf9",
            "directory": "val_images",
        },
    }

    classesIds = {
        # Small (0 - 100 px2)
        "Passenger Vehicle": 17,
        "Small car": 18,
        "Bus": 19,
        "Pickup Truck": 20,
        "Utility Truck": 21,
        "Truck": 23,
        "Cargo Truck": 24,
        "Truck Tractor": 26,
        "Trailer": 27,
        "Truck Tractor w/ Flatbed Trailer": 28,
        "Crane Truck": 32,
        "Motorboat": 41,
        "Dump Truck": 60,
        "Tractor": 62,
        "Front Loader/Bulldozer": 63,
        "Excavator": 64,
        "Cement Mixer": 65,
        "Ground Grader": 66,
        "Shipping Container": 91,
        # Medium (100 - 1000 px2)
        "Fixed-wing aircraft": 11,
        "Small aircraft": 12,
        "Helicopter": 15,
        "Truck Tractor w/ Box Trailer": 25,
        "Truck Tractor w/ Liquid Tank": 29,
        "Railway Vehicle": 33,
        "Passenger Car": 34,
        "Cargo/Container Car": 35,
        "Flat Car": 36,
        "Tank Car": 37,
        "Locomotive": 38,
        "Sailboat": 42,
        "Tugboat": 44,
        "Fishing Vessel": 47,
        "Yacht": 50,
        "Engineering Vehicle": 53,
        "Reach Stacker": 56,
        "Mobile Crane": 59,
        "Haul Truck": 61,
        "Hut/Tent": 71,
        "Shed": 72,
        "Building": 73,
        "Damaged/Demolished Building": 76,
        "Helipad": 84,
        "Storage Tank": 86,
        "Pylon": 93,
        "Tower": 94,
        # Large (> 1000 px2)
        "Passenger/Cargo Plane": 13,
        "Maritime Vessel": 40,
        "Barge": 45,
        "Ferry": 49,
        "Container Ship": 51,
        "Oil Tanker": 52,
        "Tower Crane": 54,
        "Container Crane": 55,
        "Straddle Carrier": 57,
        "Aircraft Hangar": 74,
        "Facility": 77,
        "Construction Site": 79,
        "Vehicle Lot": 83,
        "Shipping Container Lot": 89,
    }
    Idclasses = {value: key for key, value in classesIds.items()}
    classes = list(classesIds.keys())
    classesIds2 = dict(zip(classes, range(60)))
    Id2classes = {value: key for key, value in classesIds2.items()}

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new xView dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "val"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in {"train", "val"}
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.files = self._load_files(root, split)
        if self.split == "train":
            self.target = os.path.join(self.root, self.metadata["labels"]["file"])
            # self.target_file = fiona.open(self.target)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        file = self.files[index]
        image_id = os.path.basename(file["image"])
        image = self._load_image(file["image"])

        sample = {"image": image}
        if self.split == "train":
            boxes, labels = self._load_target(self.target, image_id)
            sample["boxes"] = boxes
            sample["labels"] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array = f.read()
        tensor = torch.from_numpy(array)
        return tensor

    def _load_target(self, path: str, imid: str) -> Tuple[Tensor, Tensor]:
        """Load the target bounding boxes for a single image.

        Args:
            path: path to the labels
            imid: image id

        Returns:
            the target boxes and labels
        """
        boxes = []
        labels = []
        # TODO: Loading labels seems slow
        with fiona.open(path) as src:
            filtered = filter(lambda f: f["properties"]["image_id"] == imid, src)
            for feature in filtered:
                classId = feature["properties"]["type_id"]
                class_name = self.Idclasses[classId]
                classId2 = self.classesIds2[class_name]
                box = feature["properties"]["bounds_imcoords"].split(",")
                box = torch.tensor([int(i) for i in box])
                boxes.append(box)
                labels.append(classId2)
        box_tensor: Tensor = torch.stack(boxes)
        label_tensor: Tensor = torch.tensor(labels)
        return box_tensor, label_tensor

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self, root: str, split: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, val]

        Returns:
            list of dicts containing paths for each pair of images and masks
        """
        directory = self.metadata[split]["directory"]
        image_root = os.path.join(root, directory)
        images = glob.glob(os.path.join(image_root, "*.tif"))
        files = [dict(image=image) for image in images]
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        exists = []
        split_info = self.metadata[self.split]
        exists.append(os.path.exists(os.path.join(self.root, split_info["directory"])))
        if self.split == "train":
            exists.append(
                os.path.exists(os.path.join(self.root, self.metadata["labels"]["file"]))
            )

        if all(exists):
            return

        # Check if .tar.gz files already exists (if so then extract)
        exists = []
        keys = ["train_images", "train_labels"] if self.split == "train" else ["val"]
        for k in keys:
            split_info = self.metadata[k]
            filepath = os.path.join(self.root, split_info["filename"])
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, split_info["md5"]):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root` directory, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        box_alpha: float = 0.7,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_titles: flag indicating whether to show titles above each panel
            box_alpha: alpha value of box

        Returns:
            a matplotlib Figure with the rendered sample

        """
        image = sample["image"].permute(1, 2, 0).numpy()
        boxes = sample["boxes"].cpu().numpy()
        labels = sample["labels"].cpu().numpy()

        N_GT = len(boxes)

        ncols = 1
        show_predictions = "prediction_labels" in sample

        if show_predictions:
            show_pred_boxes = False
            prediction_labels = sample["prediction_labels"].numpy()
            prediction_scores = sample["prediction_scores"].numpy()
            if "prediction_boxes" in sample:
                prediction_boxes = sample["prediction_boxes"].numpy()
                show_pred_boxes = True

            N_PRED = len(prediction_labels)
            ncols += 1

        # Display image
        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis("off")

        cm = plt.get_cmap("gist_rainbow")
        for i in range(N_GT):
            class_num = labels[i]
            color = cm(class_num / len(self.classes))

            # Add bounding boxes
            x1, y1, x2, y2 = boxes[i]
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=box_alpha,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            axs[0].add_patch(p)

            # Add labels
            label = self.Id2classes[class_num]
            caption = label
            axs[0].text(
                x1, y1 - 8, caption, color="white", size=9, backgroundcolor="none"
            )

            if show_titles:
                axs[0].set_title("Ground Truth")

        if show_predictions:
            axs[1].imshow(image)
            axs[1].axis("off")
            for i in range(N_PRED):
                score = prediction_scores[i]
                if score < 0.5:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / len(self.classes))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    p = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor="none",
                    )
                    axs[1].add_patch(p)

                    # Add labels
                    label = self.Id2classes[class_num]
                    caption = f"{label} {score:.3f}"
                    axs[1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color="white",
                        size=9,
                        backgroundcolor="none",
                    )

            if show_titles:
                axs[1].set_title("Prediction")

        plt.tight_layout()

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class XView2(NonGeoDataset):
    """xView2 dataset.

    The `xView2 <https://xview2.org/>`__
    dataset is a dataset for building disaster change detection. This dataset object
    uses the "Challenge training set (~7.8 GB)" and "Challenge test set (~2.6 GB)" data
    from the xView2 website as the train and test splits. Note, the xView2 website
    contains other data under the xView2 umbrella that are _not_ included here. E.g.
    the "Tier3 training data", the "Challenge holdout set", and the "full data".

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where the pixel values represent the class

    Dataset classes:

    0. background
    1. no damage
    2. minor damage
    3. major damage
    4. destroyed

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1911.09296

    .. versionadded:: 0.2
    """

    metadata = {
        "train": {
            "filename": "train_images_labels_targets.tar.gz",
            "md5": "a20ebbfb7eb3452785b63ad02ffd1e16",
            "directory": "train",
        },
        "test": {
            "filename": "test_images_labels_targets.tar.gz",
            "md5": "1b39c47e05d1319c17cc8763cee6fe0c",
            "directory": "test",
        },
    }
    classes = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
    colormap = ["green", "blue", "orange", "red"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new xView2 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.metadata
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files(root, split)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files["image1"])
        image2 = self._load_image(files["image2"])
        mask1 = self._load_target(files["mask1"])
        mask2 = self._load_target(files["mask2"])

        image = torch.stack(tensors=[image1, image2], dim=0)
        mask = torch.stack(tensors=[mask1, mask2], dim=0)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self, root: str, split: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of images and masks
        """
        files = []
        directory = self.metadata[split]["directory"]
        image_root = os.path.join(root, directory, "images")
        mask_root = os.path.join(root, directory, "targets")
        images = glob.glob(os.path.join(image_root, "*.png"))
        basenames = [os.path.basename(f) for f in images]
        basenames = ["_".join(f.split("_")[:-2]) for f in basenames]
        for name in set(basenames):
            image1 = os.path.join(image_root, f"{name}_pre_disaster.png")
            image2 = os.path.join(image_root, f"{name}_post_disaster.png")
            mask1 = os.path.join(mask_root, f"{name}_pre_disaster_target.png")
            mask2 = os.path.join(mask_root, f"{name}_post_disaster_target.png")
            files.append(dict(image1=image1, image2=image2, mask1=mask1, mask2=mask2))
        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        exists = []
        for split_info in self.metadata.values():
            for directory in ["images", "targets"]:
                exists.append(
                    os.path.exists(
                        os.path.join(self.root, split_info["directory"], directory)
                    )
                )

        if all(exists):
            return

        # Check if .tar.gz files already exists (if so then extract)
        exists = []
        for split_info in self.metadata.values():
            filepath = os.path.join(self.root, split_info["filename"])
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, split_info["md5"]):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root` directory, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 0.5,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render predictions on top of the imagery

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2
        image1 = draw_semantic_segmentation_masks(
            sample["image"][0], sample["mask"][0], alpha=alpha, colors=self.colormap
        )
        image2 = draw_semantic_segmentation_masks(
            sample["image"][1], sample["mask"][1], alpha=alpha, colors=self.colormap
        )
        if "prediction" in sample:  # NOTE: this assumes predictions are made for post
            ncols += 1
            image3 = draw_semantic_segmentation_masks(
                sample["image"][1],
                sample["prediction"],
                alpha=alpha,
                colors=self.colormap,
            )

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")
        if ncols > 2:
            axs[2].imshow(image3)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Pre disaster")
            axs[1].set_title("Post disaster")
            if ncols > 2:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
