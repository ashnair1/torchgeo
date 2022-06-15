# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import (
    SpaceNet1,
    SpaceNet2,
    SpaceNet3,
    SpaceNet4,
    SpaceNet5,
    SpaceNet7,
)

TEST_DATA_DIR = "tests/data/spacenet"


class Collection:
    def __init__(self, collection_id: str) -> None:
        self.collection_id = collection_id

    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(TEST_DATA_DIR, "*.tar.gz")
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch_collection(collection_id: str, **kwargs: str) -> Collection:
    return Collection(collection_id)


class TestSpaceNet1:
    @pytest.fixture(params=["rgb", "8band"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet1:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {"sn1_AOI_1_RIO": "2c29b8b72a2474abcccf04595d043485"}

        # Refer https://github.com/python/mypy/issues/1032
        monkeypatch.setattr(SpaceNet1, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return SpaceNet1(
            root, image=request.param, transforms=transforms, download=True, api_key=""
        )

    def test_getitem(self, dataset: SpaceNet1) -> None:
        x = dataset[1]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "rgb":
            assert x["image"].shape[0] == 3
        else:
            assert x["image"].shape[0] == 8

    def test_len(self, dataset: SpaceNet1) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: SpaceNet1) -> None:
        SpaceNet1(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet1(str(tmp_path))

    def test_plot(self, dataset: SpaceNet1) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()


class TestSpaceNet2:
    @pytest.fixture(params=["PAN", "MS", "PS-MS", "PS-RGB"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet2:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn2_AOI_2_Vegas": "d078c1516c19468dd374447a2c5629a2",
            "sn2_AOI_3_Paris": "e18939c16fa4d3157b24b6156251db00",
            "sn2_AOI_4_Shanghai": "09d31943b16f7c56e206e3de3f646939",
            "sn2_AOI_5_Khartoum": "5b0a8fb0c43e498fd4101287a1faaca0",
        }

        monkeypatch.setattr(SpaceNet2, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return SpaceNet2(
            root,
            image=request.param,
            collections=["sn2_AOI_2_Vegas", "sn2_AOI_5_Khartoum"],
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "PS-RGB":
            assert x["image"].shape[0] == 3
        elif dataset.image in ["MS", "PS-MS"]:
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    def test_len(self, dataset: SpaceNet2) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: SpaceNet2) -> None:
        SpaceNet2(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet2(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet2) -> None:
        dataset.collection_md5_dict["sn2_AOI_2_Vegas"] = "randommd5hash123"
        with pytest.raises(RuntimeError, match="Collection sn2_AOI_2_Vegas corrupted"):
            SpaceNet2(root=dataset.root, download=True, checksum=True)

    def test_plot(self, dataset: SpaceNet2) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()


class TestSpaceNet3:
    @pytest.fixture(params=zip(["PAN", "MS"], [False, True]))
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet3:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn3_AOI_3_Paris": "ccae432574adc5b43b8c4743d0f7b9ab",
            "sn3_AOI_5_Khartoum": "b1b94da3aa2fbf3795b6d63af58b965e",
        }

        monkeypatch.setattr(SpaceNet3, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return SpaceNet3(
            root,
            image=request.param[0],
            speed_mask=request.param[1],
            collections=["sn3_AOI_3_Paris", "sn3_AOI_5_Khartoum"],
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet3) -> None:
        # Iterate over all elements to maximize coverage
        samples = [dataset[i] for i in range(len(dataset))]
        x = samples[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "MS":
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    def test_len(self, dataset: SpaceNet3) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: SpaceNet3) -> None:
        SpaceNet3(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet3(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet3) -> None:
        dataset.collection_md5_dict["sn3_AOI_5_Khartoum"] = "randommd5hash123"
        with pytest.raises(
            RuntimeError, match="Collection sn3_AOI_5_Khartoum corrupted"
        ):
            SpaceNet3(root=dataset.root, download=True, checksum=True)

    def test_plot(self, dataset: SpaceNet3) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        dataset.plot({"image": x["image"]})
        plt.close()


class TestSpaceNet4:
    @pytest.fixture(params=["PAN", "MS", "PS-RGBNIR"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet4:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {"sn4_AOI_6_Atlanta": "275164ae8b3277fff565f81309a9c6b8"}

        test_angles = ["nadir", "off-nadir", "very-off-nadir"]

        monkeypatch.setattr(SpaceNet4, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return SpaceNet4(
            root,
            image=request.param,
            angles=test_angles,
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet4) -> None:
        # Get image-label pair with empty label to
        # ensure coverage
        x = dataset[2]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "PS-RGBNIR":
            assert x["image"].shape[0] == 4
        elif dataset.image == "MS":
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    def test_len(self, dataset: SpaceNet4) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: SpaceNet4) -> None:
        SpaceNet4(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet4(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet4) -> None:
        dataset.collection_md5_dict["sn4_AOI_6_Atlanta"] = "randommd5hash123"
        with pytest.raises(
            RuntimeError, match="Collection sn4_AOI_6_Atlanta corrupted"
        ):
            SpaceNet4(root=dataset.root, download=True, checksum=True)

    def test_plot(self, dataset: SpaceNet4) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()


class TestSpaceNet5:
    @pytest.fixture(params=zip(["PAN", "MS"], [False, True]))
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet5:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn5_AOI_7_Moscow": "78ba2cf048de22f548405c4f3a155f1b",
            "sn5_AOI_8_Mumbai": "90d8b319db20aba36b1d3e5d942c59ff",
        }

        monkeypatch.setattr(SpaceNet5, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return SpaceNet5(
            root,
            image=request.param[0],
            speed_mask=request.param[1],
            collections=["sn5_AOI_7_Moscow", "sn5_AOI_8_Mumbai"],
            transforms=transforms,
            download=True,
            api_key="",
        )

    def test_getitem(self, dataset: SpaceNet5) -> None:
        # Iterate over all elements to maximize coverage
        samples = [dataset[i] for i in range(len(dataset))]
        x = samples[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        if dataset.image == "MS":
            assert x["image"].shape[0] == 8
        else:
            assert x["image"].shape[0] == 1

    def test_len(self, dataset: SpaceNet5) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: SpaceNet5) -> None:
        SpaceNet5(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet5(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet5) -> None:
        dataset.collection_md5_dict["sn5_AOI_8_Mumbai"] = "randommd5hash123"
        with pytest.raises(RuntimeError, match="Collection sn5_AOI_8_Mumbai corrupted"):
            SpaceNet5(root=dataset.root, download=True, checksum=True)

    def test_plot(self, dataset: SpaceNet5) -> None:
        x = dataset[0].copy()
        x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        dataset.plot({"image": x["image"]})
        plt.close()


class TestSpaceNet7:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet7:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
        test_md5 = {
            "sn7_train_source": "1d9b371c27b974a383c5ab994ac71cc1",
            "sn7_train_labels": "82384eaeeed16c90436bdecde924dbfa",
            "sn7_test_source": "f404037990024006a79de4fbf499aefe",
        }

        monkeypatch.setattr(SpaceNet7, "collection_md5_dict", test_md5)
        root = str(tmp_path)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return SpaceNet7(
            root, split=request.param, transforms=transforms, download=True, api_key=""
        )

    def test_getitem(self, dataset: SpaceNet7) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        if dataset.split == "train":
            assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: SpaceNet7) -> None:
        if dataset.split == "train":
            assert len(dataset) == 2
        else:
            assert len(dataset) == 1

    def test_already_downloaded(self, dataset: SpaceNet4) -> None:
        SpaceNet7(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            SpaceNet7(str(tmp_path))

    def test_collection_checksum(self, dataset: SpaceNet4) -> None:
        dataset.collection_md5_dict["sn7_train_source"] = "randommd5hash123"
        with pytest.raises(RuntimeError, match="Collection sn7_train_source corrupted"):
            SpaceNet7(root=dataset.root, download=True, checksum=True)

    def test_plot(self, dataset: SpaceNet7) -> None:
        x = dataset[0].copy()
        if dataset.split == "train":
            x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
