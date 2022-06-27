# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datamodules import IDTReeSDataModule
from torchgeo.datasets import IDTReeS


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestIDTReeSDataModule:
    @pytest.fixture(
        params=zip([0.3, 0.3, 0.0], [0.3, 0.0, 0.0], ["test", "test", "test"])
    )
    def datamodule(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> IDTReeSDataModule:
        val_split_pct, test_split_pct, predict_on = request.param
        monkeypatch.setattr(torchgeo.datasets.idtrees, "download_url", download_url)
        data_dir = os.path.join("tests", "data", "idtrees")
        metadata = {
            "train": {
                "url": os.path.join(data_dir, "IDTREES_competition_train_v2.zip"),
                "md5": "5ddfa76240b4bb6b4a7861d1d31c299c",
                "filename": "IDTREES_competition_train_v2.zip",
            },
            "test": {
                "url": os.path.join(data_dir, "IDTREES_competition_test_v2.zip"),
                "md5": "b108931c84a70f2a38a8234290131c9b",
                "filename": "IDTREES_competition_test_v2.zip",
            },
        }
        monkeypatch.setattr(IDTReeS, "metadata", metadata)
        root = str(tmp_path)
        batch_size = 1
        num_workers = 0
        dm = IDTReeSDataModule(
            root,
            batch_size,
            num_workers,
            val_split_pct,
            test_split_pct,
            predict_on=predict_on,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: IDTReeSDataModule) -> None:
        sample = next(iter(datamodule.train_dataloader()))
        assert sample["image"].shape[0] == len(sample["boxes"]) == 1
        assert len(sample["boxes"]) == len(sample["labels"]) == 1
        assert sample["image"].shape[1] == 3

    def test_val_dataloader(self, datamodule: IDTReeSDataModule) -> None:
        sample = next(iter(datamodule.val_dataloader()))
        if datamodule.val_split_pct > 0.0:
            assert sample["image"].shape[1] == 3

    def test_test_dataloader(self, datamodule: IDTReeSDataModule) -> None:
        sample = next(iter(datamodule.test_dataloader()))
        if datamodule.test_split_pct > 0.0:
            assert sample["image"].shape[1] == 3

    def test_predict_dataloader(self, datamodule: IDTReeSDataModule) -> None:
        sample = next(iter(datamodule.predict_dataloader()))
        assert sample["image"].shape[1] == 3
