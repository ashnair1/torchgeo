# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import SpaceNet2DataModule

TEST_DATA_DIR = "tests/data/spacenet"
PREDICT_DIR = os.path.join(TEST_DATA_DIR, "predict_imgs")
SIZE_MAP = {"MS": (160, 160), "PS-MS": (640, 640), "PS-RGB": (640, 640)}


class TestSpaceNet2DataModule:
    @pytest.fixture(
        params=zip(["MS", "PS-MS", "PS-RGB"], [0.25, 0.25, 0.0], [0.25, 0.0, 0.0])
    )
    def datamodule(self, request: SubRequest) -> SpaceNet2DataModule:
        image, val_split_pct, test_split_pct = request.param
        collections = ["sn2_AOI_3_Paris", "sn2_AOI_5_Khartoum"]
        root = TEST_DATA_DIR
        batch_size = 1
        num_workers = 0
        if image in ["MS", "PS-MS"]:
            bands = [4, 2, 1]
        elif image == "PS-RGB":
            bands = [1, 2, 3]
        seed = 42
        predict_on = PREDICT_DIR
        dm = SpaceNet2DataModule(
            root,
            image,
            collections,
            bands,
            seed,
            batch_size,
            num_workers,
            val_split_pct,
            test_split_pct,
            predict_on=predict_on,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: SpaceNet2DataModule) -> None:
        sample = next(iter(datamodule.train_dataloader()))
        assert (
            sample["image"].shape[-2:]
            == sample["mask"].shape[-2:]
            == SIZE_MAP[datamodule.image]
        )
        assert sample["image"].shape[0] == sample["mask"].shape[0] == 1
        if datamodule.image == "PAN":
            assert sample["image"].shape[1] == 1
        else:
            assert sample["image"].shape[1] == 3

    def test_val_dataloader(self, datamodule: SpaceNet2DataModule) -> None:
        sample = next(iter(datamodule.val_dataloader()))
        if datamodule.val_split_pct > 0.0:
            assert (
                sample["image"].shape[-2:]
                == sample["mask"].shape[-2:]
                == SIZE_MAP[datamodule.image]
            )
            assert sample["image"].shape[0] == sample["mask"].shape[0] == 1

    def test_test_dataloader(self, datamodule: SpaceNet2DataModule) -> None:
        sample = next(iter(datamodule.test_dataloader()))
        if datamodule.test_split_pct > 0.0:
            assert (
                sample["image"].shape[-2:]
                == sample["mask"].shape[-2:]
                == SIZE_MAP[datamodule.image]
            )
            assert sample["image"].shape[0] == sample["mask"].shape[0] == 1

    def test_predict_dataloader(self, datamodule: SpaceNet2DataModule) -> None:
        sample = next(iter(datamodule.predict_dataloader()))
        assert sample["image"].shape[-2:] == SIZE_MAP[datamodule.image]
        assert sample["image"].shape[1] == 3
