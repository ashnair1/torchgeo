# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import SKIPPD, DatasetNotFoundError

pytest.importorskip('h5py', minversion='3.8')


class TestSKIPPD:
    @pytest.fixture(params=product(['nowcast', 'forecast'], ['trainval', 'test']))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SKIPPD:
        task, split = request.param
        md5 = {
            'nowcast': '6f5e54906927278b189f9281a2f54f39',
            'forecast': 'f3b5d7d5c28ba238144fa1e726c46969',
        }
        monkeypatch.setattr(SKIPPD, 'md5', md5)
        url = os.path.join('tests', 'data', 'skippd', '{}')
        monkeypatch.setattr(SKIPPD, 'url', url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        root = tmp_path
        transforms = nn.Identity()
        return SKIPPD(
            root=root,
            task=task,
            split=split,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_already_extracted(self, dataset: SKIPPD) -> None:
        SKIPPD(root=dataset.root, download=True)

    @pytest.mark.parametrize('task', ['nowcast', 'forecast'])
    def test_already_downloaded(self, tmp_path: Path, task: str) -> None:
        pathname = os.path.join(
            'tests', 'data', 'skippd', f'2017_2019_images_pv_processed_{task}.zip'
        )
        root = tmp_path
        shutil.copy(pathname, root)
        SKIPPD(root=root, task=task)

    @pytest.mark.parametrize('index', [0, 1, 2])
    def test_getitem(self, dataset: SKIPPD, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert isinstance(x['date'], str)
        if dataset.task == 'nowcast':
            assert x['image'].shape == (3, 64, 64)
        else:
            assert x['image'].shape == (48, 64, 64)

    def test_len(self, dataset: SKIPPD) -> None:
        assert len(dataset) == 3

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SKIPPD(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SKIPPD(tmp_path)

    def test_plot(self, dataset: SKIPPD) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        if dataset.task == 'nowcast':
            sample['prediction'] = sample['label']
        else:
            sample['prediction'] = sample['label'][-1]
        dataset.plot(sample)
        plt.close()
