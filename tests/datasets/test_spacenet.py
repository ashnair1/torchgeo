# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SpaceNet, SpaceNet1, SpaceNet6
from torchgeo.datasets.spacenet.spacenet9 import SpaceNet9
from torchgeo.datasets.utils import Executable


class TestSpaceNet:
    @pytest.fixture(params=[SpaceNet1, SpaceNet6])
    def dataset(
        self,
        request: SubRequest,
        aws: Executable,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> SpaceNet:
        dataset_class: type[SpaceNet] = request.param
        url = os.path.join(
            'tests',
            'data',
            'spacenet',
            dataset_class.__name__.lower(),
            '{dataset_id}',
            'train',
            '{tarball}',
        )
        monkeypatch.setattr(dataset_class, 'url', url)
        transforms = nn.Identity()
        return dataset_class(tmp_path, transforms=transforms, download=True)

    @pytest.mark.parametrize('index', [0, 1])
    def test_getitem(self, dataset: SpaceNet, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SpaceNet) -> None:
        assert len(dataset) == 4

    def test_already_extracted(self, dataset: SpaceNet) -> None:
        dataset.__class__(root=dataset.root)

    def test_already_downloaded(self, dataset: SpaceNet) -> None:
        if dataset.dataset_id == 'SN1_buildings':
            base_dir = os.path.join(dataset.root, dataset.dataset_id, dataset.split)
        elif dataset.dataset_id == 'SN6_buildings':
            base_dir = os.path.join(
                dataset.root,
                dataset.dataset_id,
                dataset.split,
                dataset.split,
                'AOI_11_Rotterdam',
            )
        for product in dataset.valid_images['train'] + list(dataset.valid_masks):
            dir = os.path.join(base_dir, product)
            shutil.rmtree(dir)
        dataset.__class__(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path, dataset: SpaceNet) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            dataset.__class__(root=os.path.join(tmp_path, 'dummy'))

    def test_plot(self, dataset: SpaceNet) -> None:
        x = dataset[0]
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask']
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_image_id(self, monkeypatch: MonkeyPatch, dataset: SpaceNet) -> None:
        file_regex = r'global_monthly_(\d+.*\d+)'
        monkeypatch.setattr(dataset, 'file_regex', file_regex)
        dataset._image_id('global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160.tif')

    def test_list_files(self, monkeypatch: MonkeyPatch, dataset: SpaceNet) -> None:
        directory_glob = os.path.join('**', 'AOI_{aoi}_*', '{product}')
        monkeypatch.setattr(dataset, 'directory_glob', directory_glob)
        dataset._list_files(aoi=1)


class TestSpaceNet9:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self, request: SubRequest, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet9:
        split = request.param
        url = os.path.join(
            'tests', 'data', 'spacenet', 'spacenet9', 'spacenet9', '{tarball}'
        )
        monkeypatch.setattr(SpaceNet9, 'url', url)
        md5s = {
            'train': '0c4474cc4478155ea7c84579c0c11cf3',
            'test': '66e1e5d39d26ebce0a63112caf2bafde',
        }
        monkeypatch.setattr(SpaceNet9, 'md5s', md5s)
        return SpaceNet9(root=tmp_path, split=split, download=True, checksum=True)

    def test_getitem(self, dataset: SpaceNet9) -> None:
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert 'optical' in sample
        assert 'sar' in sample
        assert isinstance(sample['optical'], torch.Tensor)
        assert isinstance(sample['sar'], torch.Tensor)
        # Check tiepoints for training split
        if dataset.split == 'train':
            assert 'tiepoints' in sample
            assert isinstance(sample['tiepoints'], torch.Tensor)

    def test_len(self, dataset: SpaceNet9) -> None:
        # Train has 3 samples (02_01, 02_02, 03_01), test has 2 (02, 03)
        expected_len = 3 if dataset.split == 'train' else 2
        assert len(dataset) == expected_len
        assert len(dataset.optical_images) == expected_len
        assert len(dataset.sar_images) == expected_len

    def test_tiepoints(self, dataset: SpaceNet9) -> None:
        # Tiepoints only exist for train split (3 samples)
        if dataset.split == 'train':
            assert len(dataset.tiepoints) == 3
        else:
            assert len(dataset.tiepoints) == 0

    def test_already_extracted(self, dataset: SpaceNet9) -> None:
        SpaceNet9(root=dataset.root, split=dataset.split)

    def test_already_downloaded(self, dataset: SpaceNet9) -> None:
        # Remove extracted files but keep tarball
        foldername = 'publictest' if dataset.split == 'test' else dataset.split
        extracted_path = os.path.join(dataset.root, 'spacenet9', foldername)
        if os.path.exists(extracted_path):
            shutil.rmtree(extracted_path)
        SpaceNet9(root=dataset.root, split=dataset.split)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SpaceNet9(root=os.path.join(tmp_path, 'dummy'))

    def test_plot(self, dataset: SpaceNet9) -> None:
        sample = dataset[0]
        dataset.plot(sample, show_titles=False)
        plt.close()
        dataset.plot(sample, suptitle='Test')
        plt.close()
        # Test with tiepoints disabled
        dataset.plot(sample, show_tiepoints=False)
        plt.close()
