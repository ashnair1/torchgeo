#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from collections import OrderedDict

import fiona
import numpy as np
import rasterio
from rasterio import windows
from rasterio.crs import CRS
from rasterio.transform import Affine
from torchvision.datasets.utils import calculate_md5

from torchgeo.datasets import (
    SpaceNet1,
    SpaceNet2,
    SpaceNet3,
    SpaceNet4,
    SpaceNet5,
    SpaceNet7,
)

transform = Affine(0.3, 0.0, 616500.0, 0.0, -0.3, 3345000.0)
crs = CRS.from_epsg(4326)

img_count = {
    "MS.tif": 8,
    "PAN.tif": 1,
    "PS-MS.tif": 8,
    "PS-RGB.tif": 3,
    "PS-RGBNIR.tif": 4,
    "RGB.tif": 3,
    "mosaic.tif": 3,
    "8Band.tif": 8,
}

datasets = [SpaceNet1, SpaceNet2, SpaceNet3, SpaceNet4, SpaceNet5, SpaceNet7]


def to_index(wind_):
    """
    Generates a list of index (row,col):

    [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
    """
    return [
        [wind_.row_off, wind_.col_off],
        [wind_.row_off, wind_.col_off + wind_.width],
        [wind_.row_off + wind_.height, wind_.col_off + wind_.width],
        [wind_.row_off + wind_.height, wind_.col_off],
        [wind_.row_off, wind_.col_off],
    ]


def create_test_image(img_dir, imgs):
    for img in imgs:
        imgpath = os.path.join(img_dir, img)
        Z = np.arange(4, dtype="uint16").reshape(2, 2)
        count = img_count[img]
        with rasterio.open(
            imgpath,
            "w",
            driver="GTiff",
            height=Z.shape[0],
            width=Z.shape[1],
            count=count,
            dtype=Z.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for i in range(1, dst.count + 1):
                dst.write(Z, i)

    tim = rasterio.open(imgpath)
    slice = windows.Window(1, 1, 1, 1)
    slice_index = to_index(slice)
    return [list(tim.transform * p) for p in slice_index]


def create_test_label(lbldir, lblname, coords, det_type):

    if det_type == "roads":
        meta = {
            "driver": "GeoJSON",
            "schema": {"properties": OrderedDict(), "geometry": "Polygon"},
            "crs": {"init": "epsg:4326"},
        }
        rec = {
            "type": "Feature",
            "id": "0",
            "properties": OrderedDict(),
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }
    else:
        road_properties = OrderedDict(
            [
                ("heading", "str"),
                ("lane_number", "str"),
                ("one_way_ty", "str"),
                ("paved", "str"),
                ("road_id", "int"),
                ("road_type", "str"),
                ("origarea", "int"),
                ("origlen", "float"),
                ("partialDec", "int"),
                ("truncated", "int"),
                ("bridge_type", "str"),
                ("inferred_speed_mph", "float"),
                ("inferred_speed_mps", "float"),
            ]
        )
        meta = {
            "driver": "GeoJSON",
            "schema": {"properties": road_properties, "geometry": "LineString"},
            "crs": {"init": "epsg:4326"},
        }
        dummy_vals = {"str": "a", "float": 45.0, "int": 0}
        ROAD_DICT = [(k, dummy_vals[v]) for k, v in road_properties.items()]
        rec = {
            "type": "Feature",
            "id": "0",
            "properties": OrderedDict(ROAD_DICT),
            "geometry": {"type": "LineString", "coordinates": [coords[0], coords[2]]},
        }

    out_file = os.path.join(lbldir, lblname)
    dst = fiona.open(out_file, "w", **meta)
    dst.write(rec)


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

    num_samples = 2

    for dataset in datasets:

        collections = list(dataset.collection_md5_dict.keys())
        for collection in collections:
            if dataset.dataset_id == "spacenet4":
                num_samples = 4
            elif collection == "sn5_AOI_7_Moscow":
                num_samples = 2
            elif collection == "sn5_AOI_8_Mumbai":
                num_samples = 3
            elif collection == "sn7_test_source":
                num_samples = 1
            else:
                num_samples = 2
            for sample in range(num_samples):
                out_dir = os.path.join(ROOT_DIR, collection)

                # Create img dir
                imgdir = os.path.join(out_dir, f"{collection}_img{sample + 1}")
                os.makedirs(imgdir, exist_ok=True)
                bounds = create_test_image(imgdir, list(dataset.imagery.values()))

                # Create lbl dir
                lbldir = os.path.join(out_dir, f"{collection}_img{sample + 1}-labels")
                os.makedirs(lbldir, exist_ok=True)
                det_type = "roads" if dataset in [SpaceNet3, SpaceNet5] else "buildings"
                create_test_label(lbldir, dataset.label_glob, bounds, det_type)

            # Create archive
            archive_path = os.path.join(ROOT_DIR, collection)
            shutil.make_archive(
                archive_path, "gztar", root_dir=ROOT_DIR, base_dir=collection
            )
            shutil.rmtree(out_dir)
            print(f"{collection}: " + calculate_md5(f"{archive_path}.tar.gz"))
