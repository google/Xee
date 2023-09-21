# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Type definitions for Earth Engine concepts (and others)."""
from typing import Union, TypedDict

TileIndex = tuple[int, int, int]
# x_min, y_min, x_max, y_max
Bounds = tuple[float, float, float, float]
# x_start, y_start, x_stop, y_stop
BBox = tuple[int, int, int, int]
# index_start, index_stop, x_start, y_start, x_stop, y_stop
BBox3d = tuple[int, int, int, int, int, int]
Grid = dict[str, Union[dict[str, Union[float, str]], str, float]]


class DataType(TypedDict):
  max: int
  min: int
  precision: str
  type: str


class BandInfo(TypedDict):
  crs: str
  crs_transform: list[int]  # len: 6, gdal order
  data_type: DataType
  dimensions: list[int]  # len: 2
  id: str


class ImageInfo(TypedDict):
  type: str
  bands: list[BandInfo]
  id: str
  version: int
