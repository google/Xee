# Copyright 2025 Google LLC
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
"""Helper functions for grid parameters."""
import math

import affine
import ee
from pyproj import Transformer
import shapely
from shapely.ops import transform
from typing import TypedDict, Tuple, Union


TransformType = Tuple[float, float, float, float, float, float]
ShapeType = Tuple[int, int]
ScalingType = Union[float, Tuple[float, float]]


class PixelGridParams(TypedDict):
    crs: str
    crs_transform: TransformType
    shape_d2: ShapeType


def set_scale(
    crs_transform: TransformType,
    scaling: ScalingType,
  ) -> list:
  """Update the CRS transform's scale parameters."""
  print(f'{type(scaling)=}')
  if isinstance(scaling, tuple) and len(scaling) == 2:
    x_scale, y_scale = scaling
    crs_transform[0] = x_scale
    crs_transform[4] = y_scale
  else:
    raise TypeError(f'Expected a tuple of length 2 for scaling, got {scaling}')
  affine_transform = affine.Affine(*crs_transform)
  return list(affine_transform)[:6]


def fit_geometry(
  geometry: shapely.geometry,
  *,
  geometry_crs: str = 'EPSG:4326',
  buffer: float = 0,
  grid_crs: str = 'EPSG:4326',
  grid_scale: float = None,
  grid_scale_digits: int = None,
  grid_shape: ShapeType = None,
) -> PixelGridParams: 
  """Return grid parameters that fit the geometry."""
  
  # Check that exactly one of the arguments is specified
  if (grid_scale is None) == (grid_shape is None):
    raise ValueError("Exactly one of 'grid_scale' or 'grid_shape' must be specified.")

  # Reproject geometry to the grid CRS. If the grids are the same this
  # is a no-op.
  transformer = Transformer.from_crs(
    crs_from=geometry_crs,
    crs_to=grid_crs,
    always_xy=True
  )
  reprojected_geometry = transform(transformer.transform, geometry)
  if buffer and buffer > 0:
    buffered_geom = shapely.buffer(reprojected_geometry, buffer)
  else:
    buffered_geom = reprojected_geometry
  x_min, y_min, x_max, y_max = buffered_geom.bounds

  if grid_scale:
    # Given scale & geometry, determine the translation & shape parameters. 
    x_scale = y_scale = grid_scale
    
    x_shape = math.ceil(
      (x_max / x_scale - math.floor(x_min / x_scale)) 
    )
    y_shape = math.ceil(
      (-y_min / y_scale + math.ceil(y_max / y_scale)) 
    )
  
  if grid_shape:
    # Given shape & geometry, determine the translation & scale parameters. 
    x_shape, y_shape = grid_shape
    
    x_scale = (x_max - x_min) / x_shape
    y_scale = (y_max - y_min) / y_shape

    if grid_scale_digits:
      x_scale = round(x_scale, grid_scale_digits)
      y_scale = round(y_scale, grid_scale_digits)
  
  grid_x_min = math.floor(x_min / x_scale) * x_scale
  grid_y_max = math.ceil(y_max / y_scale) * y_scale
  
  affine_transform = (
    affine.Affine.translation(grid_x_min, grid_y_max)
    * affine.Affine.scale(x_scale, -y_scale)
  )

  crs_transform = list(affine_transform)[:6]

  return dict(
    crs=grid_crs,
    crs_transform=crs_transform,
    shape_2d=(x_shape, y_shape)
  )


def extract_grid_params(
    ee_obj: Union[ee.Image, ee.ImageCollection]
  ) -> PixelGridParams:
  # Extract the pixel grid parameters from an ee.Image or ee.ImageCollection object

  if isinstance(ee_obj, ee.Image):
    img_obj = ee_obj
  elif isinstance(ee_obj, ee.ImageCollection):
    img_obj = ee_obj.first()
  else:
    raise TypeError(f'Expected ee.Image or ee.ImageCollection, got {type(ee_obj)}')
  
  first_band_info = img_obj.select(0).getInfo()['bands'][0]

  return dict(
    crs=first_band_info['crs'],
    crs_transform=first_band_info['crs_transform'],
    shape_2d=tuple(first_band_info['dimensions'])
  )
