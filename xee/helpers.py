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

import ee
from pyproj import Transformer
from rasterio.transform import Affine
import shapely
from shapely.ops import transform
from shapely.geometry import box
from typing import Literal

def set_scale(crs_transform, scaling):
  """Update the CRS transform's scale parameters."""
  match scaling:
    case int(xy_scale) | float(xy_scale):
      crs_transform[0] = xy_scale
      crs_transform[4] = xy_scale
    case (int(x_scale) | float(x_scale), int(y_scale) | float(y_scale)):
      crs_transform[0] = x_scale
      crs_transform[4] = y_scale
    case _:
      raise TypeError
  affine_transform = Affine(*crs_transform)
  return list(affine_transform)[:6]


def fit_geometry(
  geometry,
  *,
  geometry_crs='EPSG:4326',
  buffer=0,
  grid_crs='EPSG:4326',
  grid_scale=None,
  grid_scale_digits=None,
  grid_shape=None,
):
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
  
  affine_transform = Affine.translation(grid_x_min, grid_y_max) * Affine.scale(x_scale, -y_scale)
  crs_transform = list(affine_transform)[:6]

  return dict(
    crs=grid_crs,
    crs_transform=crs_transform,
    shape_2d=(x_shape, y_shape)
  )



def update_grid_translation(
    crs,
    crs_transform,
    shape,
    geometry
    ):
  """Update the grid's translateX and translateY parameters to center on the geometry."""

  return crs, crs_transform, shape



def update_shape(
    crs,
    crs_transform,
    shape,
    geometry
  ):
  # Update the shape to cover the geometry.
  return crs, crs_transform, shape


def extract_projection(
    ee_obj
  ):
  # Estimate the CRS and transform from an ee.Image or ee.ImageCollection object
  
  # proj_info = ee_obj.projection().getInfo()

  # return dict(
  #   crs=proj_info['crs'],
  #   crs_transform=proj_info['transform']
  # )
  match ee_obj:
    case ee.Image():
      print('Its an image')
      img_obj = ee_obj
    case ee.ImageCollection():
      print('Its an image collection')
      img_obj = ee_obj.first()
    case _:
      raise TypeError
  
  first_band_info = img_obj.select(0).getInfo()['bands'][0]

  return dict(
    crs=first_band_info['crs'],
    crs_transform=first_band_info['crs_transform'],
    shape_2d=first_band_info['dimensions']
  )
