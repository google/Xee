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
"""Helper functions for constructing pixel grid parameters for Xee.

These helpers produce the three required keyword arguments passed to
``xarray.open_dataset(..., engine='ee', **grid_params)``:

* ``crs`` – The target Coordinate Reference System.
* ``crs_transform`` – A 6-tuple affine transform (origin + scale) in CRS units.
* ``shape_2d`` – The (width, height) pixel shape of the output grid.

Two primary workflows:

1. :func:`extract_grid_params` – Match the *native* grid of an Earth Engine
  Image or ImageCollection.
2. :func:`fit_geometry` – Derive a grid that fits a user geometry using either
  an explicit pixel scale (``grid_scale``) or an explicit pixel shape
  (``grid_shape``).

All scale values must be expressed in the units of ``grid_crs``. For
geographic CRSs (e.g. ``EPSG:4326``) this is degrees. For projected CRSs (e.g.
UTM) this is meters.
"""
import math

import affine
import ee
from pyproj import Transformer
import shapely
from shapely.ops import transform
from typing import TypedDict, Union


TransformType = tuple[float, float, float, float, float, float]
ShapeType = tuple[int, int]
ScalingType = tuple[float, float]


class PixelGridParams(TypedDict):
  """TypedDict describing pixel grid parameters.

  - ``crs``: EPSG code or WKT for output grid CRS.
  - ``crs_transform``: 6-tuple affine transform ``(a, b, c, d, e, f)``:
      a = pixel width (x scale)
      b = row rotation (usually 0)
      c = x origin (upper-left x)
      d = column rotation (usually 0)
      e = pixel height (y scale, negative for north-up)
      f = y origin (upper-left y)
  - ``shape_2d``: ``(width, height)`` pixel counts.
  """

  crs: str
  crs_transform: TransformType
  shape_2d: ShapeType


def set_scale(
    crs_transform: TransformType,
    scaling: ScalingType,
) -> list:
  """Return a new CRS transform with updated scale components.

  Useful for adjusting an existing transform's pixel size while retaining its
  origin. A negative y scale preserves north-up orientation.

  Args:
    crs_transform: Existing 6-value transform tuple.
    scaling: ``(x_scale, y_scale)`` pair. ``y_scale`` may be negative for
      north-up images.

  Returns:
    A list of the 6 affine transform values with updated scale components.

  Raises:
    TypeError: If ``scaling`` is not a length-2 tuple.
  """
  if isinstance(scaling, tuple) and len(scaling) == 2:
    x_scale, y_scale = scaling
    crs_transform[0] = x_scale
    crs_transform[4] = y_scale
  else:
    raise TypeError(f'Expected a tuple of length 2 for scaling, got {scaling}')
  affine_transform = affine.Affine(*crs_transform)
  return list(affine_transform)[:6]


def fit_geometry(
    geometry: shapely.geometry.base.BaseGeometry,
    *,
    geometry_crs: str = 'EPSG:4326',
    buffer: float = 0,
    grid_crs: str = 'EPSG:4326',
    grid_scale: ScalingType | None = None,
    grid_scale_digits: int | None = None,
    grid_shape: ShapeType | None = None,
) -> PixelGridParams:
  """Derive grid parameters that *cover* a geometry.

  You must specify exactly one of ``grid_scale`` (pixel size) or
  ``grid_shape`` (pixel count). When a scale is provided the output pixel
  shape is computed to fully cover the buffered geometry. When a shape is
  provided the scale is inferred uniformly over the geometry's bounding box.

  Args:
    geometry: Shapely geometry defining the area of interest (in
      ``geometry_crs`` units).
    geometry_crs: CRS of the input geometry (default WGS84).
    buffer: Optional positive distance in CRS units to expand the geometry.
    grid_crs: Target CRS for the output grid.
    grid_scale: Optional ``(x_scale, y_scale)`` in ``grid_crs`` units. ``y``
      may be negative for north-up orientation.
    grid_scale_digits: If provided with ``grid_shape`` workflow, round inferred
      scales to this number of decimal places.
    grid_shape: Optional ``(width, height)`` pixel count.

  Returns:
    ``PixelGridParams`` dictionary usable with ``xarray.open_dataset``.

  Raises:
    ValueError: If both or neither of ``grid_scale`` / ``grid_shape`` provided.
    TypeError: If ``grid_scale`` is malformed.
  """

  if (grid_scale is None) == (grid_shape is None):
    raise ValueError(
        "Exactly one of 'grid_scale' or 'grid_shape' must be specified."
    )

  transformer = Transformer.from_crs(
      crs_from=geometry_crs, crs_to=grid_crs, always_xy=True
  )
  reprojected_geometry = transform(transformer.transform, geometry)
  if buffer and buffer > 0:
    buffered_geom = shapely.buffer(reprojected_geometry, buffer)
  else:
    buffered_geom = reprojected_geometry
  x_min, y_min, x_max, y_max = buffered_geom.bounds

  if grid_scale:
    if isinstance(grid_scale, tuple) and len(grid_scale) == 2:
      x_scale, y_scale = grid_scale
    else:
      raise TypeError(
          f'Expected a tuple of length 2 for grid_scale, got {grid_scale}'
      )

    # REVERTED to the more direct and robust shape calculation.
    x_shape = int(math.ceil(x_max / x_scale) - math.floor(x_min / x_scale))
    y_shape = int(
        math.ceil(y_max / abs(y_scale)) - math.floor(y_min / abs(y_scale))
    )
  else:  # grid_shape is not None
    x_shape, y_shape = grid_shape
    x_scale = (x_max - x_min) / x_shape
    y_scale = -(y_max - y_min) / y_shape

    if grid_scale_digits:
      x_scale = round(x_scale, grid_scale_digits)
      y_scale = round(y_scale, grid_scale_digits)

  grid_x_min = math.floor(x_min / x_scale) * x_scale
  grid_y_max = math.ceil(y_max / abs(y_scale)) * abs(y_scale)

  affine_transform = affine.Affine.translation(
      grid_x_min, grid_y_max
  ) * affine.Affine.scale(x_scale, y_scale)

  crs_transform = affine_transform[:6]

  return dict(
      crs=grid_crs, crs_transform=crs_transform, shape_2d=(x_shape, y_shape)
  )


def extract_grid_params(
    ee_obj: Union[ee.Image, ee.ImageCollection],
) -> PixelGridParams:
  """Return native pixel grid parameters for an EE Image or ImageCollection.

  For an ImageCollection, the first image's first band's grid definition is
  used. This matches Earth Engine's internal representation and lets you
  "match source grid" without having to inspect projection metadata manually.

  Args:
    ee_obj: ``ee.Image`` or ``ee.ImageCollection`` instance.

  Returns:
    ``PixelGridParams`` mapping the native CRS, transform, and dimensions.

  Raises:
    TypeError: If ``ee_obj`` is not a supported EE type.
  """

  if isinstance(ee_obj, ee.Image):
    img_obj = ee_obj
  elif isinstance(ee_obj, ee.ImageCollection):
    img_obj = ee_obj.first()
  else:
    raise TypeError(
        f'Expected ee.Image or ee.ImageCollection, got {type(ee_obj)}'
    )

  first_band_info = img_obj.select(0).getInfo()['bands'][0]

  return dict(
      crs=first_band_info['crs'],
      crs_transform=tuple(first_band_info['crs_transform']),
      shape_2d=tuple(first_band_info['dimensions']),
  )
