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
"""Implementation of the Google Earth Engine extension for Xarray."""

# pylint: disable=g-bad-todo

from __future__ import annotations

import concurrent.futures
import functools
import math
import os
from typing import Any, Iterable, Literal, Optional, Union
from urllib import parse
import warnings

import affine
import numpy as np
import pyproj
from pyproj.crs import CRS
import xarray
from xarray import backends
from xarray.backends import common
from xarray.backends import store as backends_store
from xarray.core import indexing
from xarray.core import utils
from xee import types

import ee


# Chunks type definition taken from Xarray
# https://github.com/pydata/xarray/blob/f13da94db8ab4b564938a5e67435ac709698f1c9/xarray/core/types.py#L173
#
# The 'int' case let's users specify `chunks=-1`, which means to load the data
# as a single chunk.
Chunks = Union[int, dict[Any, Any], Literal['auto'], None]


_BUILTIN_DTYPES = {
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
}


class EarthEngineStore(common.AbstractDataStore):
  """Read-only Data Store for Google Earth Engine."""

  # "Safe" default chunks that won't exceed the request limit.
  PREFERRED_CHUNKS: dict[str, int] = {
      'index': 24,
      'width': 512,
      'height': 512,
  }

  SCALE_UNITS: dict[str, int] = {
      'degree': 1,
      'metre': 10_000,
      'meter': 10_000,
  }

  DIMENSION_NAMES: dict[str, tuple[str, str]] = {
      'degree': ('lon', 'lat'),
      'metre': ('X', 'Y'),
      'meter': ('X', 'Y'),
  }

  @classmethod
  def open(
      cls,
      image_collection: ee.ImageCollection,
      mode: Literal['r'] = 'r',
      chunk_store: Chunks = None,
      n_images: int = -1,
      crs: Optional[str] = None,
      scale: Optional[float] = None,
      projection: Optional[ee.Projection] = None,
      geometry: Optional[ee.Geometry] = None,
      primary_dim_name: Optional[str] = None,
      primary_dim_property: Optional[str] = None,
  ) -> 'EarthEngineStore':
    if mode != 'r':
      raise ValueError(
          f'mode {mode!r} is invalid: data can only be read from Earth Engine.'
      )

    return cls(
        image_collection,
        chunks=chunk_store,
        n_images=n_images,
        crs=crs,
        scale=scale,
        projection=projection,
        geometry=geometry,
        primary_dim_name=primary_dim_name,
        primary_dim_property=primary_dim_property,
    )

  def __init__(
      self,
      image_collection: ee.ImageCollection,
      chunks: Chunks = None,
      n_images: int = -1,
      crs: Optional[str] = None,
      scale: Union[float, int, None] = None,
      projection: Optional[ee.Projection] = None,
      geometry: Optional[ee.Geometry] = None,
      primary_dim_name: Optional[str] = None,
      primary_dim_property: Optional[str] = None,
  ):
    self.image_collection = image_collection
    if n_images != -1:
      self.image_collection = image_collection.limit(n_images)

    self.projection = projection
    self.geometry = geometry
    self.primary_dim_name = primary_dim_name or 'time'
    self.primary_dim_property = primary_dim_property or 'system:time_start'

    self.n_images = self.get_info['size']
    self._props = self.get_info['props']
    #  Metadata should apply to all imgs.
    self._img_info: types.ImageInfo = self.get_info['first']

    proj = self.get_info.get('projection', {})

    self.crs_arg = crs or proj.get('crs', proj.get('wkt', 'EPSG:4326'))
    self.crs = CRS(self.crs_arg)
    # Gets the unit i.e. meter, degree etc.
    self.scale_units = self.crs.axis_info[0].unit_name
    # Get the dimensions name based on the CRS (scale units).
    self.dimension_names = self.DIMENSION_NAMES.get(
        self.scale_units, ('X', 'Y')
    )
    x_dim_name, y_dim_name = self.dimension_names
    self._props.update(
        coordinates=f'{self.primary_dim_name} {x_dim_name} {y_dim_name}',
        crs=self.crs_arg,
    )

    # Scale in the projection's units. Typically, either meters or degrees.
    # If we use the default CRS i.e. EPSG:3857, the units is in meters.
    default_scale = self.SCALE_UNITS.get(self.scale_units, 1)
    if scale is None:
      scale = default_scale
    default_transform = affine.Affine.scale(scale)

    transform = affine.Affine(*proj.get('transform', default_transform)[:6])
    self.scale_x, self.scale_y = transform.a, transform.e
    self.scale = np.sqrt(np.abs(transform.determinant))

    # Parse the dataset bounds from the native projection (either from the CRS
    # or the image geometry) and translate it to the representation that will be
    # used for all internal `computePixels()` calls.
    try:
      if isinstance(geometry, ee.Geometry):
        x_min_0, y_min_0, x_max_0, y_max_0 = _ee_bounds_to_bounds(
            self.get_info['bounds']
        )
      else:
        x_min_0, y_min_0, x_max_0, y_max_0 = self.crs.area_of_use.bounds
    except AttributeError:
      # `area_of_use` is probable `None`. Parse the geometry from the first
      # image instead (calculated in self.get_info())
      x_min_0, y_min_0, x_max_0, y_max_0 = _ee_bounds_to_bounds(
          self.get_info['bounds']
      )
    # We add and subtract the scale to solve an off-by-one error. With this
    # adjustment, we achieve parity with a pure `computePixels()` call.
    x_min, y_min = self.project(x_min_0 - self.scale_x, y_min_0)
    if _bounds_are_invalid(x_min, y_min, self.scale_units == 'degree'):
      x_min, y_min = self.project(x_min_0, y_min_0)
    x_max, y_max = self.project(x_max_0, y_max_0 + self.scale_y)
    if _bounds_are_invalid(x_max, y_max, self.scale_units == 'degree'):
      x_max, y_max = self.project(x_max_0, y_max_0)
    self.bounds = x_min, y_min, x_max, y_max

    self.chunks = self.PREFERRED_CHUNKS.copy()
    if chunks == -1:
      self.chunks = -1
    # TODO(b/291851322): Consider support for laziness when chunks=None.
    elif chunks is not None and chunks != 'auto':
      self.chunks = self._assign_index_chunks(chunks)

    self.preferred_chunks = self._assign_preferred_chunks()

  @functools.cached_property
  def get_info(self) -> dict[str, Any]:
    """Make all getInfo() calls to EE at once."""

    rpcs = [
        ('size', self.image_collection.size()),
        ('props', self.image_collection.toDictionary()),
        ('first', self.image_collection.first()),
    ]

    if isinstance(self.projection, ee.Projection):
      rpcs.append(('projection', self.projection))

    if isinstance(self.geometry, ee.Geometry):
      rpcs.append(('bounds', self.geometry.bounds()))
    else:
      rpcs.append(('bounds', self.image_collection.first().geometry().bounds()))

    # TODO(#29, #30): This RPC call takes the longest time to compute. This
    # requires a full scan of the images in the collection, which happens on the
    # EE backend. This is essential because we want the primary dimension of the
    # opened dataset to be something relevant to the data, like time (start
    # time) as opposed to a random index number.
    #
    # One optimization that could prove really fruitful: read the first and last
    # (few) values of the primary dim (read: time) and interpolate the rest
    # client-side. Ideally, this would live behind a xarray-backend-specific
    # feature flag, since it's not guaranteed that data is this consistent.
    columns = ['system:id', self.primary_dim_property]
    rpcs.append(
        ('properties',
         (
             self.image_collection
             .reduceColumns(ee.Reducer.toList().repeat(len(columns)), columns)
             .get('list'))
         )
    )

    info = ee.List([rpc for _, rpc in rpcs]).getInfo()

    return dict(zip((name for name, _ in rpcs), info))

  @property
  def image_collection_properties(self) -> tuple[list[str], list[str]]:
    system_ids, primary_coord = self.get_info['properties']
    return (system_ids, primary_coord)

  @property
  def image_ids(self) -> list[str]:
    image_ids, _ = self.image_collection_properties
    return image_ids

  def _assign_index_chunks(
      self, input_chunk_store: dict[Any, Any]
  ) -> dict[Any, Any]:
    """Assigns values of 'index', 'width', and 'height' to `self.chunks`.

    This method first attempts to retrieve values for 'index', 'width',
    and 'height' from the 'input_chunk_store' dictionary. If the values are not
    found in 'input_chunk_store', it falls back to default values stored in
    'self.PREFERRED_CHUNKS'.

    Args:
      input_chunk_store (dict): how to break up the data into chunks.

    Returns:
      dict: A dictionary containing 'index', 'width', and 'height' values.
    """
    chunks = {}
    x_dim_name, y_dim_name = self.dimension_names
    for key, dim_name in [
        ('index', self.primary_dim_name),
        ('width', x_dim_name),
        ('height', y_dim_name),
    ]:
      chunks[key] = (
          input_chunk_store.get(dim_name)
          or input_chunk_store.get(key)
          or self.PREFERRED_CHUNKS[key]
      )
    return chunks

  def _assign_preferred_chunks(self) -> Chunks:
    chunks = {}
    x_dim_name, y_dim_name = self.dimension_names
    if self.chunks == -1:
      chunks[self.primary_dim_name] = self.PREFERRED_CHUNKS['index']
      chunks[x_dim_name] = self.PREFERRED_CHUNKS['width']
      chunks[y_dim_name] = self.PREFERRED_CHUNKS['height']
    else:
      chunks[self.primary_dim_name] = self.chunks['index']
      chunks[x_dim_name] = self.chunks['width']
      chunks[y_dim_name] = self.chunks['height']
    return chunks

  def project(self, xs: float, ys: float) -> tuple[float, float]:
    transformer = pyproj.Transformer.from_crs(
        self.crs.geodetic_crs, self.crs, always_xy=True
    )
    return transformer.transform(xs, ys)

  @functools.lru_cache()
  def _band_attrs(self, band_name: str) -> types.BandInfo:
    try:
      return next((b for b in self._img_info['bands'] if b['id'] == band_name))
    except StopIteration as e:
      raise ValueError(f'Band {band_name!r} not found.') from e

  @functools.lru_cache()
  def _bands(self) -> list[str]:
    return [b['id'] for b in self._img_info['bands']]

  def open_store_variable(self, name: str) -> xarray.Variable:
    arr = EarthEngineBackendArray(name, self)
    data = indexing.LazilyIndexedArray(arr)

    x_dim_name, y_dim_name = self.dimension_names
    dimensions = [self.primary_dim_name, x_dim_name, y_dim_name]
    attrs = self._band_attrs(name)
    encoding = {
        'source': attrs['id'],
        'scale_factor': arr.scale,
        'scale_units': self.scale_units,
        'dtype': data.dtype,
        'preferred_chunks': self.preferred_chunks,
        'bounds': arr.bounds,
    }

    return xarray.Variable(dimensions, data, attrs, encoding)

  def get_dimensions(self) -> utils.Frozen[str, int]:
    return utils.FrozenDict((name, 3) for name in self._bands())

  def get_attrs(self) -> utils.Frozen[Any, Any]:
    return utils.FrozenDict(self._props)

  def _get_primary_coordinates(self) -> list[Any]:
    """Gets the primary dimension coordinate values from an ImageCollection."""
    _, primary_coords = self.image_collection_properties

    if not primary_coords:
      raise ValueError(
          f'No {self.primary_dim_property!r} values found in the'
          " 'ImageCollection'"
      )
    if self.primary_dim_property in ['system:time_start', 'system:time_end']:
      # Convert elements in primary_dim_list to np.datetime64
      primary_coords = [np.datetime64(time, 'ms') for time in primary_coords]
    return primary_coords

  def get_variables(self) -> utils.Frozen[str, xarray.Variable]:
    vars_ = [(name, self.open_store_variable(name)) for name in self._bands()]

    # Assume all vars will have the same bounds...
    v0 = vars_[0][1]

    primary_coord = np.arange(v0.shape[0])
    try:
      primary_coord = self._get_primary_coordinates()
    except (ee.EEException, ValueError) as e:
      warnings.warn(
          f'Unable to retrieve {self.primary_dim_property!r} values from an '
          f'ImageCollection due to: {e}.'
      )

    x_min_0, y_min_0, x_max_0, y_max_0 = self.bounds
    width_coord = np.linspace(x_min_0, x_max_0, v0.shape[1])
    height_coord = np.linspace(y_max_0, y_min_0, v0.shape[2])

    x_dim_name, y_dim_name = self.dimension_names

    coords = [
        (
            self.primary_dim_name,
            xarray.Variable(self.primary_dim_name, primary_coord),
        ),
        (x_dim_name, xarray.Variable(x_dim_name, width_coord)),
        (y_dim_name, xarray.Variable(y_dim_name, height_coord)),
    ]

    return utils.FrozenDict(vars_ + coords)

  def close(self) -> None:
    del self.image_collection


def _bounds_are_invalid(x: float, y: float, is_degrees=False) -> bool:
  """Check for obviously bad x and y projection values."""
  bad_num = (
      math.isnan(x)
      or math.isnan(y)
      or math.isinf(x)
      or math.isinf(y)
  )

  invalid_degree = (
      y < -90.0
      or y > 90.0
      or x < -180.0
      or x > 360.0  # degrees could be from 0 to 360...
  )

  return bad_num or (is_degrees and invalid_degree)


def _parse_dtype(data_type: types.DataType):
  """Parse a np.dtype from the 'data_type' section of ee.Image.getInfo().

  See https://developers.google.com/earth-engine/apidocs/ee-pixeltype.

  Args:
    data_type: result of a getInfo() call of an Image.

  Returns:
    A numpy.dtype object that best corresponds to the Band data type.
  """
  type_ = data_type['type']
  if type_ == 'PixelType':
    type_ = data_type['precision']

  if type_ in _BUILTIN_DTYPES:
    dt = _BUILTIN_DTYPES[type_]
  else:
    dt = getattr(np, type_)

  return np.dtype(dt)


def _ee_bounds_to_bounds(bounds: ee.Bounds) -> types.Bounds:
  coords = np.array(bounds['coordinates'], dtype=np.float32)[0]
  x_min, y_min, x_max, y_max = (
      min(coords[:, 0]),
      min(coords[:, 1]),
      max(coords[:, 0]),
      max(coords[:, 1]),
  )
  return x_min, y_min, x_max, y_max


def geometry_to_bounds(geom: ee.Geometry) -> types.Bounds:
  """Finds the CRS bounds from a ee.Geometry polygon."""
  bounds = geom.bounds().getInfo()
  return _ee_bounds_to_bounds(bounds)


class _GetComputedPixels:
  """Wrapper around `ee.data.computePixels()` to make retries simple."""

  def __getitem__(self, params) -> np.ndarray:
    return ee.data.computePixels(params)


class EarthEngineBackendArray(backends.BackendArray):
  """Array backend for Earth Engine."""

  def __init__(self, variable_name: str, ee_store: EarthEngineStore):
    self.variable_name = variable_name
    self.store = ee_store

    self.scale_x = ee_store.scale_x
    self.scale_y = ee_store.scale_y
    self.scale = ee_store.scale
    self.crs_arg = ee_store.crs_arg
    self.crs = ee_store.crs
    self.bounds = ee_store.bounds

    # It looks like different bands have different dimensions & transforms!
    # Can we get this into consistent dimensions?
    self._info = ee_store._band_attrs(variable_name)
    self.dtype = _parse_dtype(self._info['data_type'])

    x_min, y_min, x_max, y_max = self.bounds

    x_size = int(np.ceil((x_max - x_min) / np.abs(self.scale_x)))
    y_size = int(np.ceil((y_max - y_min) / np.abs(self.scale_y)))

    self.shape = (ee_store.n_images, x_size, y_size)
    self._apparent_chunks = {k: 1 for k in self.store.PREFERRED_CHUNKS.keys()}
    if isinstance(self.store.chunks, dict):
      self._apparent_chunks = self.store.chunks.copy()

  def _to_array(
      self, image: ee.Image, pixels_getter=_GetComputedPixels(), **kwargs
  ) -> np.ndarray:
    """Gets the pixels for a given image as a numpy array.

    This method includes exponential backoff (with jitter) when trying to get
    pixel data.

    Args:
      image: An EE image.
      pixels_getter: An object whose `__getitem__()` method calls
        `computePixels()`.
      **kwargs: Additional settings for `params` in `computePixels(params)`. For
        example, a `grid` dictionary.

    Returns:
      An numpy array containing the pixels computed based on the given image.
    """
    params = {
        'expression': image,
        'fileFormat': 'NUMPY_NDARRAY',
        **kwargs,
    }
    raw = common.robust_getitem(
        pixels_getter, params, catch=ee.ee_exception.EEException
    )

    # TODO(#9): Find a way to make this more efficient. This is needed because
    # `raw` is a structured array of all the same dtype (i.e. number of images).
    arr = np.array(raw.tolist(), dtype=self.dtype)
    data = arr.T

    return data

  def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
    return indexing.explicit_indexing_adapter(
        key,
        self.shape,
        indexing.IndexingSupport.BASIC,
        self._raw_indexing_method,
    )

  def _project(self, bbox: types.BBox) -> types.Grid:
    """Translate a bounding box (pixel space) to a grid (projection space).

    Here, we calculate a simple affine transformation to get a specific region
    when computing pixels.

    Args:
      bbox: Bounding box in pixel space.

    Returns:
      A Grid, to be passed into `computePixels()`'s "grid" keyword. Defines the
        appropriate region of data to return according to the Array's configured
        projection and scale.
    """
    # The origin of the image is in the top left corner. X is the minimum value
    # and Y is the maximum value.
    x_origin, _, _, y_origin = self.bounds  # x_min, x_max, y_min, y_max
    x_start, y_start, x_end, y_end = bbox
    width = x_end - x_start
    height = y_end - y_start

    return {
        # The size of the bounding box. The affine transform and project will be
        # applied, so we can think in terms of pixels.
        'dimensions': {
            'width': width,
            'height': height,
        },
        'affineTransform': {
            # Since the origin is in the top left corner, we want to translate
            # the start of the grid to the positive direction for X and the
            # negative direction for Y.
            'translateX': x_origin + self.scale_x * x_start,
            'translateY': y_origin + self.scale_y * y_start,
            # Define the scale for each pixel (e.g. the number of meters between
            # each value).
            'scaleX': self.scale_x,
            'scaleY': self.scale_y,
        },
        'crsCode': self.crs_arg,
    }

  def _key_to_slices(
      self, key: tuple[Union[int, slice], ...]
  ) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    """Convert all key indexes to slices.

    If any keys are integers, convert them to a slice (i.e. with a range of 1
    element). This way, we only have to handle getting slices. When we
    encounter integers, we keep track of the axies in a `squeeze_axes` tuple.
    Later, this will be used to squeeze the output array.

    Args:
     key: A key along the Array shape; a tuple of either ints or slices.

    Returns:
      A `key` tuple where all elements are `slice`s, and a `squeeze_axes` tuple,
      which can be used as the second argument to np.squeeze().
    """
    key_new = []
    squeeze_axes = []
    for axis, k in enumerate(key):
      if isinstance(k, int):
        squeeze_axes.append(axis)
        k = slice(k, k + 1)
      key_new.append(k)

    return tuple(key_new), tuple(squeeze_axes)

  def _slice_collection(self, image_slice: slice) -> ee.Image:
    """Reduce the ImageCollection into an Image with bands as index slices."""
    # Get the right range of Images in the collection, either a single image or
    # a range of images...
    start, stop, stride = image_slice.indices(self.shape[0])

    # If the input images have IDs, just slice them. Otherwise, we need to do
    # an expensive `toList()` operation.
    if self.store.image_ids:
      imgs = self.store.image_ids[start:stop:stride]
    else:
      # TODO(alxr, mahrsee): Find a way to make this case more efficient.
      list_range = stop - start
      col0 = self.store.image_collection
      imgs = col0.toList(list_range, offset=start).slice(0, list_range, stride)

    col = ee.ImageCollection(imgs)

    # For a more efficient slice of the series of images, we reduce each
    # image in the collection to bands on a single image.
    def reduce_bands(x, acc):
      return ee.Image(acc).addBands(x, [self.variable_name])

    aggregate_images_as_bands = ee.Image(col.iterate(reduce_bands, ee.Image()))
    # Remove the first "constant" band from the reduction.
    target_image = aggregate_images_as_bands.select(
        aggregate_images_as_bands.bandNames().slice(1)
    )

    return target_image

  def _raw_indexing_method(
      self, key: tuple[Union[int, slice], ...]
  ) -> np.typing.ArrayLike:
    key, squeeze_axes = self._key_to_slices(key)

    # TODO(#13): honor step increments
    strt, stop, _ = key[0].indices(self.shape[0])
    wmin, wmax, _ = key[1].indices(self.shape[1])
    hmin, hmax, _ = key[2].indices(self.shape[2])
    bbox = wmin, hmin, wmax, hmax
    i_range = stop - strt
    h_range = hmax - hmin
    w_range = wmax - wmin

    # User does not want to use any chunks...
    if self.store.chunks == -1:
      target_image = self._slice_collection(key[0])
      out = self._to_array(target_image, grid=self._project(bbox))

      if squeeze_axes:
        out = np.squeeze(out, squeeze_axes)

      return out

    # Here, we break up the requested bounding box into smaller bounding boxes
    # that are at most the chunk size. We will divide up the requests for
    # pixels across a thread pool. We then need to combine all the arrays into
    # one big array.
    #
    # Lucky for us, Numpy provides a specialized "concat"-like operation for
    # contiguous arrays organized in tiles: `np.block()`. If we have arrays
    # [[a, b,], [c, d]], `np.block()` will arrange them as follows:
    #   AAAbb
    #   AAAbb
    #   cccDD
    #   cccDD

    # Create an empty 3d list of lists to store arrays to be combined.
    # TODO(#10): can this be a np.array of objects?
    shape = (
        math.ceil(i_range / self._apparent_chunks['index']),
        math.ceil(w_range / self._apparent_chunks['width']),
        math.ceil(h_range / self._apparent_chunks['height']),
    )
    tiles = [
        [[None for _ in range(shape[2])] for _ in range(shape[1])]
        for _ in range(shape[0])
    ]

    # TODO(#11): Allow users to configure this via kwargs.
    with concurrent.futures.ThreadPoolExecutor() as pool:
      for (i, j, k), arr in pool.map(
          self._make_tile, self._tile_indexes(key[0], bbox)
      ):
        tiles[i][j][k] = arr

    out = np.block(tiles)

    if squeeze_axes:
      out = np.squeeze(out, squeeze_axes)

    return out

  def _make_tile(
      self, tile_index: tuple[types.TileIndex, types.BBox3d]
  ) -> tuple[types.TileIndex, np.ndarray]:
    """Get a numpy array from EE for a specific 3D bounding box (a 'tile')."""
    tile_idx, (istart, iend, *bbox) = tile_index
    target_image = self._slice_collection(slice(istart, iend))
    return tile_idx, self._to_array(
        target_image, grid=self._project(tuple(bbox))
    )

  def _tile_indexes(
      self, index_range: slice, bbox: types.BBox
  ) -> Iterable[tuple[types.TileIndex, types.BBox3d]]:
    """Calculate indexes to break up a (3D) bounding box into chunks."""
    tstep = self._apparent_chunks['index']
    wstep = self._apparent_chunks['width']
    hstep = self._apparent_chunks['height']

    start, stop, _ = index_range.indices(self.shape[0])
    wmin, hmin, wmax, hmax = bbox

    for i, t0 in enumerate(range(start, stop + 1, tstep)):
      for j, w0 in enumerate(range(wmin, wmax + 1, wstep)):
        for k, h0 in enumerate(range(hmin, hmax + 1, hstep)):
          t1 = min(t0 + tstep, stop)
          w1 = min(w0 + wstep, wmax)
          h1 = min(h0 + hstep, hmax)
          if t1 != t0 and w1 != w0 and h1 != h0:
            yield (i, j, k), (t0, t1, w0, h0, w1, h1)


class EarthEngineBackendEntrypoint(backends.BackendEntrypoint):
  """Backend for Earth Engine."""

  def _parse(self, filename_or_obj: Union[str, os.PathLike[Any]]) -> str:
    parsed = parse.urlparse(str(filename_or_obj))
    if parsed.scheme and parsed.scheme != 'ee':
      raise ValueError(
          'uri must follow the format `ee://<image/collection/path>` or '
          '`ee:<image/collection/path>`.'
      )
    return f'{parsed.netloc}{parsed.path}'

  def guess_can_open(
      self,
      filename_or_obj: Union[str, os.PathLike[Any], ee.ImageCollection]
  ) -> bool:
    """Returns True if the candidate is a valid ImageCollection."""
    if isinstance(filename_or_obj, ee.ImageCollection):
      return True
    uri = self._parse(filename_or_obj)
    # check if an image collection is in the earth engine catalog:
    try:
      ee.data.listAssets({'parent': uri, 'pageSize': 1})
      return True
    except ee.EEException:
      return False

  def open_dataset(
      self,
      filename_or_obj: Union[str, os.PathLike[Any], ee.ImageCollection],
      drop_variables: Optional[tuple[str, ...]] = None,
      chunk_store: Optional[Any] = None,
      n_images: int = -1,
      mask_and_scale: bool = True,
      decode_times: bool = True,
      decode_timedelta: Optional[bool] = None,
      use_cftime: Optional[bool] = None,
      concat_characters: bool = True,
      decode_coords: bool = True,
      crs: Optional[str] = None,
      scale: Union[float, int, None] = None,
      projection: Optional[ee.Projection] = None,
      geometry: Optional[ee.Geometry] = None,
      primary_dim_name: Optional[str] = None,
      primary_dim_property: Optional[str] = None,
  ) -> xarray.Dataset:
    """Open an Earth Engine ImageCollection as an Xarray Dataset.

    Args:
      filename_or_obj: An asset ID for an ImageCollection, or an
        ee.ImageCollection object.
      drop_variables (optional): Variables or bands to drop before opening.
      chunk_store (optioanl): how to break up the data into chunks.
      n_images (optional): The max number of EE images in the collection to
        open. Useful when there are a large number of images in the collection
        since calculating collection size can be slow. -1 indicates that all
        images should be included.
      mask_and_scale (optional): Lazily scale (using scale_factor and
        add_offset) and mask (using _FillValue).
      decode_times (optional): Decode cf times (e.g., integers since "hours
        since 2000-01-01") to np.datetime64.
      decode_timedelta (optional): If True, decode variables and coordinates
        with time units in {"days", "hours", "minutes", "seconds",
        "milliseconds", "microseconds"} into timedelta objects. If False, leave
        them encoded as numbers. If None (default), assume the same value of
        decode_time.
      use_cftime (optional): Only relevant if encoded dates come from a standard
        calendar (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.
      concat_characters (optional): Should character arrays be concatenated to
        strings, for example: ["h", "e", "l", "l", "o"] -> "hello"
      decode_coords (optional): bool or {"coordinates", "all"}, Controls which
        variables are set as coordinate variables: - "coordinates" or True: Set
        variables referred to in the ``'coordinates'`` attribute of the datasets
        or individual variables as coordinate variables. - "all": Set variables
        referred to in  ``'grid_mapping'``, ``'bounds'`` and other attributes as
        coordinate variables.
      crs (optional): The coordinate reference system (a CRS code or WKT
        string). This defines the frame of reference to coalesce all variables
        upon opening. By default, data is opened with `EPSG:4326'.
      scale (optional): The scale in the `crs` or `projection`'s units of
        measure -- either meters or degrees. This defines the scale that all
        data is represented in upon opening. By default, the scale is 1° when
        the CRS is in degrees or 10,000 when in meters.
      projection (optional): Specify an `ee.Projection` object to define the
        `scale` and `crs` (or other coordinate reference system) with which to
        coalesce all variables upon opening. By default, the scale and reference
        system is set by the the `crs` and `scale` arguments.
      geometry (optional): Specify an `ee.Geometry` to define the regional
        bounds when opening the data. When not set, the bounds are defined
        by the CRS's 'area_of_use` boundaries. If those aren't present, the
        bounds are derived from the geometry of the first image of the
        collection.
      primary_dim_name (optional): Override the name of the primary dimension of
        the output Dataset. By default, the name is 'time'.
      primary_dim_property (optional): Override the `ee.Image` property for
        which to derive the values of the primary dimension. By default, this is
        'system:time_start'.

    Returns:
      An xarray.Dataset that streams in remote data from Earth Engine.
    """

    collection = (
        filename_or_obj if isinstance(filename_or_obj, ee.ImageCollection)
        else ee.ImageCollection(self._parse(filename_or_obj))
    )

    store = EarthEngineStore.open(
        collection,
        chunk_store=chunk_store,
        n_images=n_images,
        crs=crs,
        scale=scale,
        projection=projection,
        geometry=geometry,
        primary_dim_name=primary_dim_name,
        primary_dim_property=primary_dim_property,
    )

    store_entrypoint = backends_store.StoreBackendEntrypoint()

    with utils.close_on_error(store):
      ds = store_entrypoint.open_dataset(
          store,
          mask_and_scale=mask_and_scale,
          decode_times=decode_times,
          concat_characters=concat_characters,
          decode_coords=decode_coords,
          drop_variables=drop_variables,
          use_cftime=use_cftime,
          decode_timedelta=decode_timedelta,
      )

    return ds
