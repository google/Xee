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
import copy
import functools
import importlib
import itertools
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union
from urllib import parse
import warnings

import affine
import numpy as np
import pandas as pd
import xarray
from xarray import backends
from xarray.backends import common
from xarray.backends import store as backends_store
from xarray.core import indexing
from xarray.core import utils
from xee import types

import ee


assert sys.version_info >= (3, 8)
try:
  __version__ = importlib.metadata.version('xee') or 'unknown'
except importlib.metadata.PackageNotFoundError:
  __version__ = 'unknown'


# Chunks type definition taken from Xarray
# https://github.com/pydata/xarray/blob/f13da94db8ab4b564938a5e67435ac709698f1c9/xarray/core/types.py#L173
#
# The 'int' case let's users specify `io_chunks=-1`, which means to load the
# data as a single chunk.
Chunks = Union[int, Dict[Any, Any], Literal['auto'], None]

# Types for type hints
CrsType = str
TransformType = Tuple[float, float, float, float, float, float]
ShapeType = Tuple[int, int]

_BUILTIN_DTYPES = {
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
}

# Earth Engine image:computePixels request is limited to 48 MB
# https://developers.google.com/earth-engine/reference/rest/v1/projects.image/computePixels
REQUEST_BYTE_LIMIT = 2**20 * 48  # 48 MBs

# Xee uses the ee.ImageCollection.toList function for slicing into an
# ImageCollection. This function isn't optimized for large collections. If the
# end index of the slice is beyond 10k, display a warning to the user. This
# value was chosen by trial and error.
_TO_LIST_WARNING_LIMIT = 10000

EE_AFFINE_TRANSFORM_FIELDS = [
  'scaleX',
  'shearX',
  'translateX',
  'shearY',
  'scaleY',
  'translateY'
]

# Used in ext_test.py.
def _check_request_limit(chunks: Dict[str, int], dtype_size: int, limit: int):
  """Checks that the actual number of bytes exceeds the limit."""
  index, width, height = chunks['index'], chunks['width'], chunks['height']
  # Add one for the mask byte (Earth Engine bytes-per-pixel accounting).
  actual_bytes = index * width * height * (dtype_size + 1)
  if actual_bytes > limit:
    raise ValueError(
        f'`chunks="auto"` failed! Actual bytes {actual_bytes!r} exceeds limit'
        f' {limit!r}.  Please choose another value for `chunks` (and file a'
        ' bug).'
    )


class _GetComputedPixels:
  """Wrapper around `ee.data.computePixels()` to make retries simple."""

  def __getitem__(self, params) -> np.ndarray:
    return ee.data.computePixels(params)


class EarthEngineStore(common.AbstractDataStore):
  """Read-only Data Store for Google Earth Engine."""

  # "Safe" default chunks that won't exceed the request limit.
  PREFERRED_CHUNKS: Dict[str, int] = {
      'index': 48,
      'width': 256,
      'height': 256,
  }

  GETITEM_KWARGS: Dict[str, int] = {
      'max_retries': 6,
      'initial_delay': 500,
  }

  SCALE_UNITS: Dict[str, int] = {
      'degree': 1,
      'metre': 10_000,
      'meter': 10_000,
      'm': 10_000,
  }

  DEFAULT_MASK_VALUE = np.iinfo(np.int32).max

  ATTRS_VALID_TYPES = (
      str,
      int,
      float,
      complex,
      np.ndarray,
      np.number,
      list,
      tuple,
  )

  @classmethod
  def open(
      cls,
      image_collection: ee.ImageCollection,
      crs: CrsType,
      crs_transform: TransformType,
      shape_2d: ShapeType,
      mode: Literal['r'] = 'r',
      chunk_store: Chunks = None,
      n_images: int = -1,
      primary_dim_name: Optional[str] = None,
      primary_dim_property: Optional[str] = None,
      mask_value: Optional[float] = None,
      request_byte_limit: int = REQUEST_BYTE_LIMIT,
      ee_init_kwargs: Optional[Dict[str, Any]] = None,
      ee_init_if_necessary: bool = False,
      executor_kwargs: Optional[Dict[str, Any]] = None,
      getitem_kwargs: Optional[Dict[str, int]] = None,
      fast_time_slicing: bool = False,
  ) -> 'EarthEngineStore':
    if mode != 'r':
      raise ValueError(
          f'mode {mode!r} is invalid: data can only be read from Earth Engine.'
      )

    return cls(
        image_collection,
        crs=crs,
        crs_transform=crs_transform,
        shape_2d=shape_2d,
        chunks=chunk_store,
        n_images=n_images,
        primary_dim_name=primary_dim_name,
        primary_dim_property=primary_dim_property,
        mask_value=mask_value,
        request_byte_limit=request_byte_limit,
        ee_init_kwargs=ee_init_kwargs,
        ee_init_if_necessary=ee_init_if_necessary,
        executor_kwargs=executor_kwargs,
        getitem_kwargs=getitem_kwargs,
        fast_time_slicing=fast_time_slicing,
    )

  def __init__(
      self,
      image_collection: ee.ImageCollection,
      crs: CrsType,
      crs_transform: TransformType,
      shape_2d: ShapeType,
      chunks: Chunks = None,
      n_images: int = -1,
      primary_dim_name: Optional[str] = None,
      primary_dim_property: Optional[str] = None,
      mask_value: Optional[float] = None,
      request_byte_limit: int = REQUEST_BYTE_LIMIT,
      ee_init_kwargs: Optional[Dict[str, Any]] = None,
      ee_init_if_necessary: bool = False,
      executor_kwargs: Optional[Dict[str, Any]] = None,
      getitem_kwargs: Optional[Dict[str, int]] = None,
      fast_time_slicing: bool = False,
  ):
    self.ee_init_kwargs = ee_init_kwargs
    self.ee_init_if_necessary = ee_init_if_necessary
    self.fast_time_slicing = fast_time_slicing

    # Initialize executor_kwargs
    if executor_kwargs is None:
      executor_kwargs = {}
    self.executor_kwargs = executor_kwargs

    self.getitem_kwargs = {**self.GETITEM_KWARGS, **(getitem_kwargs or {})}

    self.image_collection = image_collection
    if n_images != -1:
      self.image_collection = image_collection.limit(n_images)

    self.crs = crs
    self.crs_transform = crs_transform
    self.shape_2d = shape_2d

    self.primary_dim_name = primary_dim_name or 'time'
    self.primary_dim_property = primary_dim_property or 'system:time_start'

    self.n_images = self.get_info['size']
    self._props = self.get_info['props']
    #  Metadata should apply to all imgs.
    self._img_info: types.ImageInfo = self.get_info['first']

    self.dimension_names = ('x', 'y')
    self._props = self._make_attrs_valid(self._props)
    self.scale_x, self.scale_y = crs_transform[0], crs_transform[4]
    affine_transform = affine.Affine(*crs_transform)
    self.scale = np.sqrt(np.abs(affine_transform.determinant))

    max_dtype = self._max_itemsize()

    # TODO(b/291851322): Consider support for laziness when chunks=None.
    # By default, automatically optimize io_chunks.
    self.chunks = self._auto_chunks(max_dtype, request_byte_limit)
    if chunks == -1:
      self.chunks = -1
    elif chunks is not None and chunks != 'auto':
      self.chunks = self._assign_index_chunks(chunks)

    self.preferred_chunks = self._assign_preferred_chunks()
    if mask_value is None:
      self.mask_value = self.DEFAULT_MASK_VALUE
    else:
      self.mask_value = mask_value

  @functools.cached_property
  def get_info(self) -> Dict[str, Any]:
    """Make all getInfo() calls to EE at once."""

    rpcs = [
        ('size', self.image_collection.size()),
        ('props', self.image_collection.toDictionary()),
        ('first', self.image_collection.first()),
    ]


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
        (
            'properties',
            (
                self.image_collection.reduceColumns(
                    ee.Reducer.toList().repeat(len(columns)), columns
                ).get('list')
            ),
        )
    )

    info = ee.List([rpc for _, rpc in rpcs]).getInfo()

    return dict(zip((name for name, _ in rpcs), info))

  @property
  def image_collection_properties(self) -> Tuple[List[str], List[str]]:
    system_ids, primary_coord = self.get_info['properties']
    return (system_ids, primary_coord)

  @property
  def image_ids(self) -> List[str]:
    image_ids, _ = self.image_collection_properties
    return image_ids

  def _max_itemsize(self) -> int:
    return max(
        _parse_dtype(b['data_type']).itemsize for b in self._img_info['bands']
    )

  @classmethod
  def _auto_chunks(
      cls, dtype_bytes: int, request_byte_limit: int = REQUEST_BYTE_LIMIT
  ) -> Dict[str, int]:
    """Given the data type size and request limit, calculate optimal chunks."""
    # Taking the data type number of bytes into account, let's try to have the
    # height and width follow round numbers (powers of two) and allocate the
    # remaining bytes available for the index length. To illustrate this logic,
    # let's follow through with an example where:
    #   request_byte_limit = 2 ** 20 * 48  # = 48 MBs
    #   dtype_bytes = 8

    log_total = np.log2(request_byte_limit)  # e.g.=25.58...
    # Add one for the mask byte (Earth Engine bytes-per-pixel accounting).
    log_dtype = np.log2(dtype_bytes + 1)  # e.g.=3.16...
    log_limit = 10 * (log_total // 10)  # e.g.=20
    log_index = log_total - log_limit  # e.g.=5.58...

    # Motivation: How do we divide a number N into the closest sum of two ints?
    d = (log_limit - np.ceil(log_dtype)) / 2  # e.g.=16/2=8.0
    wd, ht = np.ceil(d), np.floor(d)  # e.g. wd=8, ht=8

    # Put back to byte space, then round to the nearst integer number of bytes.
    index = int(np.rint(2**log_index))  # e.g.=48
    width = int(np.rint(2**wd))  # e.g.=256
    height = int(np.rint(2**ht))  # e.g.=256

    return {'index': index, 'width': width, 'height': height}

  def _assign_index_chunks(
      self, input_chunk_store: Dict[Any, Any]
  ) -> Dict[Any, Any]:
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


  def project(self, bbox: types.BBox) -> types.Grid:
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
    x_start, y_start, x_end, y_end = bbox

    # Translate the crs_transform to the origin of the bounding box
    transform_image = affine.Affine(*self.crs_transform)
    transform_grid_cell = affine.Affine.translation(
      xoff=x_start * transform_image.a,
      yoff=y_start * transform_image.e
    ) * transform_image

    return {
        # The size of the bounding box. The affine transform and project will be
        # applied, so we can think in terms of pixels.
        'dimensions': {
            'width': x_end - x_start,
            'height': y_end - y_start,
        },
        'affineTransform': dict(zip(EE_AFFINE_TRANSFORM_FIELDS, transform_grid_cell)),
        'crsCode': self.crs,
    }

  def image_to_array(
      self,
      image: ee.Image,
      pixels_getter=_GetComputedPixels(),
      dtype=np.float32,
      **kwargs,
  ) -> np.ndarray:
    """Gets the pixels for a given image as a numpy array.

    This method includes exponential backoff (with jitter) when trying to get
    pixel data.

    Args:
      image: An EE image.
      pixels_getter: An object whose `__getitem__()` method calls
        `computePixels()`.
      dtype: a np.dtype. The returned array will be in this dtype.
      **kwargs: Additional settings for `params` in `computePixels(params)`. For
        example, a `grid` dictionary.

    Returns:
      A numpy array containing the pixels computed based on the given image.
    """
    image = image.unmask(self.mask_value, False)
    params = {
        'expression': image,
        'fileFormat': 'NUMPY_NDARRAY',
        **kwargs,
    }
    raw = common.robust_getitem(
        pixels_getter,
        params,
        catch=ee.ee_exception.EEException,
        **self.getitem_kwargs,
    )

    # Extract out the shape information from EE response.
    y_size, x_size = raw.shape
    n_bands = len(raw.dtype)

    # Get a view (no copy) of the data as the returned type from EE
    # then reshape to the correct shape based on the request.
    # This is needed because `raw` is a structured array of all the same dtype
    # (i.e. number of images) and this converts it to an ndarray.
    arr = raw.view(raw.dtype[0]).reshape(
        y_size,
        x_size,
        n_bands,
    )

    # try converting the data to desired dtype in place without copying
    # if conversion is not allowed then just use the EE returned dtype
    try:
      arr = arr.astype(dtype, copy=False)
    except ValueError:
      warnings.warn(
          f'Could convert EE results to requested dtype {dtype} '
          f'falling back to returned dtype from EE {np.dtype(raw.dtype[0])}'
      )

    data = arr.T
    current_mask_value = np.array(self.mask_value, dtype=data.dtype)
    # Sets EE nodata masked value to NaNs.
    data = np.where(data == current_mask_value, np.nan, data)
    return data

  @functools.lru_cache()
  def _band_attrs(self, band_name: str) -> types.BandInfo:
    try:
      return next((b for b in self._img_info['bands'] if b['id'] == band_name))
    except StopIteration as e:
      raise ValueError(f'Band {band_name!r} not found.') from e

  @functools.lru_cache()
  def _bands(self) -> List[str]:
    return [b['id'] for b in self._img_info['bands']]

  def _make_attrs_valid(self, attrs: Dict[str, Any]) -> Dict[
      str,
      Union[
          str, int, float, complex, np.ndarray, np.number, List[Any], Tuple[Any]
      ],
  ]:
    return {
        key: (
            str(value)
            if not isinstance(value, self.ATTRS_VALID_TYPES)
            else value
        )
        for key, value in attrs.items()
    }

  def open_store_variable(self, name: str) -> xarray.Variable:
    arr = EarthEngineBackendArray(name, self)
    data = indexing.LazilyIndexedArray(arr)

    x_dim_name, y_dim_name = self.dimension_names
    dimensions = [self.primary_dim_name, x_dim_name, y_dim_name]
    attrs = self._make_attrs_valid(self._band_attrs(name))
    attrs['crs'] = str(self.crs)
    encoding = {
        'source': attrs['id'],
        'scale_factor': arr.scale,
        'dtype': data.dtype,
        'preferred_chunks': self.preferred_chunks,
    }

    return xarray.Variable(dimensions, data, attrs, encoding)

  def get_dimensions(self) -> utils.Frozen[str, int]:
    return utils.FrozenDict((name, 3) for name in self._bands())

  def get_attrs(self) -> utils.Frozen[Any, Any]:
    return utils.FrozenDict(self._props)

  def _get_primary_coordinates(self) -> List[Any]:
    """Gets the primary dimension coordinate values from an ImageCollection."""
    _, primary_coords = self.image_collection_properties

    if not primary_coords:
      raise ValueError(
          f'No {self.primary_dim_property!r} values found in the'
          " 'ImageCollection'"
      )
    if self.primary_dim_property in ['system:time_start', 'system:time_end']:
      # Convert elements in primary_coords to a timestamp.
      primary_coords = [
          pd.to_datetime(time, unit='ms') for time in primary_coords
      ]
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

    x_scale, _, x_translate, _, y_scale, y_translate = self.crs_transform
    width, height = self.shape_2d
    width_coord = np.array([x_translate  + x_scale / 2 + ix * x_scale for ix in range(width)])
    height_coord = np.array([y_translate  + y_scale / 2 + iy * y_scale for iy in range(height)])

    # Make sure there's at least a single point in each dimension.
    if width_coord.ndim == 0:
      width_coord = width_coord[None, ...]
    if height_coord.ndim == 0:
      height_coord = height_coord[None, ...]

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


def _ee_bounds_to_bounds(bounds: Dict[str, Any]) -> types.Bounds:
  coords = np.array(bounds['coordinates'], dtype=np.float64)[0]
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


class EarthEngineBackendArray(backends.BackendArray):
  """Array backend for Earth Engine."""

  def __init__(self, variable_name: str, ee_store: EarthEngineStore):
    self.variable_name = variable_name
    self.store = ee_store

    self.scale = ee_store.scale

    # It looks like different bands have different dimensions & transforms!
    # Can we get this into consistent dimensions?
    self._info = ee_store._band_attrs(variable_name)
    self.dtype = np.dtype(np.float32)

    self.shape = (ee_store.n_images, ) + ee_store.shape_2d
    self._apparent_chunks = {k: 1 for k in self.store.PREFERRED_CHUNKS.keys()}
    if isinstance(self.store.chunks, dict):
      self._apparent_chunks = self.store.chunks.copy()

  def _ee_init_check(self):
    if not ee.data.is_initialized() and self.store.ee_init_if_necessary:
      warnings.warn(
          'Earth Engine is not initialized on worker. '
          'Attempting to initialize using application default credentials.'
      )

      ee_init_kwargs = _parse_ee_init_kwargs(self.store.ee_init_kwargs)
      ee.Initialize(**ee_init_kwargs)

  def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
    return indexing.explicit_indexing_adapter(
        key,
        self.shape,
        indexing.IndexingSupport.BASIC,
        self._raw_indexing_method,
    )

  def _key_to_slices(
      self, key: Tuple[Union[int, slice], ...]
  ) -> Tuple[Tuple[slice, ...], Tuple[int, ...]]:
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
    self._ee_init_check()
    start, stop, stride = image_slice.indices(self.shape[0])

    if self.store.fast_time_slicing and self.store.image_ids:
      imgs = self.store.image_ids[start:stop:stride]
    else:
      if self.store.fast_time_slicing:
        logging.warning(
            "fast_time_slicing is enabled but ImageCollection images don't have"
            ' IDs. Reverting to default behavior.'
        )
      if stop > _TO_LIST_WARNING_LIMIT:
        logging.warning(
            'Xee is indexing into the ImageCollection beyond %s images. This'
            ' operation can be slow. To improve performance, consider filtering'
            ' the ImageCollection prior to using Xee or enabling'
            ' fast_time_slicing.',
            _TO_LIST_WARNING_LIMIT,
        )
      # TODO(alxr, mahrsee): Find a way to make this case more efficient.
      list_range = stop - start
      imgs = self.store.image_collection.toList(list_range, offset=start).slice(
          0, list_range, stride
      )

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
      self, key: Tuple[Union[int, slice], ...]
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
      out = self.store.image_to_array(
          target_image, grid=self.store.project(bbox), dtype=self.dtype
      )

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

    with concurrent.futures.ThreadPoolExecutor(
        **self.store.executor_kwargs
    ) as pool:
      for (i, j, k), arr in pool.map(
          self._make_tile, self._tile_indexes(key[0], bbox)
      ):
        tiles[i][j][k] = arr

    out = np.block(tiles)

    if squeeze_axes:
      out = np.squeeze(out, squeeze_axes)

    return out

  def _make_tile(
      self, tile_index: Tuple[types.TileIndex, types.BBox3d]
  ) -> Tuple[types.TileIndex, np.ndarray]:
    """Get a numpy array from EE for a specific 3D bounding box (a 'tile')."""
    tile_idx, (istart, iend, *bbox) = tile_index
    target_image = self._slice_collection(slice(istart, iend))
    return tile_idx, self.store.image_to_array(
        target_image, grid=self.store.project(tuple(bbox)), dtype=self.dtype
    )

  def _tile_indexes(
      self, index_range: slice, bbox: types.BBox
  ) -> Iterable[Tuple[types.TileIndex, types.BBox3d]]:
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
      self, filename_or_obj: Union[str, os.PathLike[Any], ee.ImageCollection]
  ) -> bool:  # type: ignore
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
      crs: CrsType,
      crs_transform: TransformType,
      shape_2d: ShapeType,
      drop_variables: Optional[Tuple[str, ...]] = None,
      io_chunks: Optional[Any] = None,
      n_images: int = -1,
      mask_and_scale: bool = True,
      decode_times: bool = True,
      decode_timedelta: Optional[bool] = None,
      use_cftime: Optional[bool] = None,
      concat_characters: bool = True,
      decode_coords: bool = True,
      primary_dim_name: Optional[str] = None,
      primary_dim_property: Optional[str] = None,
      ee_mask_value: Optional[float] = None,
      request_byte_limit: int = REQUEST_BYTE_LIMIT,
      ee_init_if_necessary: bool = False,
      ee_init_kwargs: Optional[Dict[str, Any]] = None,
      executor_kwargs: Optional[Dict[str, Any]] = None,
      getitem_kwargs: Optional[Dict[str, int]] = None,
      fast_time_slicing: bool = False,
  ) -> xarray.Dataset:  # type: ignore
    """Open an Earth Engine ImageCollection as an Xarray Dataset.

    Args:
      filename_or_obj: An asset ID for an ImageCollection, or an
        ee.ImageCollection object.
      crs: The coordinate reference system (a CRS code or WKT
        string). This defines the frame of reference to coalesce all variables
        upon opening.
      crs_transform: Transform matrix describing the grid origin and scale
        relative to the CRS.
      shape_2d: Dimensions of the pixel grid in the form (width, height). 
      drop_variables (optional): Variables or bands to drop before opening.
      io_chunks (optional): Specifies the chunking strategy for loading data
        from EE. By default, this automatically calculates optional chunks based
        on the `request_byte_limit`.
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
      primary_dim_name (optional): Override the name of the primary dimension of
        the output Dataset. By default, the name is 'time'.
      primary_dim_property (optional): Override the `ee.Image` property for
        which to derive the values of the primary dimension. By default, this is
        'system:time_start'.
      ee_mask_value (optional): Value to mask to EE nodata values. By default,
        this is 'np.iinfo(np.int32).max' i.e. 2147483647.
      request_byte_limit: the max allowed bytes to request at a time from Earth
        Engine. By default, it is 48MBs.
      ee_init_if_necessary: boolean flag to set if auto initialize for Earth
        Engine should be attempted. Set to True if using distributed compute
        frameworks.
      ee_init_kwargs: keywords to pass to Earth Engine Initialize when
        attempting to auto init for remote workers.
      executor_kwargs (optional): A dictionary of keyword arguments to pass to
        the ThreadPoolExecutor that handles the parallel computation of pixels
        i.e. {'max_workers': 2}.
      getitem_kwargs (optional): Exponential backoff kwargs passed into the
        xarray function to index the array (`robust_getitem`).
        - 'max_retries', the maximum number of retry attempts. Defaults to 6.
        - 'initial_delay', the initial delay in milliseconds before the first
          retry. Defaults to 500.
      fast_time_slicing (optional): Whether to perform an optimization that
        makes slicing an ImageCollection across time faster. This optimization
        loads EE images in a slice by ID, so any modifications to images in a
        computed ImageCollection will not be reflected.
    Returns:
      An xarray.Dataset that streams in remote data from Earth Engine.
    """

    user_agent = f'Xee/{__version__}'
    if ee.data.getUserAgent() != user_agent:
      ee.data.setUserAgent(user_agent)

    if isinstance(filename_or_obj, ee.ImageCollection):
      collection = filename_or_obj
    elif isinstance(filename_or_obj, ee.Image):
      collection = ee.ImageCollection(filename_or_obj)
    else:
      collection = ee.ImageCollection(self._parse(filename_or_obj))

    store = EarthEngineStore.open(
        collection,
        crs=crs,
        crs_transform=crs_transform,
        shape_2d=shape_2d,
        chunk_store=io_chunks,
        n_images=n_images,
        primary_dim_name=primary_dim_name,
        primary_dim_property=primary_dim_property,
        mask_value=ee_mask_value,
        request_byte_limit=request_byte_limit,
        ee_init_kwargs=ee_init_kwargs,
        ee_init_if_necessary=ee_init_if_necessary,
        executor_kwargs=executor_kwargs,
        getitem_kwargs=getitem_kwargs,
        fast_time_slicing=fast_time_slicing,
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


def _parse_ee_init_kwargs(
    ee_init_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
  """Parses Earth Engine Initialize kwargs.

  Generate credentials if credentials_function is specified.

  Args:
    ee_init_kwargs: A dictionary of keyword arguments to pass to Earth Engine
      Initialize, or None.

  Returns:
    A dictionary of keyword arguments to pass to Earth Engine Initialize.
  """
  ee_init_kwargs = copy.copy(ee_init_kwargs) or {}
  if (
      'credentials' in ee_init_kwargs
      and 'credentials_function' in ee_init_kwargs
  ):
    raise ValueError(
        'Cannot specify both credentials and credentials_function.'
    )
  if 'credentials_function' in ee_init_kwargs:
    credentials_function: Callable[[], Any] = ee_init_kwargs.pop(
        'credentials_function'
    )
    ee_init_kwargs['credentials'] = credentials_function()

  return ee_init_kwargs
