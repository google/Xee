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
r"""Integration tests for the Google Earth Engine backend for Xarray."""
import json
import os
import pathlib
import tempfile

from absl.testing import absltest
from google.auth import identity_pool
import numpy as np
import xarray as xr
from xarray.core import indexing
import xee

import ee

_SKIP_RASTERIO_TESTS = False
try:
  import rasterio  # pylint: disable=g-import-not-at-top
  import rioxarray  # pylint: disable=g-import-not-at-top,unused-import
except ImportError:
  _SKIP_RASTERIO_TESTS = True

_CREDENTIALS_PATH_KEY = 'GOOGLE_APPLICATION_CREDENTIALS'
_SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/earthengine',
]


def _read_identity_pool_creds() -> identity_pool.Credentials:
  credentials_path = os.environ[_CREDENTIALS_PATH_KEY]
  with open(credentials_path) as file:
    json_file = json.load(file)
    credentials = identity_pool.Credentials.from_info(json_file)
    return credentials.with_scopes(_SCOPES)


def init_ee_for_tests():
  ee.Initialize(
      credentials=_read_identity_pool_creds(),
      opt_url=ee.data.HIGH_VOLUME_API_BASE_URL,
  )


class EEBackendArrayTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    init_ee_for_tests()
    self.store = xee.EarthEngineStore(
        ee.ImageCollection('LANDSAT/LC08/C01/T1').filterDate(
            '2017-01-01', '2017-01-03'
        ),
        n_images=64,
    )
    self.lnglat_store = xee.EarthEngineStore(
        ee.ImageCollection.fromImages([ee.Image.pixelLonLat()]),
        chunks={'index': 256, 'width': 512, 'height': 512},
        n_images=64,
    )
    self.conus_store = xee.EarthEngineStore(
        ee.ImageCollection('GRIDMET/DROUGHT').filterDate(
            '2020-03-30', '2020-04-01'
        ),
        n_images=64,
    )
    self.all_img_store = xee.EarthEngineStore(
        ee.ImageCollection('LANDSAT/LC08/C01/T1').filterDate(
            '2017-01-01', '2017-01-03'
        )
    )

  def test_creates_lat_long_array(self):
    arr = xee.EarthEngineBackendArray('longitude', self.lnglat_store)
    self.assertEqual((1, 360, 180), arr.shape)

  def test_can_create_object(self):
    arr = xee.EarthEngineBackendArray('B4', self.store)

    self.assertIsNotNone(arr)

    self.assertEqual((64, 360, 180), arr.shape)
    self.assertEqual(np.int32, arr.dtype)
    self.assertEqual('B4', arr.variable_name)

  def test_basic_indexing(self):
    arr = xee.EarthEngineBackendArray('B4', self.store)
    self.assertEqual(arr[indexing.BasicIndexer((0, 0, 0))], 0)
    self.assertEqual(arr[indexing.BasicIndexer((-1, -1, -1))], np.array([0]))

  def test_basic_indexing__nonzero(self):
    arr = xee.EarthEngineBackendArray('longitude', self.lnglat_store)

    zero_idx = arr[indexing.BasicIndexer((0, 0, 0))]
    self.assertTrue(np.allclose(zero_idx, -179.5), f'Actual: {zero_idx}')

    last_idx = arr[indexing.BasicIndexer((-1, -1, -1))]
    self.assertTrue(np.allclose(last_idx, 179.5), f'Actual: {last_idx}')

  def test_basic_indexing_multiple_images(self):
    arr = xee.EarthEngineBackendArray('B4', self.store)
    first_two = arr[indexing.BasicIndexer((slice(0, 2), 0, 0))]
    self.assertTrue(np.allclose(first_two, np.array([0, 0])))
    first_three = arr[indexing.BasicIndexer((slice(0, 3), 0, 0))]
    self.assertTrue(np.allclose(first_three, np.array([0, 0, 0])))
    last_two = arr[indexing.BasicIndexer((slice(-3, -1), 0, 0))]
    self.assertTrue(np.allclose(last_two, np.array([0, 0])))
    last_three = arr[indexing.BasicIndexer((slice(-4, -1), 0, 0))]
    self.assertTrue(np.allclose(last_three, np.array([0, 0, 0])))

  def test_slice_indexing(self):
    arr = xee.EarthEngineBackendArray('B5', self.store)
    first_10 = indexing.BasicIndexer((0, slice(0, 10), slice(0, 10)))
    self.assertTrue(np.allclose(arr[first_10], np.zeros((10, 10))))
    last_5 = indexing.BasicIndexer((0, slice(-5, -1), slice(-5, -1)))
    expected_last_5 = np.zeros((4, 4))
    self.assertTrue(
        np.allclose(expected_last_5, arr[last_5]), f'Actual:\n{arr[last_5]}'
    )

  def test_slice_indexing__non_global(self):
    arr = xee.EarthEngineBackendArray('spi2y', self.conus_store)
    first_10 = indexing.BasicIndexer((0, slice(0, 10), slice(0, 10)))
    self.assertTrue(np.allclose(arr[first_10], np.zeros((10, 10))))
    last_5 = indexing.BasicIndexer((0, slice(-5, -1), slice(-5, -1)))
    expected_last_5 = np.zeros((4, 4))
    self.assertTrue(
        np.allclose(expected_last_5, arr[last_5]), f'Actual:\n{arr[last_5]}'
    )

  # TODO(alxr): Add more tests here to check for off-by-one errors...
  def test_chunk_bboxes(self):
    arr = xee.EarthEngineBackendArray('latitude', self.lnglat_store)
    bbox = 500, 500, 1025, 1025
    actual = list(arr._tile_indexes(slice(0, 1), bbox))
    self.assertEqual(
        [
            ((0, 0, 0), (0, 1, 500, 500, 1012, 1012)),
            ((0, 0, 1), (0, 1, 500, 1012, 1012, 1025)),
            ((0, 1, 0), (0, 1, 1012, 500, 1025, 1012)),
            ((0, 1, 1), (0, 1, 1012, 1012, 1025, 1025)),
        ],
        actual,
    )

  def test_keys_to_slices(self):
    arr = xee.EarthEngineBackendArray('latitude', self.lnglat_store)

    scalar = (100, 512, 512)
    actual, actual_squeeze = arr._key_to_slices(scalar)
    self.assertEqual(
        actual, (slice(100, 101), slice(512, 513), slice(512, 513))
    )
    self.assertEqual(actual_squeeze, (0, 1, 2))

    vector = (slice(10, 20), 477, 111)
    actual, actual_squeeze = arr._key_to_slices(vector)
    self.assertEqual(actual, (slice(10, 20), slice(477, 478), slice(111, 112)))
    self.assertEqual(actual_squeeze, (1, 2))

    w_vector = (64, slice(128, 512), 99)
    _, actual_squeeze = arr._key_to_slices(w_vector)
    self.assertEqual(actual_squeeze, (0, 2))

    h_vector = (7, 10, slice(11, 13))
    _, actual_squeeze = arr._key_to_slices(h_vector)
    self.assertEqual(actual_squeeze, (0, 1))

    identity = np.zeros((512, 512, 512))
    top_corner = (511, 511, 511)
    actual, actual_squeeze = arr._key_to_slices(top_corner)
    self.assertEqual(identity[actual], 0.0)
    self.assertEqual(actual_squeeze, (0, 1, 2))

  def test_slice_indexing_multiple_images(self):
    arr = xee.EarthEngineBackendArray('B5', self.store)
    first_10 = indexing.BasicIndexer((slice(0, 2), slice(0, 10), slice(0, 10)))
    self.assertTrue(np.allclose(arr[first_10], np.zeros((2, 10, 10))))
    last_5 = indexing.BasicIndexer(
        (slice(-3, -1), slice(-5, -1), slice(-5, -1))
    )
    expected_last_5 = np.zeros((2, 4, 4))
    self.assertTrue(
        np.allclose(expected_last_5, arr[last_5]), f'Actual:\n{arr[last_5]}'
    )

  def test_slice_indexing__medium(self):
    try:
      # divisible
      arr = xee.EarthEngineBackendArray('B5', self.store)
      big_slice = (slice(0, 256), slice(0, 1024), slice(0, 1024))
      _ = arr[indexing.BasicIndexer(big_slice)]
      # remainder
      arr = xee.EarthEngineBackendArray('B5', self.store)
      big_slice = (slice(0, 256), slice(0, 513), slice(0, 513))
      _ = arr[indexing.BasicIndexer(big_slice)]
    except ee.ee_exception.EEException:
      self.fail('Hit API query limits.')

  def test_slice_indexing__medium__non_global(self):
    try:
      # divisible
      arr = xee.EarthEngineBackendArray('spi2y', self.conus_store)
      big_slice = (slice(0, 256), slice(0, 1024), slice(0, 1024))
      _ = arr[indexing.BasicIndexer(big_slice)]
      # remainder
      arr = xee.EarthEngineBackendArray('spi2y', self.conus_store)
      big_slice = (slice(0, 256), slice(0, 513), slice(0, 513))
      _ = arr[indexing.BasicIndexer(big_slice)]
    except ee.ee_exception.EEException:
      self.fail('Hit API query limits.')

  def test_slice_indexing__big(self):
    try:
      arr = xee.EarthEngineBackendArray('B5', self.store)
      big_slice = (slice(0, 256), slice(0, 32768), slice(0, 32768))
      _ = arr[indexing.BasicIndexer(big_slice)]
    except ee.ee_exception.EEException:
      self.fail('Hit API query limits.')

  def test__to_array__retries_on_error(self):
    class ErroneousPixelsGetter:

      def __init__(self):
        self.count = 0

      def __getitem__(self, params):
        if self.count < 3:
          self.count += 1
          raise ee.ee_exception.EEException('Too many requests!')
        return ee.data.computePixels(params)

    arr = xee.EarthEngineBackendArray('B5', self.store)
    grid = self.store.project((0, 10, 0, 10))
    getter = ErroneousPixelsGetter()
    self.store.image_to_array(
        self.store.image_collection.first(),
        pixels_getter=getter,
        grid=grid,
        dtype=arr.dtype,
    )

    self.assertEqual(getter.count, 3)


class EEBackendEntrypointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    init_ee_for_tests()
    self.entry = xee.EarthEngineBackendEntrypoint()

  def test_guess_can_open__collection_name(self):
    self.assertTrue(self.entry.guess_can_open('LANDSAT/LC08/C01/T1'))
    self.assertFalse(
        self.entry.guess_can_open('LANDSAT/SomeRandomCollection/C01/T1')
    )
    self.assertTrue(self.entry.guess_can_open('ee://LANDSAT/LC08/C01/T1'))
    self.assertTrue(self.entry.guess_can_open('ee:LANDSAT/LC08/C01/T1'))
    self.assertFalse(self.entry.guess_can_open('ee::LANDSAT/LC08/C01/T1'))

  def test_guess_can_open__image_collection(self):
    ic = ee.ImageCollection('LANDSAT/LC08/C01/T1').filterDate(
        '2017-01-01', '2017-01-03'
    )

    self.assertTrue(self.entry.guess_can_open(ic))
    # Should not be able to open a feature collection.
    self.assertFalse(self.entry.guess_can_open('WRI/GPPD/power_plants'))

  def test_open_dataset__sanity_check(self):
    ds = self.entry.open_dataset(
        pathlib.Path('LANDSAT') / 'LC08' / 'C01' / 'T1',
        drop_variables=tuple(f'B{i}' for i in range(3, 12)),
        scale=25.0,  # in degrees
        n_images=3,
    )
    self.assertEqual(dict(ds.dims), {'time': 3, 'lon': 14, 'lat': 7})
    self.assertNotEmpty(dict(ds.coords))
    self.assertEqual(
        list(ds.data_vars.keys()),
        [f'B{i}' for i in range(1, 3)] + ['BQA'],
    )
    for v in ds.values():
      self.assertIsNotNone(v.data)
      self.assertFalse(v.isnull().all(), 'All values are null!')
      self.assertEqual(v.shape, (3, 14, 7))

  def test_open_dataset__n_images(self):
    ds = self.entry.open_dataset(
        pathlib.Path('LANDSAT') / 'LC08' / 'C01' / 'T1',
        drop_variables=tuple(f'B{i}' for i in range(3, 12)),
        n_images=1,
        scale=25.0,  # in degrees
    )

    self.assertLen(ds.time, 1)

  def test_can_chunk__opened_dataset(self):
    ds = xr.open_dataset(
        'NASA/GPM_L3/IMERG_V06',
        crs='EPSG:4326',
        scale=0.25,
        engine=xee.EarthEngineBackendEntrypoint,
    ).isel(time=slice(0, 1))

    try:
      ds.chunk().compute()
    except ValueError:
      self.fail('Chunking failed.')

  def test_honors_geometry(self):
    ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(
        '1992-10-05', '1993-03-31'
    )
    leg1 = ee.Geometry.Rectangle(113.33, -43.63, 153.56, -10.66)
    ds = xr.open_dataset(
        ic,
        engine=xee.EarthEngineBackendEntrypoint,
        geometry=leg1,
    )
    standard_ds = xr.open_dataset(
        ic,
        engine=xee.EarthEngineBackendEntrypoint,
    )

    self.assertEqual(ds.dims, {'time': 4248, 'lon': 40, 'lat': 35})
    self.assertNotEqual(ds.dims, standard_ds.dims)

  def test_honors_projection(self):
    ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(
        '1992-10-05', '1993-03-31'
    )
    ds = xr.open_dataset(
        ic,
        engine=xee.EarthEngineBackendEntrypoint,
        projection=ic.first().select(0).projection(),
    )
    standard_ds = xr.open_dataset(
        ic,
        engine=xee.EarthEngineBackendEntrypoint,
    )

    self.assertEqual(ds.dims, {'time': 4248, 'lon': 3600, 'lat': 1800})
    self.assertNotEqual(ds.dims, standard_ds.dims)

  def test_parses_ee_url(self):
    ds = self.entry.open_dataset(
        'ee://LANDSAT/LC08/C01/T1',
        drop_variables=tuple(f'B{i}' for i in range(3, 12)),
        scale=25.0,  # in degrees
        n_images=3,
    )
    self.assertEqual(dict(ds.dims), {'time': 3, 'lon': 14, 'lat': 7})
    ds = self.entry.open_dataset(
        'ee:LANDSAT/LC08/C01/T1',
        drop_variables=tuple(f'B{i}' for i in range(3, 12)),
        scale=25.0,  # in degrees
        n_images=3,
    )
    self.assertEqual(dict(ds.dims), {'time': 3, 'lon': 14, 'lat': 7})

  def test_data_sanity_check(self):
    # This simple test uncovered a bug with the default definition of `scale`.
    # The issue was that `scale_y` was not being set to a negative value by
    # default. This lead to a junk projection calculation and zeroing all data.
    era5 = xr.open_dataset(
        'ECMWF/ERA5_LAND/HOURLY',
        engine=xee.EarthEngineBackendEntrypoint,
        n_images=1,
    )
    temperature_2m = era5.isel(time=0).temperature_2m
    self.assertNotEqual(temperature_2m.min(), 0.0)
    self.assertNotEqual(temperature_2m.max(), 0.0)

  def test_validate_band_attrs(self):
    ds = self.entry.open_dataset(
        'ee:LANDSAT/LC08/C01/T1',
        drop_variables=tuple(f'B{i}' for i in range(3, 12)),
        scale=25.0,  # in degrees
        n_images=3,
    )
    valid_types = (str, int, float, complex, np.ndarray, np.number, list, tuple)

    # Check attrs on the dataset itself
    for _, value in ds.attrs.items():
      self.assertIsInstance(value, valid_types)

    # Check attrs on each variable within the dataset
    for variable in ds.variables.values():
      for _, value in variable.attrs.items():
        self.assertIsInstance(value, valid_types)

  @absltest.skipIf(_SKIP_RASTERIO_TESTS, 'rioxarray module not loaded')
  def test_write_projected_dataset_to_raster(self):
    # ensure that a projected dataset written to a raster intersects with the
    # point used to create the initial image collection
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = os.path.join(temp_dir, 'test.tif')

      crs = 'epsg:32610'
      proj = ee.Projection(crs)
      point = ee.Geometry.Point([-122.44, 37.78])
      geom = point.buffer(1024).bounds()

      col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      col = col.filterBounds(point)
      col = col.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 5))
      col = col.limit(10)

      ds = xr.open_dataset(
          col,
          engine=xee.EarthEngineBackendEntrypoint,
          scale=10,
          crs=crs,
          geometry=geom,
      )

      ds = ds.isel(time=0).transpose('Y', 'X')
      ds.rio.set_spatial_dims(x_dim='X', y_dim='Y', inplace=True)
      ds.rio.write_crs(crs, inplace=True)
      ds.rio.reproject(crs, inplace=True)
      ds.rio.to_raster(temp_file)

      with rasterio.open(temp_file) as raster:
        # see https://gis.stackexchange.com/a/407755 for evenOdd explanation
        bbox = ee.Geometry.Rectangle(raster.bounds, proj=proj, evenOdd=False)
        intersects = bbox.intersects(point, 1, proj=proj)
        self.assertTrue(intersects.getInfo())

  @absltest.skipIf(_SKIP_RASTERIO_TESTS, 'rioxarray module not loaded')
  def test_write_dataset_to_raster(self):
    # ensure that a dataset written to a raster intersects with the point used
    # to create the initial image collection
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = os.path.join(temp_dir, 'test.tif')

      crs = 'EPSG:4326'
      proj = ee.Projection(crs)
      point = ee.Geometry.Point([-122.44, 37.78])
      geom = point.buffer(1024).bounds()

      col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      col = col.filterBounds(point)
      col = col.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 5))
      col = col.limit(10)

      ds = xr.open_dataset(
          col,
          engine=xee.EarthEngineBackendEntrypoint,
          scale=0.0025,
          geometry=geom,
      )

      ds = ds.isel(time=0).transpose('lat', 'lon')
      ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
      ds.rio.write_crs(crs, inplace=True)
      ds.rio.reproject(crs, inplace=True)
      ds.rio.to_raster(temp_file)

      with rasterio.open(temp_file) as raster:
        # see https://gis.stackexchange.com/a/407755 for evenOdd explanation
        bbox = ee.Geometry.Rectangle(raster.bounds, proj=proj, evenOdd=False)
        intersects = bbox.intersects(point, 1, proj=proj)
        self.assertTrue(intersects.getInfo())


if __name__ == '__main__':
  absltest.main()
