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
import shapely
import xarray as xr
from xarray.core import indexing
import xee
from xee import helpers

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

# Define grid parameters for tests 
_TEST_GRID_PARAMS = {
  'crs': 'EPSG:4326',
  'crs_transform': (1.0, 0, -180.0, 0, -1.0, 90.0),
  'shape_2d': (360, 180)
}

def _read_identity_pool_creds() -> identity_pool.Credentials:
  credentials_path = os.environ[_CREDENTIALS_PATH_KEY]
  with open(credentials_path) as file:
    json_file = json.load(file)
    credentials = identity_pool.Credentials.from_info(json_file)
    return credentials.with_scopes(_SCOPES)


def init_ee_for_tests():
  init_params = {
      'opt_url': ee.data.HIGH_VOLUME_API_BASE_URL,
  }

  if _CREDENTIALS_PATH_KEY in os.environ:
    credentials = _read_identity_pool_creds()
    init_params['credentials'] = credentials
    init_params['project'] = credentials.project_number
  ee.Initialize(**init_params)


class EEBackendArrayTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    init_ee_for_tests()
    self.store = xee.EarthEngineStore(
        ee.ImageCollection('LANDSAT/LC08/C02/T1').filterDate(
            '2017-01-01', '2017-01-03'
        ),
        n_images=64,
        getitem_kwargs={'max_retries': 10, 'initial_delay': 1500},
        **_TEST_GRID_PARAMS,
    )
    self.store_with_neg_mask_value = xee.EarthEngineStore(
        ee.ImageCollection('LANDSAT/LC08/C02/T1').filterDate(
            '2017-01-01', '2017-01-03'
        ),
        **_TEST_GRID_PARAMS,
        n_images=64,
        mask_value=-9999,
    )
    self.lnglat_store = xee.EarthEngineStore(
        ee.ImageCollection.fromImages([ee.Image.pixelLonLat()]),
        **_TEST_GRID_PARAMS,
        chunks={'index': 256, 'width': 512, 'height': 512},
        n_images=64,
    )
    self.conus_store = xee.EarthEngineStore(
        ee.ImageCollection('GRIDMET/DROUGHT').filterDate(
            '2020-03-30', '2020-04-01'
        ),
        **_TEST_GRID_PARAMS,
        n_images=64,
        getitem_kwargs={'max_retries': 9},
    )
    self.all_img_store = xee.EarthEngineStore(
        ee.ImageCollection('LANDSAT/LC08/C02/T1').filterDate(
            '2017-01-01', '2017-01-03'
        ),
        **_TEST_GRID_PARAMS,
    )

  def test_creates_lat_long_array(self):
    arr = xee.EarthEngineBackendArray('longitude', self.lnglat_store)
    self.assertEqual((1, 360, 180), arr.shape)

  def test_can_create_object(self):
    arr = xee.EarthEngineBackendArray('B4', self.store)

    self.assertIsNotNone(arr)

    self.assertEqual((64, 360, 180), arr.shape)
    self.assertEqual(np.float32, arr.dtype)
    self.assertEqual('B4', arr.variable_name)

  def test_basic_indexing(self):
    arr = xee.EarthEngineBackendArray('B4', self.store)
    self.assertEqual(np.isnan(arr[indexing.BasicIndexer((0, 0, 0))]), True)
    self.assertEqual(np.isnan(arr[indexing.BasicIndexer((-1, -1, -1))]), True)

  def test_basic_indexing_on_int_ee_image(self):
    arr = xee.EarthEngineBackendArray('B4', self.store_with_neg_mask_value)
    self.assertEqual(np.isnan(arr[indexing.BasicIndexer((0, 0, 0))]), True)
    self.assertEqual(np.isnan(arr[indexing.BasicIndexer((-1, -1, -1))]), True)

  def test_basic_indexing__nonzero(self):
    arr = xee.EarthEngineBackendArray('longitude', self.lnglat_store)

    zero_idx = arr[indexing.BasicIndexer((0, 0, 0))]
    self.assertTrue(np.allclose(zero_idx, -179.5), f'Actual: {zero_idx}')

    last_idx = arr[indexing.BasicIndexer((-1, -1, -1))]
    self.assertTrue(np.allclose(last_idx, 179.5), f'Actual: {last_idx}')

  def test_basic_indexing_multiple_images(self):
    arr = xee.EarthEngineBackendArray('B4', self.store)
    first_two = arr[indexing.BasicIndexer((slice(0, 2), 0, 0))]
    np.testing.assert_equal(first_two, np.full(2, np.nan))
    first_three = arr[indexing.BasicIndexer((slice(0, 3), 0, 0))]
    np.testing.assert_equal(first_three, np.full(3, np.nan))
    last_two = arr[indexing.BasicIndexer((slice(-3, -1), 0, 0))]
    np.testing.assert_equal(last_two, np.full(2, np.nan))
    last_three = arr[indexing.BasicIndexer((slice(-4, -1), 0, 0))]
    np.testing.assert_equal(last_three, np.full(3, np.nan))

  def test_slice_indexing(self):
    arr = xee.EarthEngineBackendArray('B5', self.store)
    first_10 = indexing.BasicIndexer((0, slice(0, 10), slice(0, 10)))
    np.testing.assert_equal(arr[first_10], np.full((10, 10), np.nan))
    last_5 = indexing.BasicIndexer((0, slice(-5, -1), slice(-5, -1)))
    expected_last_5 = np.full((4, 4), np.nan)
    np.testing.assert_equal(expected_last_5, arr[last_5])

  def test_slice_indexing__non_global(self):
    arr = xee.EarthEngineBackendArray('spi2y', self.conus_store)
    first_10 = indexing.BasicIndexer((0, slice(0, 10), slice(0, 10)))
    np.testing.assert_equal(arr[first_10], np.full((10, 10), np.nan))
    last_5 = indexing.BasicIndexer((0, slice(-5, -1), slice(-5, -1)))
    expected_last_5 = np.full((4, 4), np.nan)
    np.testing.assert_equal(
        expected_last_5, arr[last_5], f'Actual:\n{arr[last_5]}'
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
    np.testing.assert_equal(arr[first_10], np.full((2, 10, 10), np.nan))
    last_5 = indexing.BasicIndexer(
        (slice(-3, -1), slice(-5, -1), slice(-5, -1))
    )
    expected_last_5 = np.full((2, 4, 4), np.nan)
    np.testing.assert_equal(expected_last_5, arr[last_5])

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

  def test_getitem_kwargs(self):
    arr = xee.EarthEngineBackendArray('B4', self.store)
    self.assertEqual(arr.store.getitem_kwargs['initial_delay'], 1500)
    self.assertEqual(arr.store.getitem_kwargs['max_retries'], 10)

    arr1 = xee.EarthEngineBackendArray('longitude', self.lnglat_store)
    self.assertEqual(arr1.store.getitem_kwargs['initial_delay'], 500)
    self.assertEqual(arr1.store.getitem_kwargs['max_retries'], 6)

    arr2 = xee.EarthEngineBackendArray('spi2y', self.conus_store)
    self.assertEqual(arr2.store.getitem_kwargs['initial_delay'], 500)
    self.assertEqual(arr2.store.getitem_kwargs['max_retries'], 9)


class EEBackendEntrypointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    init_ee_for_tests()
    self.entry = xee.EarthEngineBackendEntrypoint()

  def test_guess_can_open__collection_name(self):
    self.assertTrue(self.entry.guess_can_open('LANDSAT/LC08/C02/T1'))
    self.assertFalse(
        self.entry.guess_can_open('LANDSAT/SomeRandomCollection/C02/T1')
    )
    self.assertTrue(self.entry.guess_can_open('ee://LANDSAT/LC08/C02/T1'))
    self.assertTrue(self.entry.guess_can_open('ee:LANDSAT/LC08/C02/T1'))
    self.assertFalse(self.entry.guess_can_open('ee::LANDSAT/LC08/C02/T1'))

  def test_guess_can_open__image_collection(self):
    ic = ee.ImageCollection('LANDSAT/LC08/C02/T1').filterDate(
        '2017-01-01', '2017-01-03'
    )

    self.assertTrue(self.entry.guess_can_open(ic))
    # Should not be able to open a feature collection.
    self.assertFalse(self.entry.guess_can_open('WRI/GPPD/power_plants'))

  def test_open_dataset__sanity_check(self):
    """Test opening a simple image collection in geographic coordinates."""
    n_images, width, height = 3, 4, 5 
    ds = self.entry.open_dataset(
        pathlib.Path('ECMWF') / 'ERA5' / 'MONTHLY',
        n_images=n_images,
        crs='EPSG:4326',
        crs_transform=(12.0, 0, -180.0, 0, -25.0, 90.0),
        shape_2d=(width, height),
    )
    self.assertEqual(dict(ds.sizes), {'time': 3, 'y': height, 'x': width})
    self.assertNotEmpty(dict(ds.coords))
    self.assertEqual(
      list(ds.data_vars.keys()),
      [
        'mean_2m_air_temperature',
        'minimum_2m_air_temperature',
        'maximum_2m_air_temperature',
        'dewpoint_2m_temperature',
        'total_precipitation',
        'surface_pressure',
        'mean_sea_level_pressure',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m'
      ]
    )
    # Loop through the data variables.
    for v in ds.values():
      self.assertIsNotNone(v.data)
      self.assertFalse(v.isnull().all(), 'All values are null!')
      self.assertEqual(v.shape, (n_images, width, height))


  def test_open_dataset__n_images(self):
    ds = self.entry.open_dataset(
        pathlib.Path('LANDSAT') / 'LC08' / 'C02' / 'T1',
        drop_variables=tuple(f'B{i}' for i in range(3, 12)),
        n_images=1,
        **_TEST_GRID_PARAMS
    )
    self.assertLen(ds.time, 1)

  def test_open_dataset_image_to_imagecollection(self):
    """Ensure that opening an ee.Image is the same as opening a single image ee.ImageCollection."""
    img = ee.Image('CGIAR/SRTM90_V4')
    ic = ee.ImageCollection(img)
    ds1 = xr.open_dataset(img, engine='ee', **_TEST_GRID_PARAMS)
    ds2 = xr.open_dataset(ic, engine='ee', **_TEST_GRID_PARAMS)
    self.assertTrue(ds1.identical(ds2))

  def test_can_chunk__opened_dataset(self):
    ds = xr.open_dataset(
        'NASA/GPM_L3/IMERG_V07',
        engine=xee.EarthEngineBackendEntrypoint,
        **_TEST_GRID_PARAMS
    ).isel(time=slice(0, 1))

    try:
      ds.chunk().compute()
    except ValueError:
      self.fail('Chunking failed.')


  def test_honors_geometry_simple_utm(self):
    """Test that a non-geographic projection can be used."""
    ic = ee.ImageCollection([
      ee.Image('LANDSAT/LC09/C02/T1_L2/LC09_043034_20211116').select(0)
        .addBands(ee.Image.pixelLonLat()),
    ])
    min_x, max_x = 10, 12
    min_y, max_y = -4, 0
    width = max_x - min_x
    height = max_y - min_y
    ds = xr.open_dataset(
        ic,
        engine=xee.EarthEngineBackendEntrypoint,
        crs='EPSG:32610', 
        crs_transform=(30, 0, 448485+103000, 0, -30, 4263915-84000),  # Origin over SF
        shape_2d=(width, height),
    )

    self.assertEqual(ds.sizes, {'time': 1, 'y': height, 'x': width})
    np.testing.assert_allclose(
        ds['latitude'].values, 
        np.array([[
          [37.764977, 37.764706, 37.764435, 37.764164],
          [37.764973, 37.7647  , 37.76443 , 37.764164]
        ]])
    )
    np.testing.assert_allclose(
        ds['longitude'].values, 
        np.array([[
          [-122.41528, -122.41529, -122.41529, -122.41529],
          [-122.41495, -122.41495, -122.41495, -122.41495]
        ]])
    )
    np.testing.assert_allclose(
        ds['SR_B1'].values, 
        np.array([[
          [14332., 13622., 12058., 11264.],
          [12254., 10379., 10701., 11150.]
        ]])
    )


  @absltest.skipIf(_SKIP_RASTERIO_TESTS, 'rioxarray module not loaded')
  def test_expected_precise_transform(self):
    data = np.empty((162, 121), dtype=np.float32)
    bbox = (
        -53.94158617595226,
        -12.078281822698678,
        -53.67209159071253,
        -11.714464132625046,
    )
    x_res = (bbox[2] - bbox[0]) / data.shape[1]
    y_res = (bbox[3] - bbox[1]) / data.shape[0]
    raster = xr.DataArray(
        data,
        coords={
            'y': np.linspace(bbox[3], bbox[1] + x_res, data.shape[0]),
            'x': np.linspace(bbox[0], bbox[2] - y_res, data.shape[1]),
        },
        dims=('y', 'x'),
    )
    raster.rio.write_crs('EPSG:4326', inplace=True)
    ic = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .filterDate(ee.DateRange('2014-01-01', '2014-01-02'))
        .select('precipitation')
    )
    xee_dataset = xr.open_dataset(
        ee.ImageCollection(ic),
        engine='ee',
        crs=str(raster.rio.crs),
        crs_transform=raster.rio.transform()[:6],
        shape_2d=data.shape
    )
    self.assertNotEqual(abs(x_res), abs(y_res))
    np.testing.assert_allclose(
        np.array(xee_dataset.rio.transform()),
        np.array(raster.rio.transform()),
    )

  def test_parses_ee_url(self):
    """Test the ee: URL parsing."""
    n_images, width, height = 3, 10, 20
    test_params = {
      'n_images': n_images,
      'crs': 'EPSG:4326',
      'crs_transform': (12.0, 0, -180.0, 0, -25.0, 90.0),
      'shape_2d': (width, height)
    }
    ds1 = self.entry.open_dataset('ee://LANDSAT/LC08/C02/T1', **test_params)
    ds2 = self.entry.open_dataset('ee:LANDSAT/LC08/C02/T1', **test_params)
    self.assertEqual(dict(ds1.sizes), {'time': n_images, 'y': height, 'x': width})
    self.assertEqual(dict(ds2.sizes), {'time': n_images, 'y': height, 'x': width})
    np.testing.assert_allclose(
      ds1['B1'].compute().values,
      ds2['B1'].compute().values
    )

  def test_data_sanity_check(self):
    # This simple test uncovered a bug with the default definition of `scale`.
    # The issue was that `scale_y` was not being set to a negative value by
    # default. This lead to a junk projection calculation and zeroing all data.
    era5 = xr.open_dataset(
        'ECMWF/ERA5_LAND/HOURLY',
        engine=xee.EarthEngineBackendEntrypoint,
        n_images=1,
        **_TEST_GRID_PARAMS
    )
    temperature_2m = era5.isel(time=0).temperature_2m
    self.assertNotEqual(temperature_2m.min(), 0.0)
    self.assertNotEqual(temperature_2m.max(), 0.0)

  def test_validate_band_attrs(self):
    ds = self.entry.open_dataset(
        'ee:LANDSAT/LC08/C02/T1',
        drop_variables=tuple(f'B{i}' for i in range(3, 12)),
        n_images=3,
        **_TEST_GRID_PARAMS
    )
    valid_types = (str, int, float, complex, np.ndarray, np.number, list, tuple)

    # Check attrs on the dataset itself
    for _, value in ds.attrs.items():
      self.assertIsInstance(value, valid_types)

    # Check attrs on each variable within the dataset
    for variable in ds.variables.values():
      for _, value in variable.attrs.items():
        self.assertIsInstance(value, valid_types)

  def test_fast_time_slicing(self):
    band = 'temperature_2m'
    hourly = (
        ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        .filterDate('2024-01-01', '2024-01-02')
        .select(band)
    )
    first = hourly.first()
    props = ['system:id', 'system:time_start']
    fake_collection = ee.ImageCollection(
        hourly.toList(count=hourly.size()).replace(
            first, ee.Image(0).rename(band).copyProperties(first, props)
        )
    )

    params = dict(
        filename_or_obj=fake_collection,
        engine=xee.EarthEngineBackendEntrypoint,
        crs='EPSG:4326',
        crs_transform=(1, 0, -100, 0, 1, 50),
        shape_2d=(3, 4),
    )

    # With slow slicing, the returned data should include the modified image.
    slow_slicing = xr.open_dataset(**params)
    slow_slicing_data = getattr(slow_slicing[dict(time=0)], band).as_numpy()
    self.assertTrue(np.all(slow_slicing_data == 0))

    # With fast slicing, the returned data should include the original image.
    fast_slicing = xr.open_dataset(**params, fast_time_slicing=True)
    fast_slicing_data = getattr(fast_slicing[dict(time=0)], band).as_numpy()   
    self.assertTrue(np.all(fast_slicing_data > 0))

  @absltest.skipIf(_SKIP_RASTERIO_TESTS, 'rioxarray module not loaded')
  def test_write_projected_dataset_to_raster(self):
    # ensure that a projected dataset written to a raster intersects with the
    # point used to create the initial image collection
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = os.path.join(temp_dir, 'test.tif')

      crs = 'EPSG:32610'
      proj = ee.Projection(crs)
      
      point = shapely.geometry.Point(-122.44, 37.78)
      ee_point = ee.Geometry.Point(list(point.coords)[0])

      # Create a collection of 10 low-cloud images intersecting a point.
      col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      col = col.filterBounds(ee_point)
      col = col.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 5))
      col = col.limit(10)

      grid_dict = helpers.fit_geometry(
        geometry=point.buffer(0.1),
        grid_crs=crs,
        grid_scale=(100, -100)
      )

      ds = xr.open_dataset(
          col,
          engine=xee.EarthEngineBackendEntrypoint,
          **grid_dict
      )

      ds = ds.isel(time=0)
      ds.rio.write_crs(crs, inplace=True)
      ds.rio.reproject(crs, inplace=True)
      ds.rio.to_raster(temp_file)

      with rasterio.open(temp_file) as raster:
        # see https://gis.stackexchange.com/a/407755 for evenOdd explanation
        bbox = ee.Geometry.Rectangle(raster.bounds, proj=proj, evenOdd=False)
        intersects = bbox.intersects(ee_point, 1, proj=proj)
        self.assertTrue(intersects.getInfo())


class GridHelpersTest(absltest.TestCase):
  """Test grid helper functions that require GEE access."""

  def setUp(self):
    super().setUp()
    init_ee_for_tests()
    self.entry = xee.EarthEngineBackendEntrypoint()
  
  def test_extract_grid_params_from_image(self):
    img = ee.Image('LANDSAT/LT05/C02/T1_TOA/LT05_031034_20110619')
    grid_params = helpers.extract_grid_params(img)
    self.assertEqual(grid_params['shape_2d'], (7881, 6981))
    self.assertEqual(grid_params['crs'], 'EPSG:32613')
    np.allclose(grid_params['crs_transform'], [30, 0, 643185, 0, -30, 4255815])

  def test_extract_grid_params_from_image_collection(self):
    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30');
    grid_params = helpers.extract_grid_params(dem)
    self.assertEqual(grid_params['shape_2d'], (3601, 3601))
    self.assertEqual(grid_params['crs'], 'EPSG:4326')
    np.allclose(grid_params['crs_transform'], [0.000278, 0, 5.999861, 0, -0.000278, 1.000139])

  def test_extract_grid_params_from_invalid_object(self):
    with self.assertRaises(TypeError):
      helpers.extract_grid_params('a string object')


class ReadmeCodeTest(absltest.TestCase):
  """Tests a copy of code contained in the Xee README."""

  def setUp(self):
    super().setUp()
    init_ee_for_tests()
    self.entry = xee.EarthEngineBackendEntrypoint()

  def test_extract_projection_from_image(self):

    ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('1992-10-05', '1993-03-31')
    grid_params = helpers.extract_grid_params(ic)

    # Open any Earth Engine ImageCollection by specifying the Xarray engine as 'ee':
    ds = xr.open_dataset(
      'ee://ECMWF/ERA5_LAND/HOURLY',
      engine='ee',
      **grid_params
    )
    
    # Open all bands in a specific projection:
    ds = xr.open_dataset(
      'ee://ECMWF/ERA5_LAND/HOURLY',
      engine='ee',
      crs='EPSG:32610',
      crs_transform=(30, 0, 448485 + 103000, 0, -30, 4263915 - 84000),  # In San Francisco, California
      shape_2d=(64, 64),
    )

    # Open an ImageCollection (maybe, with EE-side filtering or processing):
    ds = xr.open_dataset(
      ic,
      engine='ee',
      crs='EPSG:32610',
      crs_transform=(30, 0, 448485 + 103000, 0, -30, 4263915 - 84000),  # In San Francisco, California
      shape_2d=(64, 64),
    )

    # Open an ImageCollection with a specific EE projection or geometry:

    grid_params = helpers.fit_geometry(
      geometry=shapely.geometry.box(113.33, -43.63, 153.56, -10.66),
      grid_crs='EPSG:4326',
      grid_shape=(256, 256)
    )

    ds = xr.open_dataset(
        ic,
        engine='ee',
        **grid_params
    )

    # Open a single Image:
    img = ee.Image('LANDSAT/LC08/C02/T1_TOA/LC08_044034_20140318')
    grid_params = helpers.extract_grid_params(img)
    ds = xr.open_dataset(
      img,
      engine='ee',
      **grid_params
    )


if __name__ == '__main__':
  absltest.main()
