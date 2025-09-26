"""Xee Unit Tests."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import affine
import shapely
from unittest import mock
import xee
from xee import ext
from xee import helpers


class EEStoreStandardDatatypesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='int8',
          dtype=np.dtype('int8'),
          expected_chunks={'index': 48, 'width': 1024, 'height': 512},
      ),
      dict(
          testcase_name='int32',
          dtype=np.dtype('int32'),
          expected_chunks={'index': 48, 'width': 512, 'height': 256},
      ),
      dict(
          testcase_name='int64',
          dtype=np.dtype('int64'),
          expected_chunks={'index': 48, 'width': 256, 'height': 256},
      ),
      dict(
          testcase_name='float32',
          dtype=np.dtype('float32'),
          expected_chunks={'index': 48, 'width': 512, 'height': 256},
      ),
      dict(
          testcase_name='float64',
          dtype=np.dtype('float64'),
          expected_chunks={'index': 48, 'width': 256, 'height': 256},
      ),
      dict(
          testcase_name='complex64',
          dtype=np.dtype('complex64'),
          expected_chunks={'index': 48, 'width': 256, 'height': 256},
      ),
  )
  def test_auto_chunks__handles_standard_dtypes(self, dtype, expected_chunks):
    self.assertEqual(
        xee.EarthEngineStore._auto_chunks(dtype.itemsize),
        expected_chunks,
        '%r fails.' % dtype,
    )


class EEStoreTest(parameterized.TestCase):

  def test_auto_chunks__handles_range_of_dtype_sizes(self):
    dt = 0
    try:
      for dt in range(1, 1024):
        _ = xee.EarthEngineStore._auto_chunks(dt)
    except ValueError:
      self.fail(f'Could not handle data type size {dt}.')

  def test_auto_chunks__matches_observed_values(self):
    observed_results = {
        1: 50331648,
        2: 37748736,
        4: 31457280,
        8: 28311552,
        16: 26738688,
        32: 25952256,
        64: 25559040,
        128: 25362432,
        256: 25264128,
        512: 25214976,
    }

    for dtype_bytes, expected_bytes in observed_results.items():
      chunks = xee.EarthEngineStore._auto_chunks(dtype_bytes)
      actual_bytes = np.prod(list(chunks.values())) * (
          dtype_bytes + 1
      )  # added +1 to account for the mask byte
      self.assertEqual(
          expected_bytes,
          actual_bytes,
          f'dtype_bytes: {dtype_bytes}, Expected: {expected_bytes}, '
          f'Actual: {actual_bytes}, Chunks: {chunks}',
      )

  def test_exceeding_byte_limit__raises_error(self):
    dtype_size = 8
    # does not fail
    chunks = {'index': 48, 'width': 256, 'height': 256}
    ext._check_request_limit(chunks, dtype_size, xee.REQUEST_BYTE_LIMIT)

    # fails
    chunks = {'index': 1024, 'width': 1024, 'height': 1024}
    with self.assertRaises(ValueError):
      ext._check_request_limit(chunks, dtype_size, xee.REQUEST_BYTE_LIMIT)

  @mock.patch(
      'xee.ext.EarthEngineStore.get_info',
      new_callable=mock.PropertyMock,
  )
  def test_init_with_affine_transform(self, mock_get_info):
    """Test that an affine.Affine object can be passed for crs_transform."""
    mock_get_info.return_value = {
        'size': 1,
        'props': {},
        'first': {
            'bands': [{
                'id': 'b1',
                'data_type': {'type': 'PixelType', 'precision': 'float'}
            }]
        },
    }
    transform_tuple = (1.0, 0.0, -180.0, 0.0, -1.0, 90.0)
    transform_affine = affine.Affine(*transform_tuple)

    store = xee.EarthEngineStore(
        image_collection=mock.MagicMock(),
        crs='EPSG:4326',
        crs_transform=transform_affine,
        shape_2d=(360, 180),
    )

    self.assertIsInstance(store.crs_transform, tuple)
    self.assertEqual(store.crs_transform, transform_tuple)
    self.assertEqual(store.scale_x, 1.0)
    self.assertEqual(store.scale_y, -1.0)
    self.assertEqual(store.scale, 1.0)

  @mock.patch(
      'xee.ext.EarthEngineStore.get_info',
      new_callable=mock.PropertyMock,
  )
  def test_project(self, mock_get_info):
    """Test that the project method correctly calculates the grid."""
    mock_get_info.return_value = {
        'size': 1,
        'props': {},
        'first': {
            'bands': [{
                'id': 'b1',
                'data_type': {'type': 'PixelType', 'precision': 'float'}
            }]
        },
    }
    transform_tuple = (0.25, 0.0, -180.0, 0.0, -0.5, 90.0)
    store = xee.EarthEngineStore(
        image_collection=mock.MagicMock(),
        crs='EPSG:4326',
        crs_transform=transform_tuple,
        shape_2d=(1440, 720),
    )

    bbox = (10, 20, 30, 40)  # x_start, y_start, x_end, y_end
    grid = store.project(bbox)

    self.assertEqual(grid['dimensions']['width'], 20)
    self.assertEqual(grid['dimensions']['height'], 20)
    self.assertEqual(grid['crsCode'], 'EPSG:4326')
    # Check that the translation is correct: c + (x_start * a), f + (y_start * e)
    self.assertAlmostEqual(grid['affineTransform']['translateX'], -180.0 + (10 * 0.25))
    self.assertAlmostEqual(grid['affineTransform']['translateY'], 90.0 + (20 * -0.5))

  @mock.patch(
      'xee.ext.EarthEngineStore.get_info',
      new_callable=mock.PropertyMock,
  )
  def test_init_with_tuple_transform(self, mock_get_info):
      """Test that a tuple object can be passed for crs_transform."""
      # (Setup the mock_get_info.return_value just like in the other test)
      mock_get_info.return_value = {
          'size': 1, 'props': {},
          'first': {'bands': [{'id': 'b1', 'data_type': {'type': 'PixelType', 'precision': 'float'}}]}
      }
      transform_tuple = (1.0, 0.0, -180.0, 0.0, -1.0, 90.0)

      # Pass the tuple directly
      store = xee.EarthEngineStore(
          image_collection=mock.MagicMock(),
          crs='EPSG:4326',
          crs_transform=transform_tuple,
          shape_2d=(360, 180),
      )

      # Assert that the tuple was stored correctly
      self.assertEqual(store.crs_transform, transform_tuple)

  def test_init_with_invalid_transform_type(self):
      """Test that a TypeError is raised for invalid crs_transform types."""
      with self.assertRaises(TypeError):
          # Pass a list, which is an invalid type
          invalid_transform = [1.0, 0.0, -180.0, 0.0, -1.0, 90.0]
          xee.EarthEngineStore(
              image_collection=mock.MagicMock(),
              crs='EPSG:4326',
              crs_transform=invalid_transform,
              shape_2d=(360, 180),
          )


class ParseEEInitKwargsTest(absltest.TestCase):

  def test_parse_ee_init_kwargs__empty(self):
    self.assertDictEqual(ext._parse_ee_init_kwargs(None), {})

  def test_parse_ee_init_kwargs__credentials(self):
    self.assertDictEqual(
        ext._parse_ee_init_kwargs(
            {
                'credentials': 'foo',
                'other': 'bar',
            }
        ),
        {
            'credentials': 'foo',
            'other': 'bar',
        },
    )

  def test_parse_ee_init_kwargs__credentials_function(self):
    self.assertDictEqual(
        ext._parse_ee_init_kwargs(
            {
                'credentials_function': lambda: 'foo',
                'other': 'bar',
            }
        ),
        {
            'credentials': 'foo',
            'other': 'bar',
        },
    )

  def test_parse_ee_init_kwargs__credentials_and_credentials_function(self):
    with self.assertRaises(ValueError):
      ext._parse_ee_init_kwargs(
          {
              'credentials': 'foo',
              'credentials_function': lambda: 'foo',
              'other': 'bar',
          }
      )


class GridHelpersTest(absltest.TestCase):
  """Test grid helper functions that do not require GEE access."""

  def test_set_scale(self): 
    """Test that the scale values of the CRS transform can be updated."""
    crs_transform = [1, 0, 100, 0, 5, 200]
    scaling = (123, 456)
    crs_transform_new = helpers.set_scale(crs_transform, scaling)
    np.testing.assert_allclose(
        crs_transform_new,
        [123, 0, 100, 0, 456, 200]
    )


  def test_fit_geometry_specify_scale(self):
    """Test generating grid parameters to match a geometry, specifying the scale."""
    grid_dict = helpers.fit_geometry(
      geometry=shapely.Polygon([(10.1, 10.1),
                                (10.1, 10.9),
                                (11.9, 10.1)]),
      grid_crs='EPSG:4326',
      grid_scale=0.5
    )
    self.assertEqual(
      grid_dict['crs_transform'],
      [0.5, 0, 10, 0, -0.5, 11.0]
    )
    self.assertEqual(
      grid_dict['shape_2d'],
      (4, 2)
    )


  def test_fit_geometry_specify_scale_utm(self):
    """Test generating grid parameters to match a UTM geometry, specifying the scale."""
    grid_dict = helpers.fit_geometry(
      geometry=shapely.Polygon([(551000, 4179000),
                                (551000, 4179000),
                                (552000, 4180000),
                                (552000, 4180000)]),  # over San Francisco                       
      geometry_crs='EPSG:32610',                      
      grid_crs='EPSG:4326',
      grid_scale=0.01
    )
    self.assertEqual(
      grid_dict['crs_transform'],
      [0.01, 0.0, -122.43, 0.0, -0.01, 37.77]
    )
    self.assertEqual(
      grid_dict['shape_2d'],
      (3, 2)
    )


  def test_fit_geometry_specify_shape(self):
    """Test generating grid parameters to match a geometry, specifying the shape."""
    grid_dict = helpers.fit_geometry(
      geometry=shapely.Polygon([(10.0, 2.0),
                                (10.0, 3.0),
                                (12.0, 2.0)]),
      grid_crs='EPSG:4326',
      grid_shape=(4, 2)
    )
    np.testing.assert_allclose(
      grid_dict['crs_transform'],
      [0.5, 0, 10, 0, -0.5, 3],
      rtol=1e-4,
    )


if __name__ == '__main__':
  absltest.main()
