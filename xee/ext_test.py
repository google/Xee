"""Xee Unit Tests."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import xee
from xee import ext


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


if __name__ == '__main__':
  absltest.main()
