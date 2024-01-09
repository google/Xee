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
          expected_chunks={'index': 48, 'width': 1024, 'height': 1024},
      ),
      dict(
          testcase_name='int32',
          dtype=np.dtype('int32'),
          expected_chunks={'index': 48, 'width': 512, 'height': 512},
      ),
      dict(
          testcase_name='int64',
          dtype=np.dtype('int64'),
          expected_chunks={'index': 48, 'width': 512, 'height': 256},
      ),
      dict(
          testcase_name='float32',
          dtype=np.dtype('float32'),
          expected_chunks={'index': 48, 'width': 512, 'height': 512},
      ),
      dict(
          testcase_name='float64',
          dtype=np.dtype('float64'),
          expected_chunks={'index': 48, 'width': 512, 'height': 256},
      ),
      dict(
          testcase_name='complex64',
          dtype=np.dtype('complex64'),
          expected_chunks={'index': 48, 'width': 512, 'height': 256},
      ),
  )
  def test_auto_chunks__handles_standard_dtypes(self, dtype, expected_chunks):
    self.assertEqual(
        xee.EarthEngineStore._auto_chunks(dtype.itemsize),
        expected_chunks,
        '%r fails.' % dtype,
    )


class EEStoreTest(absltest.TestCase):

  def test_auto_chunks__handles_range_of_dtype_sizes(self):
    dt = 0
    try:
      for dt in range(1, 1024):
        _ = xee.EarthEngineStore._auto_chunks(dt)
    except ValueError:
      self.fail(f'Could not handle data type size {dt}.')

  def test_auto_chunks__is_optimal_for_powers_of_two(self):
    for p in range(10):
      dt = 2**p
      chunks = xee.EarthEngineStore._auto_chunks(dt)
      self.assertEqual(
          xee.REQUEST_BYTE_LIMIT, np.prod(list(chunks.values())) * dt
      )

  def test_exceeding_byte_limit__raises_error(self):
    dtype_size = 8
    # does not fail
    chunks = {'index': 48, 'width': 512, 'height': 256}
    ext._check_request_limit(chunks, dtype_size, xee.REQUEST_BYTE_LIMIT)

    # fails
    chunks = {'index': 1024, 'width': 1024, 'height': 1024}
    with self.assertRaises(ValueError):
      ext._check_request_limit(chunks, dtype_size, xee.REQUEST_BYTE_LIMIT)


if __name__ == '__main__':
  absltest.main()
