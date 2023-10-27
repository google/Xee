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
r"""Exports EE ImageCollections to Zarr using Xarray-Beam."""

import logging

from absl import app
from absl import flags
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam
import xee

import ee


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_INPUT = flags.DEFINE_string(
    'input', '', help='The input Earth Engine ImageCollection.'
)
_CRS = flags.DEFINE_string(
    'crs',
    'EPSG:4326',
    help='Coordinate Reference System for output Zarr.',
)
_SCALE = flags.DEFINE_float('scale', 0.25, help='Scale factor for output Zarr.')
_TARGET_CHUNKS = flags.DEFINE_string(
    'target_chunks',
    '',
    help=(
        'chunks on the input Zarr dataset to change on the outputs, in the '
        'form of a comma separated dimension=size pairs, e.g., '
        "--target_chunks='x=10,y=10'. Omitted dimensions are not changed and a "
        'chunksize of -1 indicates not to chunk a dimension.'
    ),
)
_OUTPUT = flags.DEFINE_string('output', '', help='The output zarr path.')
_RUNNER = flags.DEFINE_string('runner', None, help='beam.runners.Runner')


# Borrowed from the xbeam examples:
# https://github.com/google/xarray-beam/blob/4f4fcb965a65b5d577601af311d0e0142ee38076/examples/xbeam_rechunk.py#L41
def _parse_chunks_str(chunks_str: str) -> dict[str, int]:
  chunks = {}
  parts = chunks_str.split(',')
  for part in parts:
    k, v = part.split('=')
    chunks[k] = int(v)
  return chunks


def main(argv: list[str]) -> None:
  assert _INPUT.value, 'Must specify --input'
  assert _OUTPUT.value, 'Must specify --output'

  source_chunks = {'time': 24}
  target_chunks = dict(source_chunks, **_parse_chunks_str(_TARGET_CHUNKS.value))

  ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

  ds = xr.open_dataset(
      _INPUT.value,
      crs=_CRS.value,
      scale=_SCALE.value,
      engine=xee.EarthEngineBackendEntrypoint,
  )
  template = xbeam.make_template(ds)
  itemsize = max(variable.dtype.itemsize for variable in template.values())

  with beam.Pipeline(runner=_RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(ds, source_chunks)
        | xbeam.Rechunk(
            ds.sizes,
            source_chunks,
            target_chunks,
            itemsize=itemsize,
        )
        | xbeam.ChunksToZarr(_OUTPUT.value, template, target_chunks)
    )


if __name__ == '__main__':
  app.run(main)
