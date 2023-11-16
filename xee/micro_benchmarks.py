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
r"""Manual benchmarks for the Google Earth Engine backend for Xarray.

These are intended to always be run manually since they are more expensive test
to run.
"""

import cProfile
import os
import tempfile
import timeit
from typing import List

from absl import app
import numpy as np
import xarray
import xee

import ee


REPEAT = 10
LOOPS = 1
PROFILE = False


def init_ee_for_tests():
  ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def open_dataset() -> None:
  _ = xarray.open_dataset(
      'NASA/GPM_L3/IMERG_V06', engine=xee.EarthEngineBackendEntrypoint
  )


def open_and_chunk() -> None:
  ds = xarray.open_dataset(
      'NASA/GPM_L3/IMERG_V06',
      crs='EPSG:4326',
      scale=0.25,
      chunks={'index': 24, 'width': 512, 'height': 512},
      engine=xee.EarthEngineBackendEntrypoint,
  )
  ds.chunk()


def open_and_write() -> None:
  with tempfile.TemporaryDirectory() as tmpdir:
    ds = xarray.open_dataset(
        'NASA/GPM_L3/IMERG_V06',
        crs='EPSG:4326',
        scale=0.25,
        chunks={'time': 24, 'lon': 1440, 'lat': 720},
        engine=xee.EarthEngineBackendEntrypoint,
    )
    ds = ds.isel(time=slice(0, 24))
    ds.to_zarr(os.path.join(tmpdir, 'imerg.zarr'))


def main(_: List[str]) -> None:
  print('Initializing EE...')
  init_ee_for_tests()
  print(f'[{REPEAT} time(s) with {LOOPS} loop(s) each.]')
  for fn in ['open_dataset()', 'open_and_chunk()', 'open_and_write()']:
    if PROFILE:
      cProfile.run(fn)
    timer = timeit.Timer(fn, globals=globals())
    res = timer.repeat(REPEAT, number=LOOPS)
    avg, std, best, worst = np.mean(res), np.std(res), np.min(res), np.max(res)
    print(f'{fn}:avg={avg:.2f},std={std:.2f},best={best:.2f},worst={worst:.2f}')


if __name__ == '__main__':
  app.run(main)
