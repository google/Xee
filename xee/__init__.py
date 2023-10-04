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
"""A Google Earth Engine backend for Xarray.

Warning: Experimental! Use at your own risk.

Supported today (2023-06-13):
- Pixel Space Chunking: Will split up large pixel requests into smaller chunks
  to get around EE `computePixels()` byte limits
- User-defined projections: Users can specify a CRS and Scale when opening a
  dataset. All bands should appear in a specific projection space.
- Index Chunking: Users can open all images in the collection (metadata lookup
  may be slow) or specify `n_images` to open at at time.

Needs to be done:
- Full Xarray API support: There are features like cf encoding that are standard
  in Xarray that have been put off till later.
- Performance Tuning (with micro benchmarks): We need to methodically optimize
  the numpy, EE client, and parallelism bits of this client.
- Robustness testing: Again, this is experimental. Beware of sharp edges!

Contributions are welcome! Before committing your change, please check if there
is an existing Github issue.
"""
import importlib
import sys

from .ext import *


assert sys.version_info >= (3, 8)
__version__ = importlib.metadata.version('xee') or 'unknown'
