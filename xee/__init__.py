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
"""Public Xee API.

End users typically:

1. Define pixel grid parameters using helper functions like :func:`fit_geometry`
        or :func:`extract_grid_params`.
2. Call :func:`xarray.open_dataset` with ``engine='ee'`` and the returned
        ``grid_params``.

The backend classes are exposed for advanced or library integration use, but
most workflows only need the helpers and the xarray interface.
"""

from .ext import *  # noqa: F401,F403  (backend classes)
from .ext import __version__  # noqa: F401
from .helpers import fit_geometry, extract_grid_params, set_scale, PixelGridParams  # noqa: F401

__all__ = [
    # Version.
    '__version__',
    # Helper functions.
    'fit_geometry',
    'extract_grid_params',
    'set_scale',
    'PixelGridParams',
    # Selected backend surface (avoid * pollution for autosummary ordering).
    'EarthEngineBackendEntrypoint',
    'EarthEngineStore',
    'EarthEngineBackendArray',
]
