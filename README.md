> **⚠️ Breaking Change in v0.1.0**
>
> A major refactor was released in v0.1.0, introducing breaking changes to the Xee API. In most cases, existing code written for pre-v0.1.0 versions will require updates to remain compatible.
>
> - See the [Migration Guide](https://github.com/google/Xee/blob/main/docs/migration-guide-v0.1.0.md) for details on updating your code.
> - If you need more time to migrate, you can pin your environment to the latest pre-v0.1.0 release.
>
> **During the v0.1.0 prerelease window:** `pip install xee` and `conda install xee` may still install the previous stable line.
> To use the refactored API documented here, install a prerelease with `pip install --upgrade --pre xee` or pin an RC such as `pip install xee==0.1.0rc1`.

# Xee: Xarray + Google Earth Engine

![Xee Logo](https://raw.githubusercontent.com/google/Xee/main/docs/xee-logo.png)

Xee is an Xarray backend for Google Earth Engine. Open `ee.Image` / `ee.ImageCollection` objects as lazy `xarray.Dataset`s and analyze petabyte‑scale Earth data with the scientific Python stack.

[![image](https://img.shields.io/pypi/v/xee.svg)](https://pypi.python.org/pypi/xee)
[![image](https://static.pepy.tech/badge/xee)](https://pepy.tech/project/xee)
[![Conda Recipe](https://img.shields.io/badge/recipe-xee-green.svg)](https://github.com/conda-forge/xee-feedstock)
[![image](https://img.shields.io/conda/vn/conda-forge/xee.svg)](https://anaconda.org/conda-forge/xee)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/xee.svg)](https://anaconda.org/conda-forge/xee)

## Install

For the refactored v0.1.0 API documented below (prerelease period):

```bash
pip install --upgrade --pre xee
```

or pin a specific release candidate:

```bash
pip install xee==0.1.0rc1
```

For the current stable line (pre-v0.1.0 API):

```bash
pip install --upgrade xee
```

or

```bash
conda install -c conda-forge xee
```

Note: conda-forge may lag PyPI during prerelease testing. Use pip for the latest RC builds.

## Minimal example

```python
import ee
import xarray as xr
from xee import helpers

# Authenticate once (on a persistent machine):
#   earthengine authenticate

project = 'PROJECT-ID'  # Set your Earth Engine registered Google Cloud project ID
# Initialize (high‑volume endpoint recommended for reading stored collections)
ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')

# Open a dataset by matching its native grid
ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
grid = helpers.extract_grid_params(ic)
ds = xr.open_dataset(ic, engine='ee', **grid)
print(ds)
```

Next steps:

- [Quickstart](https://github.com/google/Xee/blob/main/docs/quickstart.md)
- [Concepts (grid params, CRS, orientation)](https://github.com/google/Xee/blob/main/docs/concepts.md)
- [User Guide (workflows)](https://github.com/google/Xee/blob/main/docs/guide.md)

## Features

- Lazy, parallel pixel retrieval through Earth Engine
- Flexible output grid definition (fixed resolution or fixed shape)
- CF-friendly dimension order: `[time, y, x]`
- Plays nicely with Xarray, Dask, and friends

## Community & Support

- [Discussions](https://github.com/google/Xee/discussions)
- [Issues](https://github.com/google/Xee/issues)

## Contributing

See [Contributing](https://github.com/google/Xee/blob/main/docs/contributing.md) and sign the required CLA. For local development, we recommend the Pixi environments defined in this repository for reproducible test and docs runs.

## License

[Apache 2.0](https://github.com/google/Xee/blob/main/LICENSE)

`SPDX-License-Identifier: Apache-2.0`

This is not an official Google product.

