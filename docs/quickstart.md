---
title: Quickstart
---

# Quickstart

Get up and running with Xee in a few minutes.

## 1. Install

Use pip (or conda):

```bash
pip install --upgrade xee
```

```bash
conda install -c conda-forge xee
```

Optional (plotting): `pip install matplotlib`.

## 2. Earth Engine access

You need an Earth Engine–enabled Google Cloud project. If you haven't done this yet, follow the Earth Engine [Access guide](https://developers.google.com/earth-engine/guides/access#get_access_to_earth_engine).

Authenticate once on a persistent machine:

```bash
earthengine authenticate
```

Or inside ephemeral environments (e.g. Colab):

```python
import ee
ee.Authenticate()
```

Initialize (high‑volume endpoint recommended for reading stored ImageCollections):

```python
import ee
ee.Initialize(
    project='YOUR-PROJECT-ID',
    opt_url='https://earthengine-highvolume.googleapis.com'
)
```

For computed collections (server-side expressions) you can omit `opt_url` to use the standard endpoint which benefits from caching.

## 3. Open your first dataset

```python
import ee, xarray as xr
from xee import helpers

ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
grid = helpers.extract_grid_params(ic)  # match source projection & resolution
ds = xr.open_dataset(ic, engine='ee', **grid)
print(ds)
```

Plot the first time slice (matplotlib required):

```python
ds['temperature_2m'].isel(time=0).plot()
```

## 4. Next steps

| Goal | Where to go |
|------|-------------|
| Learn grid parameter patterns | [Concepts](concepts.md) |
| Fit a custom area or scale | [User Guide](guide.md) |
| API signatures | [API Reference](api.md) |
| Migrate 0.0.x code | [Migration Guide](migration-guide-v0.1.0.md) |
| Performance tips | [Performance & Limits](performance.md) |
| Troubleshooting common issues | [FAQ](faq.md) |

## 5. Minimal workflow recap

1. Install Xee & authenticate EE
2. Initialize EE client
3. Derive grid parameters (match source or fit a geometry)
4. Call `xr.open_dataset(..., engine='ee', **grid)`
5. Use Xarray normally (select, compute, visualize, export)

## 6. Example: custom AOI at fixed size

```python
import shapely
from xee import helpers

aoi = shapely.geometry.box(113.33, -43.63, 153.56, -10.66)  # Australia
grid = helpers.fit_geometry(
    geometry=aoi,
    grid_crs='EPSG:4326',   # degrees
    grid_shape=(256, 256)   # (width, height) pixels
)

ds = xr.open_dataset('ee://ECMWF/ERA5_LAND/MONTHLY_AGGR', engine='ee', **grid)
```

## 7. Having trouble?

See the [FAQ](faq.md) and open a [discussion](https://github.com/google/Xee/discussions) if needed.
