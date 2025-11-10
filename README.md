# Xee: Xarray + Google Earth Engine

![Xee Logo](https://raw.githubusercontent.com/google/Xee/main/docs/xee-logo.png)

_An Xarray extension for Google Earth Engine._

Xee bridges the gap between Google Earth Engine's massive data catalog and the scientific Python ecosystem. It provides a custom Xarray backend that allows you to open any `ee.ImageCollection` as if it were a local `xarray.Dataset`. Data is loaded lazily and in parallel, enabling you to work with petabyte-scale archives of satellite and climate data using the power and flexibility of Xarray and its integrations with libraries like Dask.

[![image](https://img.shields.io/pypi/v/xee.svg)](https://pypi.python.org/pypi/xee)
[![image](https://static.pepy.tech/badge/xee)](https://pepy.tech/project/xee)
[![Conda
Recipe](https://img.shields.io/badge/recipe-xee-green.svg)](https://github.com/conda-forge/xee-feedstock)
[![image](https://img.shields.io/conda/vn/conda-forge/xee.svg)](https://anaconda.org/conda-forge/xee)
[![Conda
Downloads](https://img.shields.io/conda/dn/conda-forge/xee.svg)](https://anaconda.org/conda-forge/xee)

## How to use

Install with pip:

```shell
pip install --upgrade xee
```

Install with conda:

```shell
conda install -c conda-forge xee
```

Then, authenticate Earth Engine:

```shell
earthengine authenticate --quiet
```

Now, in your Python environment, make the following imports and initialize the Earth Engine client with your project ID. Using the high-volume API endpoint is recommended.

```python
import ee
import xarray as xr
from xee import helpers
import shapely

ee.Initialize(
    project='PROJECT-ID',  # Replace with your project ID
    opt_url='https://earthengine-highvolume.googleapis.com'
)
```

### Specifying the Output Grid

To open a dataset, you must specify the desired output pixel grid. The `xee.helpers` module simplifies this process by providing several convenient workflows, summarized below.

| Goal | Method | When to Use |
| :--- | :--- | :--- |
| **Match Source Grid** | Use `helpers.extract_grid_params()` to get the parameters from an EE object. | When you want the data in its original, default projection and scale. |
| **Fit Area to a Shape** | Use `helpers.fit_geometry()` with the `geometry` and `grid_shape` arguments. | When you need a consistent output array size (e.g., for ML models) and the exact pixel size is less important. |
| **Fit Area to a Scale** | Use `helpers.fit_geometry()` with the `geometry` and `grid_scale` arguments. | When the specific resolution (e.g., 30 meters, 0.01 degrees) is critical for your analysis. |
| **Manual Override** | Pass `crs`, `crs_transform`, and `shape_2d` directly to `xr.open_dataset`. | For advanced cases where you already have an exact grid definition. |

> **Important Note on Units:** All grid parameter values must be in the units of the specified Coordinate Reference System (`crs`).
> * For a geographic CRS like `'EPSG:4326'`, the units are in **degrees**.
> * For a projected CRS like `'EPSG:32610'` (UTM), the units are in **meters**.
> This applies to the translation values in `crs_transform` and the pixel sizes in `grid_scale`.

### Usage Examples

Here are common workflows for opening datasets with `xee`, corresponding to the methods in the table above.

#### Match Source Grid

This is the simplest case, using `helpers.extract_grid_params` to match the dataset's default grid.

```python
ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
grid_params = helpers.extract_grid_params(ic)
ds = xr.open_dataset(ic, engine='ee', **grid_params)
```

#### Fit Area to a Shape

Define a grid over an area of interest by specifying the number of pixels. `helpers.fit_geometry` will calculate the correct `crs_transform`.

```python
aoi = shapely.geometry.box(113.33, -43.63, 153.56, -10.66) # Australia
grid_params = helpers.fit_geometry(
    geometry=aoi,
    grid_crs='EPSG:4326',
    grid_shape=(256, 256)
)

ds = xr.open_dataset('ee://ECMWF/ERA5_LAND/MONTHLY_AGGR', engine='ee', **grid_params)
```

#### Fit Area to a Scale (Resolution)

> **A Note on `grid_scale` and Y-Scale Orientation**
> When using `fit_geometry` with `grid_scale`, you are defining both the pixel size and the grid's orientation via the sign of the y-scale.
> * A **negative `y_scale`** (e.g., `(10000, -10000)`) is the standard for "north-up" satellite and aerial imagery, creating a grid with a **top-left** origin.
> * A **positive `y_scale`** (e.g., `(10000, 10000)`) is used by some datasets and creates a grid with a **bottom-left** origin.
> You may need to inspect your source dataset's projection information to determine the correct sign to use. If you use `grid_shape`, a standard negative y-scale is assumed.

The following example defines a grid over an area by specifying the pixel size in meters. `fit_geometry` will reproject the geometry and calculate the correct `shape_2d`.

```python
aoi = shapely.geometry.box(113.33, -43.63, 153.56, -10.66) # Australia
grid_params = helpers.fit_geometry(
    geometry=aoi,
    geometry_crs='EPSG:4326',       # CRS of the input geometry
    grid_crs='EPSG:32662',          # Target CRS in meters (Plate Carrée)
    grid_scale=(10000, -10000)      # Define a 10km pixel size
)

ds = xr.open_dataset('ee://ECMWF/ERA5_LAND/MONTHLY_AGGR', engine='ee', **grid_params)
```

#### Open a Custom Region at Source Resolution

This workflow is ideal for analyzing a specific area while maintaining the dataset's original resolution.

```python
# 1. Get the original grid parameters from the target ImageCollection
ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
source_params = helpers.extract_grid_params(ic)

# 2. Extract the source CRS and scale
source_crs = source_params['crs']
source_transform = source_params['crs_transform']
source_scale = (source_transform[0], source_transform[4]) # (x_scale, y_scale)

# 3. Use the source parameters to fit the grid to a specific geometry
aoi = shapely.geometry.box(113.33, -43.63, 153.56, -10.66) # Australia
final_grid_params = helpers.fit_geometry(
    geometry=aoi,
    geometry_crs='EPSG:4326',
    grid_crs=source_crs,
    grid_scale=source_scale
)

# 4. Open the dataset with the final, combined parameters
ds = xr.open_dataset(ic, engine='ee', **final_grid_params)
```

#### Manual Override

For use cases where you know the exact grid parameters, you can provide them directly.

```python
# Manually define a 512x512 pixel grid with 1-degree pixels in EPSG:4326
manual_crs = 'EPSG:4326'
manual_transform = (0.1, 0, -180.05, 0, -0.1, 90.05) # Values are in degrees
manual_shape = (512, 512)

ds = xr.open_dataset(
    'ee://ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    crs=manual_crs,
    crs_transform=manual_transform,
    shape_2d=manual_shape,
)
```

#### Open a Pre-Processed ImageCollection

A key feature of Xee is its ability to open a computed `ee.ImageCollection`. This allows you to leverage Earth Engine's powerful server-side processing for tasks like filtering, band selection, and calculations before loading the data into Xarray.

```python
# Define an AOI as a shapely object for the helper function
sf_aoi_shapely = shapely.geometry.Point(-122.4, 37.7).buffer(0.2)
# Create an ee.Geometry from the shapely object for server-side filtering
coords = list(sf_aoi_shapely.exterior.coords)
sf_aoi_ee = ee.Geometry.Polygon(coords)

# Define a function to calculate NDVI and add it as a band
def add_ndvi(image):
    # Landsat 9 SR bands: NIR = B5, Red = B4
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    return image.addBands(ndvi)

# Build the pre-processed collection
processed_collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate('2024-06-01', '2024-09-01')
    .filterBounds(sf_aoi_ee)
    .map(add_ndvi)
    .select(['NDVI']))

# Define the output grid using a helper
grid_params = helpers.fit_geometry(
    geometry=sf_aoi_shapely,
    grid_crs='EPSG:32610',     # Target CRS in meters (UTM Zone 10N)
    grid_scale=(30, -30)        # Use Landsat's 30m resolution
)

# Open the fully processed collection
ds = xr.open_dataset(processed_collection, engine='ee', **grid_params)
```

#### Open a single Image

The `helpers` work the same way for a single `ee.Image`.

```python
img = ee.Image('ECMWF/ERA5_LAND/MONTHLY_AGGR/202501')
grid_params = helpers.extract_grid_params(img)
ds = xr.open_dataset(img, engine='ee', **grid_params)
```

#### Visualize a Single Time Slice

Once you have your `xarray.Dataset`, you can visualize a single time slice of a variable to verify the results. This requires the `matplotlib` library, which is an optional dependency.

If you don't have it installed, you can add it with pip:

```shell
pip install matplotlib
```

Xarray's plotting functions work directly with the data, which now follows the standard `(y, x)` dimension ordering convention used by NetCDF-CF and matplotlib.

```python

# First, open a dataset using one of the methods above
aoi = shapely.geometry.box(113.33, -43.63, 153.56, -10.66) # Australia
grid_params = helpers.fit_geometry(
    geometry=aoi,
    grid_crs='EPSG:4326',
    grid_shape=(256, 256)
)
ds = xr.open_dataset('ECMWF/ERA5_LAND/MONTHLY_AGGR', engine='ee', **grid_params)

# Select the 2m air temperature for the first time step
temp_slice = ds['temperature_2m'].isel(time=0)

# Plot directly - no transpose needed!
temp_slice.plot()
```

See [examples](https://github.com/google/Xee/tree/main/examples) or
[docs](https://github.com/google/Xee/tree/main/docs) for more uses and
integrations.

## Getting help

If you encounter issues using Xee, you can:

1. Open a new or add to an existing [Xee discussion
   topic](https://github.com/google/Xee/discussions)
2. Open an [Xee issue](https://github.com/google/Xee/issues). To increase the
   likelihood of the issue being resolved, use this [template Colab
   notebook](https://colab.research.google.com/drive/1vAgfAPhKGJd4G9ZUOzciqZ7MbqJjlMLR)
   to create a reproducible script.

## How to run integration tests

The Xee integration tests only pass on Xee branches (no forks). Please run the
integration tests locally before sending a PR. To run the tests locally,
authenticate using `earthengine authenticate` and run the following:

```bash
python -m unittest xee/ext_integration_test.py
```

or

```bash
python -m pytest xee/ext_integration_test.py
```

## License

This is not an official Google product.

```
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
