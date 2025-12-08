# User Guide

This guide collects practical workflows. For underlying theory see [Core Concepts](concepts.md). For a minimal setup see the [Quickstart](quickstart.md).

## Match Source Grid

Use `helpers.extract_grid_params` to mirror the dataset's native projection & resolution.

```python
import ee, xarray as xr
from xee import helpers

ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
grid_params = helpers.extract_grid_params(ic)
ds = xr.open_dataset(ic, engine='ee', **grid_params)
```

## Fit Area to a Shape

Derive a grid that covers an AOI with a fixed pixel count (resolution floats).

```python
import shapely
from xee import helpers

aoi = shapely.geometry.box(113.33, -43.63, 153.56, -10.66) # Australia
grid_params = helpers.fit_geometry(
    geometry=aoi,
    grid_crs='EPSG:4326',
    grid_shape=(256, 256)
)

ds = xr.open_dataset('ee://ECMWF/ERA5_LAND/MONTHLY_AGGR', engine='ee', **grid_params)
```

## Fit Area to a Scale (Resolution)

Fix physical pixel size; grid dimensions derived from AOI extent.

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

## Custom Region at Source Resolution

Fit an AOI but keep original pixel size.

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

## Manual Override

Direct specification for reproducibility / alignment with external rasters.

```python
# Manually define a 512x512 pixel grid with 1-degree pixels in EPSG:4326
manual_crs = 'EPSG:4326'
manual_transform = (0.1, 0, -180.05, 0, -0.1, 90.05) # Values are in degrees
manual_shape = (512, 512)

ds = xr.open_dataset('ee://ECMWF/ERA5_LAND/MONTHLY_AGGR', engine='ee', crs=manual_crs, crs_transform=manual_transform, shape_2d=manual_shape)
```

## Pre-Processed ImageCollection

Apply server-side operations before opening for efficiency.

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
    grid_scale=(30, -30)       # Use Landsat's 30m resolution
)

# Open the fully processed collection
ds = xr.open_dataset(processed_collection, engine='ee', **grid_params)
```

## Single Image

```python
img = ee.Image('ECMWF/ERA5_LAND/MONTHLY_AGGR/202501')
grid_params = helpers.extract_grid_params(img)
ds = xr.open_dataset(img, engine='ee', **grid_params)
```

## Visualize a Time Slice

Requires `matplotlib` (`pip install matplotlib`).

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

# Plot the data
temp_slice.plot()
```

## Further Resources

- [Core Concepts](concepts.md)
- [Performance & Limits](performance.md)
- [FAQ](faq.md)
- Examples: see [examples](https://github.com/google/Xee/tree/main/examples) directory in the repository
