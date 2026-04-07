# Migration Guide: Xee v0.1.0

This guide helps you update your code from Xee v0.0.x to v0.1.0. The 0.1 release includes two major improvements:

1. **New grid parameter system** for specifying output geography
2. **Updated dimension ordering** from `[time, x, y]` to `[time, y, x]`

## Quick Migration Checklist

- [ ] Replace `scale` and `geometry` parameters with grid parameter helpers
- [ ] Remove `.transpose()` calls before plotting or passing to libraries expecting `[time, y, x]`
- [ ] Update any code that explicitly references dimension order
- [ ] Test your workflows with the new API

## 1. Geography Specification Changes

### Old API (v0.0.x)

The old API used simple `crs`, `scale`, and `geometry` parameters:

```python
import ee
import xarray as xr

ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    crs='EPSG:4326',
    scale=0.25,  # pixel size in degrees
    geometry=ee.Geometry.Rectangle([-180, -90, 180, 90])
)
```

### New API (v0.1.0)

The new API requires explicit grid parameters: `crs`, `crs_transform`, and `shape_2d`. We provide helper functions to make this easy:

#### Option 1: Match Source Grid (Recommended for simplicity)

```python
import ee
import xarray as xr
from xee import helpers

ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
grid_params = helpers.extract_grid_params(ic)
ds = xr.open_dataset(ic, engine='ee', **grid_params)
```

#### Option 2: Fit Geometry with Specific Scale

```python
import ee
import xarray as xr
from xee import helpers
import shapely

# Define your area of interest using shapely
aoi = shapely.geometry.box(-180, -90, 180, 90)  # Global extent

grid_params = helpers.fit_geometry(
    geometry=aoi,
    grid_crs='EPSG:4326',
    grid_scale=(0.25, -0.25)  # (x_scale, y_scale) in degrees
)

ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    **grid_params
)
```

#### Option 3: Fit Geometry with Specific Shape

```python
import shapely
from xee import helpers

aoi = shapely.geometry.box(113.33, -43.63, 153.56, -10.66)  # Australia

grid_params = helpers.fit_geometry(
    geometry=aoi,
    grid_crs='EPSG:4326',
    grid_shape=(256, 256)  # (width, height) in pixels
)

ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    **grid_params
)
```

### Migration Examples

#### Example 1: Global dataset at fixed scale

**Before (v0.1.0):**
```python
ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    crs='EPSG:4326',
    scale=1.0,
    geometry=ee.Geometry.Rectangle([-180, -90, 180, 90])
)
```

**After (v0.1.0):**
```python
import shapely
from xee import helpers

global_geom = shapely.geometry.box(-180, -90, 180, 90)
grid_params = helpers.fit_geometry(
    geometry=global_geom,
    grid_crs='EPSG:4326',
    grid_scale=(1.0, -1.0)  # Note: negative y-scale for north-up orientation
)

ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    **grid_params
)
```

#### Example 2: Regional dataset with EE geometry

**Before (v0.1.0):**
```python
import ee

aoi = ee.Geometry.Rectangle([-122.5, 37.0, -121.5, 38.0])
ds = xr.open_dataset(
    'LANDSAT/LC09/C02/T1_L2',
    engine='ee',
    crs='EPSG:32610',
    scale=30,
    geometry=aoi
)
```

**After (v0.1.0):**
```python
import ee
import shapely
from xee import helpers

# Convert EE geometry to shapely (or create directly with shapely)
aoi = shapely.geometry.box(-122.5, 37.0, -121.5, 38.0)

grid_params = helpers.fit_geometry(
    geometry=aoi,
    geometry_crs='EPSG:4326',  # Input geometry CRS
    grid_crs='EPSG:32610',     # Output grid CRS (UTM Zone 10N)
    grid_scale=(30, -30)       # 30m resolution
)

ds = xr.open_dataset(
    'LANDSAT/LC09/C02/T1_L2',
    engine='ee',
    **grid_params
)
```

#### Example 3: Using source resolution for a custom area

**Before (v0.1.0):**
```python
# You had to manually determine the scale from the dataset
ds = xr.open_dataset(
    collection,
    engine='ee',
    crs='EPSG:4326',
    scale=0.25,  # Manually determined
    geometry=my_region
)
```

**After (v0.1.0):**
```python
from xee import helpers
import shapely

# 1. Extract source grid parameters
ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
source_params = helpers.extract_grid_params(ic)

# 2. Get the source scale
source_crs = source_params['crs']
source_transform = source_params['crs_transform']
source_scale = (source_transform[0], source_transform[4])

# 3. Apply to your custom region
my_region = shapely.geometry.box(-10, 35, 5, 50)  # Western Europe
grid_params = helpers.fit_geometry(
    geometry=my_region,
    geometry_crs='EPSG:4326',
    grid_crs=source_crs,
    grid_scale=source_scale
)

ds = xr.open_dataset(ic, engine='ee', **grid_params)
```

## 2. Dimension Ordering Changes

### What Changed

Xee v0.1.0 outputs dimensions in `[time, y, x]` order (matching CF conventions and most geospatial tools), instead of the previous `[time, x, y]` order.

### Impact on Your Code

#### Plotting

**Before (v0.1.0):**
```python
# Required transpose for correct visualization
ds['temperature_2m'].isel(time=0).transpose().plot()
```

**After (v0.1.0):**
```python
# No transpose needed - plots correctly by default
ds['temperature_2m'].isel(time=0).plot()
```

#### Integration with Other Libraries

Many geospatial libraries expect `[time, y, x]` ordering. You may have been using `.transpose()` to accommodate this.

**Before (v0.1.0):**
```python
# Had to transpose for libraries expecting [time, y, x]
data_array = ds['temperature_2m'].transpose('time', 'y', 'x')
export_to_geotiff(data_array)
```

**After (v0.1.0):**
```python
# Dimension order is already correct
data_array = ds['temperature_2m']
export_to_geotiff(data_array)
```

#### Explicit Dimension Access

If you have code that explicitly references dimension positions, update it:

**Before (v0.1.0):**
```python
# Dimensions were [time, x, y]
time_dim, x_dim, y_dim = ds['temperature_2m'].dims
# or
width = ds['temperature_2m'].shape[1]   # x dimension
height = ds['temperature_2m'].shape[2]  # y dimension
```

**After (v0.1.0):**
```python
# Dimensions are now [time, y, x]
time_dim, y_dim, x_dim = ds['temperature_2m'].dims
# or
height = ds['temperature_2m'].shape[1]  # y dimension
width = ds['temperature_2m'].shape[2]   # x dimension
```

**Better approach (dimension-agnostic):**
```python
# This works in both versions
dims = ds['temperature_2m'].dims
width = ds.sizes['x']
height = ds.sizes['y']
time_length = ds.sizes['time']
```

## 3. Common Migration Patterns

### Pattern 1: Simple global analysis

**Before (v0.1.0):**
```python
import ee
import xarray as xr

ee.Initialize()
ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    crs='EPSG:4326',
    scale=1.0,
    geometry=ee.Geometry.Rectangle([-180, -90, 180, 90])
)
mean_temp = ds['temperature_2m'].mean(dim='time')
mean_temp.transpose().plot()
```

**After (v0.1.0):**
```python
import ee
import xarray as xr
from xee import helpers
import shapely

ee.Initialize()
global_geom = shapely.geometry.box(-180, -90, 180, 90)
grid_params = helpers.fit_geometry(
    geometry=global_geom,
    grid_crs='EPSG:4326',
    grid_scale=(1.0, -1.0)
)

ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    **grid_params
)
mean_temp = ds['temperature_2m'].mean(dim='time')
mean_temp.plot()  # No transpose needed
```

### Pattern 2: Regional analysis with preprocessing

**Before (v0.1.0):**
```python
import ee
import xarray as xr

def add_ndvi(image):
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    return image.addBands(ndvi)

aoi = ee.Geometry.Rectangle([-122.5, 37.5, -122.0, 38.0])
collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate('2024-01-01', '2024-12-31')
    .filterBounds(aoi)
    .map(add_ndvi)
    .select(['NDVI']))

ds = xr.open_dataset(
    collection,
    engine='ee',
    crs='EPSG:32610',
    scale=30,
    geometry=aoi
)
```

**After (v0.1.0):**
```python
import ee
import xarray as xr
from xee import helpers
import shapely

def add_ndvi(image):
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    return image.addBands(ndvi)

# Use shapely for geometry
aoi_shapely = shapely.geometry.box(-122.5, 37.5, -122.0, 38.0)

# Create ee.Geometry for server-side filtering
coords = list(aoi_shapely.exterior.coords)
aoi_ee = ee.Geometry.Polygon(coords)

collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate('2024-01-01', '2024-12-31')
    .filterBounds(aoi_ee)
    .map(add_ndvi)
    .select(['NDVI']))

grid_params = helpers.fit_geometry(
    geometry=aoi_shapely,
    geometry_crs='EPSG:4326',
    grid_crs='EPSG:32610',
    grid_scale=(30, -30)
)

ds = xr.open_dataset(collection, engine='ee', **grid_params)
```

### Pattern 3: Export workflows

**Before (v0.1.0):**
```python
import xarray as xr

ds = xr.open_dataset(
    collection,
    engine='ee',
    crs='EPSG:4326',
    scale=0.1,
    geometry=region
)

# Transpose for proper export
data = ds['variable'].transpose('time', 'y', 'x')
data.to_netcdf('output.nc')
```

**After (v0.1.0):**
```python
import xarray as xr
from xee import helpers

grid_params = helpers.fit_geometry(
    geometry=region,
    grid_crs='EPSG:4326',
    grid_scale=(0.1, -0.1)
)

ds = xr.open_dataset(collection, engine='ee', **grid_params)

# Already in correct dimension order
data = ds['variable']
data.to_netcdf('output.nc')
```

## 4. Understanding Grid Parameters

### The Three Required Parameters

1. **`crs`**: Coordinate Reference System (e.g., `'EPSG:4326'`, `'EPSG:32610'`)
2. **`crs_transform`**: Affine transformation tuple `(a, b, c, d, e, f)` where:
   - `a` = pixel width (x-scale)
   - `b` = row rotation (typically 0)
   - `c` = x-coordinate of upper-left corner
   - `d` = column rotation (typically 0)
   - `e` = pixel height (y-scale, typically negative for north-up)
   - `f` = y-coordinate of upper-left corner
3. **`shape_2d`**: Tuple of `(width, height)` in pixels

### Helper Function Summary

| Function | Use Case | Parameters |
|----------|----------|------------|
| `helpers.extract_grid_params(ee_obj)` | Match the native grid of an EE Image/ImageCollection | EE object |
| `helpers.fit_geometry(..., grid_scale=...)` | Define grid by pixel size | geometry, CRS, scale |
| `helpers.fit_geometry(..., grid_shape=...)` | Define grid by pixel count | geometry, CRS, shape |

### Manual Grid Definition

For advanced use cases, you can still define grid parameters manually:

```python
ds = xr.open_dataset(
    'ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    crs='EPSG:4326',
    crs_transform=(1.0, 0, -180.0, 0, -1.0, 90.0),
    shape_2d=(360, 180)
)
```

## 5. Troubleshooting

### Issue: "Missing required parameter"

**Error:** `TypeError: missing required argument: 'crs_transform'`

**Solution:** You're using the old API syntax. Update to use grid parameter helpers:

```python
# Add these imports
from xee import helpers
import shapely

# Replace your old xr.open_dataset call with helper-based approach
grid_params = helpers.fit_geometry(
    geometry=your_geometry,
    grid_crs='EPSG:4326',
    grid_scale=(your_scale, -your_scale)
)
ds = xr.open_dataset(collection, engine='ee', **grid_params)
```

### Issue: "Plots are rotated/flipped"

**Problem:** You're still using `.transpose()` from v0.0.x code

**Solution:** Remove the `.transpose()` call - v0.1.0 outputs in the correct orientation by default

### Issue: "Dimension order is wrong for my export"

**Check:** What order does your export library expect?

Most modern geospatial tools expect `[time, y, x]` (which v0.1.0 provides). If you have legacy code expecting `[time, x, y]`, you can still transpose:

```python
# Only if your downstream tool requires the old ordering
data = ds['variable'].transpose('time', 'x', 'y')
```

### Issue: "I need the old behavior"

If you must maintain the old API temporarily, you can pin to v0.0.x:

```bash
pip install "xee<0.1.0"
```

However, we strongly recommend migrating to v0.1.0 for better CF compliance and ecosystem compatibility.

## 6. Testing Your Migration

After updating your code, verify it works correctly:

```python
import ee
import xarray as xr
from xee import helpers
import shapely

# Initialize Earth Engine
ee.Initialize(project='YOUR-PROJECT')

# Test 1: Open a dataset
ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').limit(5)
grid_params = helpers.extract_grid_params(ic)
ds = xr.open_dataset(ic, engine='ee', **grid_params)

# Test 2: Check dimensions
print("Dimensions:", ds['temperature_2m'].dims)
# Should print: ('time', 'y', 'x')

# Test 3: Plot without transpose
ds['temperature_2m'].isel(time=0).plot()

# Test 4: Verify CRS and transform
print("CRS:", grid_params['crs'])
print("Transform:", grid_params['crs_transform'])
print("Shape:", grid_params['shape_2d'])
```

## 7. Additional Resources

- [Main Guide](guide.md) - Complete usage guide with examples
- [API Documentation](api.md) - Detailed API reference
- [Client vs Server Guide](client-vs-server.ipynb) - Examples using v0.1.0 API
- [GitHub Issues](https://github.com/google/Xee/issues) - Report problems or ask questions

## Need Help?

If you encounter issues during migration:

1. Check this guide for common patterns
2. Review the [updated examples](https://github.com/google/Xee/tree/main/examples)
3. Open a [GitHub Discussion](https://github.com/google/Xee/discussions)
4. File an [issue](https://github.com/google/Xee/issues) with a reproducible example

---

**Welcome to Xee v0.1.0!** We believe these changes make the library more powerful, standards-compliant, and easier to integrate with the broader scientific Python ecosystem.

