---
title: Core Concepts
---

# Core Concepts

This page clarifies the ideas you need to be effective with Xee.

## Pixel Grid Parameters

Opening EE data requires specifying an output pixel grid. Xee uses three explicit parameters:

| Parameter | Meaning |
|-----------|---------|
| `crs` | Coordinate Reference System for the output grid (e.g. `EPSG:4326`, `EPSG:32610`). |
| `crs_transform` | Affine transform tuple `(x_scale, x_skew, x_trans, y_skew, y_scale, y_trans)` describing pixel size, rotation/skew, and origin translation in CRS units. |
| `shape_2d` | `(width, height)` of the output grid in pixels. |

### Understanding `crs_transform`

The tuple follows the [Rasterio/`affine.Affine`](https://affine.readthedocs.io/en/latest/) standard. The coefficients correspond to:

- `a`: Scale X (pixel width)
- `b`: Shear X (row rotation)
- `c`: Translation X (x-origin)
- `d`: Shear Y (column rotation)
- `e`: Scale Y (pixel height, usually negative)
- `f`: Translation Y (y-origin)

**Note:** This ordering (a, b, c, d, e, f) differs from the GDAL `GeoTransform` sequence, which is (c, a, b, f, d, e). Ensure you map the translation indices 0 and 3 in GDAL to indices 2 and 5 for Xee.

Instead of constructing these manually, prefer helpers:

- `extract_grid_params(obj)`: Match an `ee.Image` or `ee.ImageCollection` source grid.
- `fit_geometry(geometry, grid_crs, grid_scale=(x, y))`: Define pixel size (resolution) over an AOI.
- `fit_geometry(geometry, grid_crs, grid_shape=(w, h))`: Define output array dimensions, letting resolution float.

### Y Scale Sign & Orientation

`crs_transform[4]` (the y-scale) is negative for north-up imagery (top-left origin) and positive for bottom-left origin layouts. Helpers default to negative (north-up). When matching a source grid, its sign is preserved.

## Dimension Ordering

Datasets are returned as `[time, y, x]` aligning with CF conventions and most geospatial libraries.

## Stored vs Computed Collections

- Stored: unmodified `ee.ImageCollection('ID')` — use high‑volume endpoint for throughput.
- Computed: collections after `.map()`, `.select()`, filtering, band math — standard endpoint sometimes more efficient due to caching.

## Choosing a Grid Strategy

| Situation | Recommended approach |
|-----------|---------------------|
| Just explore dataset | `extract_grid_params` |
| Train ML model (fixed input size) | `fit_geometry(..., grid_shape=...)` |
| Preserve known resolution | `fit_geometry(..., grid_scale=...)` |
| Export with exact projection | Manual parameters (advanced) |

## CRS Units & Transforms

All scale and translation values in the `crs_transform` are expressed in the units of the specified `crs`.

*   **Projected CRSs** (e.g., `EPSG:3857`, **UTM** zones) use linear units, typically **meters** or feet.
*   **Geographic CRSs** (e.g., `EPSG:4326`) use angular units, typically **degrees**.

> **Note on Geographic Distortion:**
> Geographic CRSs (like `EPSG:4326`) define pixels in degrees. Because the ground distance of a degree of longitude shrinks as you move from the equator to the poles, a grid in this CRS will have **non-uniform ground pixel sizes**.
>
> *   **Distortion:** A "square" pixel in degrees becomes a narrow rectangle in meters at high latitudes.
> *   **Analysis Impact:** Euclidean distance and area calculations performed directly on the array (e.g., assuming `1 pixel = X meters`) will be incorrect.
>
> If your analysis requires uniform measurement of **distance or area**, consider reprojecting to a projected CRS (meters) suitable for your region of interest.

## Chunking & Lazy Loading

Data is paged from EE using pixel chunks (bounded by EE's max request size). Xarray+Dask operations trigger parallel pixel fetches, respecting [EE quota limits](https://developers.google.com/earth-engine/guides/usage). See [Performance & Limits](performance.md) for tuning advice.

## Error Patterns

| Symptom | Likely cause | Mitigation |
|---------|--------------|------------|
| Quota exceeded / 429 | Too many parallel pixel requests | Reduce Dask workers or chunk size. |
| Empty array / all NaNs | AOI outside dataset extent | Verify geometry CRS & bounds; try `extract_grid_params`. |
| Distorted aspect | Wrong y-scale sign | Use helpers or invert sign of `grid_scale[1]`. |

## Helpers vs Manual Override

Helpers encapsulate reprojection, bounding logic, and transform math. Manual construction is only needed for reproducibility of pre-agreed custom grids or advanced alignment with external rasters.

## Safe Defaults

- Prefer matching source for exploratory analysis.
- Use `grid_shape` when pixel count matters (consistent model input shape).
- Use `grid_scale` for resolution-sensitive metrics (e.g., indices, physical units).
