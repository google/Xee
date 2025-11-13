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

Instead of constructing these manually, prefer helpers:

- `extract_grid_params(obj)`: Match an `ee.Image` or `ee.ImageCollection` source grid.
- `fit_geometry(geometry, grid_crs, grid_scale=(x, y))`: Define pixel size (resolution) over an AOI.
- `fit_geometry(geometry, grid_crs, grid_shape=(w, h))`: Define output array dimensions, letting resolution float.

### Y Scale Sign & Orientation

`crs_transform[4]` (the y-scale) is negative for north-up imagery (top-left origin) and positive for bottom-left origin layouts. Helpers default to negative (north-up). When matching a source grid, its sign is preserved.

## Dimension Ordering

Datasets are returned as `[time, y, x]` (v1.0+) aligning with CF conventions and most geospatial libraries. Prior versions used `[time, x, y]`. If code assumed positional indices, update to name-based access: `ds.sizes['x']`, `ds.sizes['y']`.

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

All scale/translation values are expressed in units of `crs`. Degrees for geographic CRSs; meters (or feet) for projected CRSs. Plate Carrée (`EPSG:4326`) has non-uniform ground size — consider a projected CRS for area/length sensitive analysis.

## Chunking & Lazy Loading

Data is paged from EE using pixel chunks (bounded by EE's max request size). Xarray+Dask operations trigger parallel pixel fetches, respecting EE quota limits (e.g., ~100 QPS for certain endpoints). See [Performance & Limits](performance.md) for tuning advice.

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
