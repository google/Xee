---
title: FAQ
---

# FAQ

## Why is the y-scale negative?
North-up raster conventions define the origin at the top-left so rows increase downward; a negative y-scale encodes that orientation. Some EE projections use positive y-scale (bottom-left origin). Matching source grid preserves orientation.

## How do I pick between `grid_shape` and `grid_scale`?
Use `grid_shape` when a fixed pixel width/height is required (e.g., ML model inputs). Use `grid_scale` when the physical resolution matters (e.g., aligning with 30 m Landsat data).

## I get 429 quota errors. What do I do?
Reduce parallelism (fewer Dask workers), narrow the AOI or time range, combine server-side operations before opening, or switch to the standard endpoint for computed collections.

## Can I open a computed `ee.ImageCollection`?
Yes. Build the collection with filtering / mapping functions, then pass the resulting collection object directly to `xr.open_dataset(..., engine='ee')` with grid parameters.

## How do I reproduce the same grid later?
Store `crs`, `crs_transform`, and `shape_2d` in metadata or write a helper that re-derives them from the same AOI using `fit_geometry`. Manual override is fine for archival reproducibility.

## How can I export to Zarr?
Use Xarray's `.to_zarr()` on a materialized dataset or see the examples in `examples/` (e.g., Earth Engine to Zarr pipeline). For very large pipelines consider Xarray-Beam.

## Why did dimension ordering change?
To align with CF conventions (`[time, y, x]`) and reduce the need for transposes in plotting / interoperability.

## I'm seeing empty arrays / NaNs.
Your AOI may fall outside the dataset extent or the CRS mismatch caused an unexpected reprojection. Try matching source grid first to confirm availability.

## Do I need shapely geometries?
Helpers accept shapely for convenience. If you already have an EE geometry, you can convert it or use bounding box approaches. Shapely makes reprojection and area reasoning simpler client-side.
