---
title: Performance & Limits
---

# Performance & Limits

Guidance for working efficiently within Earth Engine and Xee constraints.

## Endpoints

| Endpoint | Use case | Notes |
|----------|----------|-------|
| High‑volume | Reading stored ImageCollections | Higher throughput, intended for bulk pixel access. |
| Standard | Computed collections / iterative dev | Caching can accelerate repeated computations. |

Switch endpoints by passing / omitting `opt_url` in `ee.Initialize`.

## Quotas & Request Parallelism

Earth Engine imposes QPS limits. Large Dask graphs may overrun quotas and cause retries or 429 errors.

Recommendations:

1. Start with modest parallelism (e.g., `DASK_NUM_WORKERS=4`).
2. Coarsen grid (larger pixels) or reduce AOI when prototyping.
3. Consolidate operations server-side (EE `.map`, `.select`, band math) before opening in Xee.
4. Cache intermediate results in memory rather than re-opening repeatedly.

## Chunk Size Considerations

EE responses have an upper size limit (tens of MB). Xee's backend picks reasonable pixel window sizes automatically. If you see many small requests, consider choosing a coarser grid or limiting variable selection to needed bands.

## Memory Pressure

Lazy arrays only materialize when you perform computations. Use Xarray operations that retain laziness (`.mean`, `.sel`, `.where`) before calling `.compute()`.

## Common Optimizations

| Goal | Strategy |
|------|----------|
| Faster experimentation | Limit time range (`isel(time=slice(0, N))`) |
| Stable resolution | Use `fit_geometry` with `grid_scale` |
| Uniform model inputs | Use `fit_geometry` with `grid_shape` |
| Avoid re-fetching | Persist results: `ds.to_zarr()` (advanced) |

## Troubleshooting Slowdowns

1. Inspect task graph size: `ds['var'].data.__dask_graph__()` (diagnostic only).
2. Verify you're not re-initializing EE in each worker.
3. Reduce concurrency; watch for fewer 429 responses.
4. Narrow AOI or temporal range.

## Export & Large Pipelines

For heavy export / transformation workflows consider combining Xee with [Xarray-Beam](https://xarray-beam.readthedocs.io/) or exporting via examples in the repository.
