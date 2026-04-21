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

Earth Engine imposes [QPS limits](https://developers.google.com/earth-engine/guides/usage). Large Dask graphs may overrun quotas and cause retries or 429 errors.

Recommendations:

1. Start with modest parallelism (e.g., `DASK_NUM_WORKERS=4`).
2. Coarsen grid (larger pixels) or reduce AOI when prototyping.
3. Consolidate operations server-side (EE `.map`, `.select`, band math) before opening in Xee.
4. Cache intermediate results in memory rather than re-opening repeatedly.

## Retry Tuning

Xee uses exponential backoff with jitter for:

- Pixel requests (`getitem_kwargs`) used during array reads.
- Metadata `getInfo()` requests (`getinfo_kwargs`) used during dataset setup and
	helper metadata fetches.

Defaults:

- `getitem_kwargs`: `max_retries=6`, `initial_delay=500` ms
- `getinfo_kwargs`: `max_retries=6`, `initial_delay=1000` ms

`getinfo_kwargs` starts with a longer default delay to reduce setup-time retry bursts against EE metadata endpoints.

You can tune these in `xr.open_dataset(...)`:

```python
ds = xr.open_dataset(
		collection,
		engine='ee',
		crs='EPSG:4326',
		crs_transform=(0.25, 0, -180, 0, -0.25, 90),
		shape_2d=(1440, 720),
		getitem_kwargs={'max_retries': 8, 'initial_delay': 500},
		getinfo_kwargs={'max_retries': 8, 'initial_delay': 1000},
)
```

Rule of thumb:

1. If failures happen during dataset open / metadata fetch, tune
	 `getinfo_kwargs` first.
2. If failures happen during chunk reads / compute, tune `getitem_kwargs` first.
3. Reduce Dask concurrency before increasing retries too aggressively.

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
