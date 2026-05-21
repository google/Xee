# Open Dataset Reference (`engine='ee'`)

This page is the canonical user-facing reference for calling:

```python
xr.open_dataset(..., engine='ee')
```

## How The Call Chain Works

In plain terms:

1. You call `xarray.open_dataset(..., engine='ee')`.
2. Xarray routes that call to Xee's backend entrypoint method:
   `xee.EarthEngineBackendEntrypoint.open_dataset`.
3. That entrypoint creates and uses `xee.EarthEngineStore` internally to stream
   pixels and metadata.

`EarthEngineStore` is an internal/core backend type. Most users should treat
`xr.open_dataset(..., engine='ee')` as the public API and use this page as the
parameter reference.

Related API pages:

- [EarthEngineBackendEntrypoint autosummary](_autosummary/xee.EarthEngineBackendEntrypoint)
- [EarthEngineStore autosummary](_autosummary/xee.EarthEngineStore)

## Required vs Optional Parameters

When `engine='ee'`, the grid parameters are required at call time:

- `crs`
- `crs_transform`
- `shape_2d`

Most other parameters are optional tuning or decoding controls.

Input source (`filename_or_obj`) can be one of:

- An `ee.ImageCollection` object
- An `ee.Image` object (wrapped internally as an ImageCollection)
- An asset id string/path, including `ee://...` / `ee:...` style URIs

## Canonical Parameter List

The signature and parameter docs below are rendered from the backend method
used at runtime, so this reference stays aligned with implementation behavior.

```{eval-rst}
.. currentmodule:: xee
.. automethod:: EarthEngineBackendEntrypoint.open_dataset
```

## Parameter Name Mapping (User API vs Core Backend)

Most users should pass arguments to `xr.open_dataset(..., engine='ee')`.
Some names differ in the core backend API (`EarthEngineStore.open`).

| User-facing (`xr.open_dataset`) | Core backend (`EarthEngineStore.open`) | Notes |
|---|---|---|
| `filename_or_obj` | `image_collection` | Backend always operates on an `ee.ImageCollection` |
| `io_chunks` | `chunk_store` / `chunks` | Same concept, different name at different layers |
| `ee_mask_value` | `mask_value` | Same behavior |

If you are reading backend API pages, these name differences are expected.

## Practical Parameter Guide

The list below explains the most common practical usage patterns for parameters
you may see in user docs and backend API docs.

### `image_collection` (`ee.ImageCollection`)

- Backend/core parameter corresponding to user-facing `filename_or_obj`.
- You usually pass either an EE object (`ee.ImageCollection`/`ee.Image`) or
  an asset URI string into `xr.open_dataset`; Xee normalizes to an
  `ee.ImageCollection` internally.
- Asset paths usually come from either:
  - The public Earth Engine Data Catalog:
    <https://developers.google.com/earth-engine/datasets/catalog>
  - The Awesome GEE Community Catalog (community datasets):
    <https://gee-community-catalog.org/projects/>
  - Your own Earth Engine assets (personal, team, or project-owned):
    <https://developers.google.com/earth-engine/guides/asset_manager>
- Example catalog path:
  `ECMWF/ERA5_LAND/MONTHLY_AGGR` (or URI form `ee://ECMWF/ERA5_LAND/MONTHLY_AGGR`).

### `crs` (`str`)

- Output coordinate reference system for all opened variables.
- Required at runtime for `engine='ee'`.
- Prefer `helpers.extract_grid_params(...)` / `helpers.fit_geometry(...)`
  unless you explicitly need a manual override.

### `crs_transform` (`tuple[float, float, float, float, float, float] | Affine`)

- Geotransform defining pixel size/origin in the selected CRS.
- Required at runtime for `engine='ee'`.
- Keep this consistent with `shape_2d`; mismatches can cause confusing bounds
  or orientation outcomes.

### `shape_2d` (`tuple[int, int]`)

- Pixel grid size in `(width, height)` order.
- Required at runtime for `engine='ee'`.
- Large shapes increase memory and request pressure.

### `chunks` (`int | dict[Any, Any] | Literal['auto'] | None`)

- Default: `None`.
- Dask/Xarray chunking in the returned dataset.
- Affects downstream compute scheduling/memory behavior, not just EE request
  boundaries.
- Start with modest time chunks and tune only when needed.

### `n_images` (`int`)

- Default: `-1` (include all images).
- Limit the number of images loaded from the collection (`-1` means all).
- Useful for quick iteration, debugging, or very large collections.

### `primary_dim_name` (`str | None`)

- Default: `None` (resolved to `time`).
- Rename the primary stacked dimension (default: `time`).
- Usually keep default unless integrating with an existing schema.

### `primary_dim_property` (`str | None`)

- Default: `None` (resolved to `system:time_start`).
- EE image property used to derive primary-dimension coordinate values
  (default: `system:time_start`).
- Change only if your collection indexing semantics depend on another property.

### `mask_value` (`float | None`)

- Default: `None` (resolved to `np.iinfo(np.int32).max`, i.e. `2147483647`).
- Backend/core mask sentinel corresponding to user-facing `ee_mask_value`.
- Used to convert EE nodata/sentinel pixels to NaN-friendly behavior.

### `request_byte_limit` (`int`)

- Default: `48 * 1024 * 1024` (48 MB).
- Upper bound for per-request payload size.
- Advanced tuning control: Earth Engine size constraints vary by workload.
- Prefer lowering this value when you hit request-size instability.
- Avoid increasing unless validated for your specific dataset/query pattern.

### `ee_init_kwargs` (`dict[str, Any] | None`)

- Default: `None`.
- Keyword arguments forwarded to `ee.Initialize(...)` during optional worker
  auto-initialization.
- Useful in distributed settings where workers need credentials/project config.

### `ee_init_if_necessary` (`bool`)

- Default: `False`.
- Whether Xee should attempt EE initialization on demand (commonly for remote
  workers).
- Keep `False` for standard local workflows where EE is already initialized.

### `executor_kwargs` (`dict[str, Any] | None`)

- Default: `None` (internally treated as `{}`).
- Thread pool settings for parallel pixel retrieval.
- Advanced tuning: increasing worker count may improve throughput or trigger
  more rate/quota pressure depending on workload.

### `getitem_kwargs` (`dict[str, int] | None`)

- Default: `None` (uses internal defaults: `max_retries=6`,
  `initial_delay=500` ms).
- Retry/backoff tuning for array indexing fetches.
- Useful for transient quota/rate errors.
- Tune conservatively (`max_retries`, `initial_delay`) and prefer reducing
  concurrency before aggressive retry expansion.

### `fast_time_slicing` (`bool`)

- Default: `False`.
- Enables a faster slice path by loading images by ID.
- Important: for computed/modified ImageCollections, this can return original
  asset images (looked up by ID) rather than your computed image values.

## `fast_time_slicing` Deep Dive

`fast_time_slicing=True` is an important optimization, but it changes how time
slices are resolved.

What it does:

- `False` (default): Xee slices directly from the in-memory EE
  `ImageCollection` object.
- `True`: Xee slices by `system:id` first and then loads by those IDs.

Why this can be confusing:

- If your collection is computed/modified (for example: `.map(...)`, band math,
  clipping/masking, or replacing images), slicing by ID can bypass those
  computed modifications and return the original images associated with the
  IDs.
- In other words, `fast_time_slicing=True` can be faster but may not reflect
  computed collection transformations.

When to use it:

- Good fit: direct/stored collections where you want faster time slicing and
  are not depending on computed per-image transformations.
- Use caution: computed collections where transformed pixel values must be
  preserved in reads.

Practical recommendation:

1. Start with `fast_time_slicing=False` for correctness-sensitive workflows.
2. Enable `fast_time_slicing=True` only after validating that sampled outputs
   match your intended processing semantics.
3. If enabled and your collection lacks image IDs, Xee logs a warning and
   falls back to default (non-fast) behavior.

## Common Recipes

### 1. Match source projection/resolution

Use this when you want output aligned to the dataset's native grid.

```python
import ee
import xarray as xr
from xee import helpers

ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
grid = helpers.extract_grid_params(ic)

ds = xr.open_dataset(ic, engine='ee', **grid)
```

### 2. Manual grid override

Use this when you must align with an external raster/grid spec.

```python
import xarray as xr

manual_crs = 'EPSG:4326'
manual_transform = (0.25, 0, -180, 0, -0.25, 90)
manual_shape = (1440, 720)  # (width, height)

ds = xr.open_dataset(
    'ee://ECMWF/ERA5_LAND/MONTHLY_AGGR',
    engine='ee',
    crs=manual_crs,
    crs_transform=manual_transform,
    shape_2d=manual_shape,
)
```

### 3. Performance/chunking tuning

Use this when throughput or memory behavior needs tuning.

```python
import ee
import xarray as xr
from xee import helpers

ic = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
grid = helpers.extract_grid_params(ic)

ds = xr.open_dataset(
    ic,
    engine='ee',
    **grid,
    chunks={'time': 12},
    io_chunks={'time': 24, 'x': 256, 'y': 256},
    request_byte_limit=32 * 1024 * 1024,
)
```

```{admonition} Advanced tuning only
:class: warning

`io_chunks` and `request_byte_limit` are advanced controls. Earth Engine
imposes response/request size constraints, so these values usually require
trial-and-error for each workload.

Start from defaults and tune conservatively. In most cases, reducing request
size is safer than increasing it.
```

Notes:

- `chunks` controls Dask chunking in Xarray.
- `io_chunks` controls request windows used by Xee for Earth Engine reads.
- `request_byte_limit` limits per-request payload size. Prefer reducing this if
  you encounter request-size failures or unstable reads.
- Avoid increasing `request_byte_limit` unless you have validated behavior
  against Earth Engine limits for your specific dataset and query pattern.

## Object vs URI Inputs

For `engine='ee'`, these are equivalent in outcome once resolved:

- Passing an EE object (`ee.ImageCollection`/`ee.Image`)
- Passing a URI/asset id string (`ee://...` style or asset path string)

Object inputs are often convenient in notebooks where you've already built a
computed collection. URI/asset id strings are useful for concise, declarative
loading and config-driven workflows.
