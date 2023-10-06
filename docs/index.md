# Xee: A Google Earth Engine extension for Xarray

Xee is an Xarray extension for Google Earth Engine. It aims to help users view
Earth Engine's [data catalog](https://developers.google.com/earth-engine/datasets)
through the lense of arrays.

In this documentation, we assume readers have some familiarity with
[Earth Engine](https://earthengine.google.com/), [Xarray](https://xarray.dev/),
and Python. Here, we'll dive into core concepts related to the integration
between these tools.

## Contents

<!-- TODO(#38): Documentation Plan
- Why Xee?
- Core features
  - `open_dataset()`
  - `open_mfdatasets()`
  - Projections & Geometry
  - Xarray slicing & indexing 101
  - Combining ee.ImageCollection and Xarray APIs.
  - Plotting
  - Lazy Evaluation & `load()`
- Advanced projections
- Performance tuning: A tale of two chunks
- Walkthrough: calculating NDVI
- Integration with Xarray-Beam
- Integration with ML pipeline clients -->

```{toctree}
:maxdepth: 1
why-xee.md
api.md
```
