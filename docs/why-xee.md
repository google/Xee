# Why Xee?

We noticed two clusters of users working with climate and weather data at
Google Research: Some were [Xarray](https://xarray.dev) (and
[Zarr](https://zarr.dev/)) centric and others, Google Earth Engine centric. Xee
came about as an effort to bring these two groups of developers closer together.

## Goals

Primary Goals:

- Make [EE-curated data](https://developers.google.com/earth-engine/datasets)
  accessible to users in the Xarray community and to the wider scientific Python
  ecosystem.
- Make it trivial to avoid quota limits when computing pixels from Earth Engine.
- Provide an easy way for scientists and ML practitioners to coalesce Earth data
  at different scales into a common resolution.

Secondary Goals:

- Provide a succinct interface for querying Earth Engine data at scale (i.e. via
  [Xarray-Beam](https://xarray-beam.readthedocs.io/)).
- Make it trivial to quickly [export Earth Engine data to Zarr](https://github.com/google/xee/tree/main/examples#export-earth-engine-imagecollections-to-zarr-with-xarray-beam).
- Provide compelling alternative for the need to export Zarr in the first
  place (e.g. during the ML training process).

## Approach

With the addition of Earth Engine's [Pixel API](https://medium.com/google-earth/pixels-to-the-people-2d3c14a46da6),
it became possible to easily get NumPy array data from `ee.Image`s. In building
tools atop of this, we noticed that the best practices for managing data were
Xarray-shaped. For example:

- Our codebases involved many similar LOC to translate between Earth Engine and
  arrays: Users typically thought in NumPy and molded EE's Python client to fit
  those idioms.
- We often needed to page `computePixel()` requests in a way that's strikingly
 similar to Dask/Xarray's concept of [`chunks`](https://docs.xarray.dev/en/stable/user-guide/dask.html#what-is-a-dask-array).
- Users were wrapping NumPy arrays within dataclasses to associate metadata and
  labels with data.

In an attempt to group these disparate solutions into a singular interface, we
experimented with wrapping `computePixels()` into
[Xarray's standard mechanism for defining backends](https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html). The result of this effort is Xee.


## An array by any other name? (Xee vs Zarr)

[Zarr](https://zarr.dev/) has been growing in relevance to the world of [cloud-based scientific data](https://doi.org/10.1109/MCSE.2021.3059437).
Members of the open source community have [demonstrated](https://www.youtube.com/watch?v=0bqpxX3Nn_A)
that Zarr is more of a data protocol rather than a data format. In many ways,
Xee is inspired by this work. To this end, we'd like to point out some
similarities and differences between Zarr backed and Earth Engine backed data in
Xarray.

Similarities:
- **Xarray-compatible**: Of course, this library proves that both types of data
  stores can be compatible with Xarray. [Zarr](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr)
  reading and writing is deeply integrated into Xarray as well.
- **Optimal IO Chunks**: Ultimately, cloud-based data stores will inherently
  involve networking overhead. There are similarities in the best way to page
  data across a network into a local context: the optimal Zarr chunk
  size is around [10-100 MBs](https://esipfed.github.io/cloud-computing-cluster/optimization-practices.html#chunk-size). With Earth Engine's backend, the maximum chunk size possible
  is 48 MBs.

Differences:
- **Quota vs No Quota**: Since Earth Engine is API based, there are quota
  restrictions that limit IO, namely a 100 QPS limit on data requests. Readers
  all need to be authenticated and tied to a GCP project quota. Zarr, on the
  other hand, has a lower level access pattern. Reading is delegating to basic
  permissions on cloud buckets.
- **On the fly vs up-front data shaping**: In Zarr, the representation of data
  at rest fundamentally influences performance at query time. For this reason,
  [rechunking](https://xarray-beam.readthedocs.io/en/latest/rechunking.html) and
  projecting is a common routine performed up front on Zarr when data does not
  quite fit the problem at hand. Earth Engine provides a more flexible interface
  than this. Since datasets are pyramided (either at [ingestion](https://developers.google.com/earth-engine/help_collection_criteria) or server-side), users are free to request the
  resolution and projection of the data during dataset open. Similarly, while
  Earth Engine's internal dataset does fit an internal chunking scheme, chunking
  schemes are a lot more fungibile.

We hope that this comparison provides the user of a set of useful precedents
for working with cloud-based datasets.

