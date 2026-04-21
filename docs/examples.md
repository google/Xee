---
title: Examples
orphan: true
---

# Examples

```{admonition} Status note
:class: warning

Most examples linked here currently target the pre-v0.1.0 API and may not work
as-is with the refactored v0.1.0 interface documented in this site.

These examples are being updated and expanded.
```

This page points to runnable end-to-end examples maintained in the repository.

## Core examples

- [examples/README.md](https://github.com/google/Xee/blob/main/examples/README.md)
- [examples/ee_to_zarr.py](https://github.com/google/Xee/blob/main/examples/ee_to_zarr.py)
- [examples/ee_to_zarr_reqs.txt](https://github.com/google/Xee/blob/main/examples/ee_to_zarr_reqs.txt)

## Dataflow pipeline example

- [examples/dataflow/README.md](https://github.com/google/Xee/blob/main/examples/dataflow/README.md)
- [examples/dataflow/ee_to_zarr_dataflow.py](https://github.com/google/Xee/blob/main/examples/dataflow/ee_to_zarr_dataflow.py)
- [examples/dataflow/requirements.txt](https://github.com/google/Xee/blob/main/examples/dataflow/requirements.txt)
- [examples/dataflow/Dockerfile](https://github.com/google/Xee/blob/main/examples/dataflow/Dockerfile)

## Choosing where to start

- New to Xee: start with [Quickstart](quickstart.md), then
  [examples/README.md](https://github.com/google/Xee/blob/main/examples/README.md).
- Need reproducible chunked outputs: use
  [examples/ee_to_zarr.py](https://github.com/google/Xee/blob/main/examples/ee_to_zarr.py).
- Need scalable batch execution: use the Dataflow example set.
