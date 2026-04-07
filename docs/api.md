# Xee API

```{eval-rst}
.. currentmodule:: xee
```

## User grid helpers

High-level utilities for deriving or matching pixel grid parameters passed to
``xarray.open_dataset(..., engine='ee')``.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    fit_geometry
    extract_grid_params
    set_scale
    PixelGridParams
```

## Core extension backend

Lower-level interfaces used internally by the xarray backend. Most users do
not need these directly; they're documented for advanced workflows and
debugging.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    EarthEngineBackendEntrypoint
    EarthEngineStore
    EarthEngineBackendArray
```

## Other utilities

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    geometry_to_bounds
```