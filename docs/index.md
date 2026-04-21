# Xee Documentation

```{admonition} Breaking Change in v0.1.0
:class: warning

A major refactor was released in v0.1.0, introducing breaking changes to the Xee API. In most cases, existing code written for pre-v0.1.0 versions will require updates to remain compatible.

- See the [Migration Guide](migration-guide-v0.1.0.md) for details on updating your code.
- If you need more time to migrate, you can pin your environment to the latest pre-v0.1.0 release.

During the v0.1.0 prerelease window, `pip install xee` and `conda install xee` may still install the previous stable line. To follow this documentation for the refactored API, install with `pip install --upgrade --pre xee` (or a pinned RC such as `pip install xee==0.1.0rc1`).
```


Xee is an Xarray extension for Google Earth Engine that lets you open `ee.Image` and `ee.ImageCollection` objects as lazy `xarray.Dataset`s.

```{toctree}
:maxdepth: 2

quickstart
installation
concepts
guide
open_dataset
client-vs-server
performance
api
migration-guide-v0.1.0
faq
why-xee
contributing
code-of-conduct
```

