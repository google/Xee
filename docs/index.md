# Xee Documentation

```{admonition} Breaking Change in v0.1.0
:class: warning

A major refactor was released in v0.1.0, introducing breaking changes to the Xee API. In most cases, existing code written for pre-v0.1.0 versions will require updates to remain compatible.

- See the [Migration Guide](migration-guide-v0.1.0.md) for details on updating your code.
- If you need more time to migrate, you can pin your environment to the latest pre-v0.1.0 release.
```


Xee is an Xarray extension for Google Earth Engine that lets you open `ee.Image` and `ee.ImageCollection` objects as lazy `xarray.Dataset`s.

```{toctree}
:maxdepth: 2

quickstart
installation
concepts
guide
client-vs-server
performance
api
migration-guide-v0.1.0
faq
why-xee
contributing
code-of-conduct
```

