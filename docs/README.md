# Xee Documentation (source files)

> **Breaking Change in v0.1.0+**
>
> Xee v0.1.0 introduced a refactored API with breaking changes relative to the 0.0.x series. The documentation in this folder reflects the v0.1.0+ API.
>
> If you are upgrading from 0.0.x, see [migration-guide-v0.1.0.md](migration-guide-v0.1.0.md).

## For nicely rendered documentation

Visit **Read the Docs**: https://xee.readthedocs.io/en/latest/

## About this folder

This `docs/` folder contains the source files used to build the documentation site with Sphinx and MyST.

If you're browsing on GitHub:
- Start from [`index.md`](index.md) for the documentation landing page
- Or build the docs locally (see below)

## Build locally (optional)

```bash
cd docs
make html
open _build/html/index.html  # or xdg-open on Linux
```

## Project information

For project overview and repository information, see the root [`README.md`](../README.md).
