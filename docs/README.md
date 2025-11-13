# Xee Documentation (source files)

> **⚠️ Breaking Change in v0.1.0!**
>
> A major refactor was released in v0.1.0, introducing breaking changes to the Xee API. In most cases, existing code written for pre-v0.1.0 versions will require updates to remain compatible.
>
> - See the [Migration Guide](migration-guide-v0.1.0.md) for details on updating your code.
> - If you need more time to migrate, you can pin your environment to the latest pre-v0.1.0 release.

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
