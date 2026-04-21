# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.

### Recommended development setup

We recommend using Pixi for reproducible local development. The repository
defines dedicated environments for tests and docs so contributors can run the
same commands across Linux, macOS, and Windows.

Install Pixi by following the upstream installation instructions:
<https://pixi.sh/latest/>

From the repository root, use:

```bash
pixi install -e tests
pixi install -e docs
```

The main development commands are:

```bash
pixi run -e tests pytest -q xee/ext_test.py
pixi run -e tests pytest -q xee/ext_integration_test.py
pixi run -e docs docs-build
pixi run -e docs docs-check
```

`docs-build` builds the HTML docs, while `docs-check` runs a stricter Sphinx
build with warnings treated as errors.

### Running tests

The Xee integration tests only pass on Xee branches (no forks). Please run the integration tests locally before sending a PR. To run the tests locally, authenticate using `earthengine authenticate` and run one of the following:

```bash
pixi run -e tests python -m unittest xee/ext_integration_test.py
```

or

```bash
pixi run -e tests python -m pytest xee/ext_integration_test.py
```

For regular unit tests, run:

```bash
pixi run -e tests pytest -q xee/ext_test.py
```

Before opening a PR, run at least:

```bash
pixi run -e tests pytest -q xee/ext_test.py
pixi run -e docs docs-check
```

If your change touches Earth Engine integration behavior and you are working on
an Xee branch, also run the integration tests locally.