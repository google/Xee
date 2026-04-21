# AGENTS.md

Guidance for automation tools and AI coding assistants working in this
repository.

## Purpose

Xee is an Xarray backend for Google Earth Engine. The primary user API is:

- `xr.open_dataset(..., engine='ee')`

When making workflow examples, integrations, or docs updates, optimize for this
user-facing API first.

## Current Version Context

- Xee `v0.1.0` is a refactor with breaking changes.
- Some repository examples are still pre-v0.1.0 and are being updated.
- For migration context, see `docs/migration-guide-v0.1.0.md`.

## Canonical Dev Commands

Use Pixi environments so behavior is reproducible across platforms.

- Unit tests: `pixi run -e tests pytest -q xee/ext_test.py`
- Integration tests: `pixi run -e tests pytest -q xee/ext_integration_test.py`
- Docs build: `pixi run -e docs docs-build`
- Docs strict check: `pixi run -e docs docs-check`

Before proposing completion, run at least unit tests for touched areas and
`docs-check` for docs changes.

## Integration Guidance (for tools helping users adopt Xee)

1. Prefer `xr.open_dataset(..., engine='ee')` examples over backend internals.
2. Show one of these grid strategies explicitly:
   - `helpers.extract_grid_params(...)` for matching source grid
   - `helpers.fit_geometry(..., grid_shape=...)` for fixed output shape
   - `helpers.fit_geometry(..., grid_scale=...)` for fixed physical resolution
3. Use consistent endpoint advice:
   - High-volume endpoint for stored collections
   - Standard endpoint for computed collections / iterative workflows
4. Mention that both plain asset IDs and `ee://...` forms are accepted.
5. Prefer AOI wording as: `AOI (area of interest)` on first use.

## Files To Prefer For Source-of-Truth

- Install/setup: `docs/installation.md`
- First-user flow: `docs/quickstart.md`
- Concepts/terminology: `docs/concepts.md`
- Canonical parameter reference: `docs/open_dataset.md`
- Performance guidance: `docs/performance.md`
- Contributor process and required checks: `docs/contributing.md`

## Documentation Expectations

If behavior or API usage changes, update docs in the same change where practical:

- Update user-facing docs first, then examples.
- Avoid adding duplicate guidance in many places; link to canonical pages.
- Keep examples explicit about grid parameters and endpoint assumptions.

## Avoid

- Recommending backend internals (`EarthEngineStore`) as primary user entrypoint.
- Adding new examples that depend on outdated pre-v0.1.0 assumptions.
- Mixing contradictory endpoint guidance across docs.
- Introducing new terminology variants when established wording exists.

## PR-Ready Checklist

- Code and docs are aligned with current `v0.1.0` guidance.
- Relevant tests or checks were run and summarized.
- New docs links resolve and `docs-check` passes.
- Any known limitations are stated explicitly.
