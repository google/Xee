# Releasing Xee (internal runbook)

This document is an internal release runbook for maintainers. It is intentionally
kept out of the published docs site.

## Overview

Xee uses `setuptools-scm`, so the package version comes from git tags.

- Prerelease tags: `vX.Y.ZrcN` (for example, `v0.1.1rc1`)
- Stable tags: `vX.Y.Z` (for example, `v0.1.1`)
- Release branch per train: `vX.Y.Z-release`

The release workflows are:

- `Release-prerelease`: creates or reuses `vX.Y.Z-release`, tags `vX.Y.ZrcN`, creates a GitHub prerelease
- `Release-stable`: requires existing `vX.Y.Z-release`, tags `vX.Y.Z`, creates a GitHub release

Publishing a GitHub release (non-draft) triggers `.github/workflows/publish.yml`,
which builds and uploads to PyPI.

## Prerelease workflow

Use this when starting or continuing a release candidate cycle.

1. Open the GitHub Actions page and select `Release-prerelease`.
2. Click Run workflow and provide:
   - `version`: base version such as `0.1.1`
   - `rc_number`: release candidate number such as `1`, `2`, `3`
   - `publish_immediately`: set `false` to create a draft first
3. The workflow will:
   - create `v0.1.1-release` from `main` if it does not exist, or reuse it if it does
   - create and push tag `v0.1.1rc1`
   - create a draft (or published) GitHub prerelease
4. Review release notes in GitHub Releases.
5. Publish the prerelease when ready.

If fixes are needed, apply them on the `vX.Y.Z-release` branch, then rerun
`Release-prerelease` with an incremented `rc_number`.

## Stable workflow

Use this to finalize a tested release branch.

1. Open the GitHub Actions page and select `Release-stable`.
2. Click Run workflow and provide:
   - `version`: stable version such as `0.1.1`
   - `publish_immediately`: set `false` to create a draft first
3. The workflow will:
   - verify `v0.1.1-release` exists (and fail with guidance if not)
   - create and push tag `v0.1.1`
   - create a draft (or published) GitHub release
4. Review release notes in GitHub Releases.
5. Publish the release when ready.

## Notes and safeguards

- No manual version file bump is required for Xee.
- Avoid creating stable releases directly from `main`; use the release branch path.
- If a stable release was published by mistake, coordinate with maintainers before
  deleting tags or editing release records.
