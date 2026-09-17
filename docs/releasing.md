# 🚀 Releasing Xee (Internal Runbook)

This document is an internal release runbook for maintainers. It is
intentionally kept out of the published docs site.

## Overview

The `Xee` release process is aligned with the Earth Engine client library
releases.

Releases are created directly from the `main` branch using GitHub Actions and
dynamic versioning (`setuptools_scm`).

---

## 🔍 How to Determine the Version Number

Before triggering a release, the releaser needs to choose the target version
number according to [Semantic Versioning](https://semver.org)
(`MAJOR.MINOR.PATCH`).

### 1. Check the Latest Released Version
Visit the **[Releases](https://github.com/google/Xee/releases)** page and look
for the release with the green `Latest` badge at the top (e.g., `v0.1.1`).

### 2. Inspect Changes Since the Last Release
Visit the GitHub compare page comparing the latest tag to `main`:
`https://github.com/google/Xee/compare/<LATEST_TAG>...main`
*(Example: [`v0.1.1...main`](https://github.com/google/Xee/compare/v0.1.1...main))*.

This view lists all merged PRs, commits, and file diffs since the last release.

### 3. Choose the Version Bump (Major, Minor, or Patch)

Given the latest version `0.Y.Z` (e.g., `0.1.1`):

* **Patch Bump (`0.Y.Z+1` ➔ `0.1.2`):**
  * Use for routine maintenance releases containing **bug fixes**,
    **dependency bumps**, **documentation updates**, or internal CI/build
    improvements without new user-facing APIs.
* **Minor Bump (`0.Y+1.0` ➔ `0.2.0`):**
  * Use when **new user-facing features**, **new backend helpers**, or new
    functions/modules are added in a backward-compatible manner.
* **Major Bump (`X+1.0.0` ➔ `1.0.0`):**
  * Reserved for major milestone releases or incompatible public API changes.

---

## 📅 Step-by-Step Release Instructions

The release process follows a safe **Draft ➔ Review ➔ Publish** workflow.

### Step 1: Trigger the Release Workflow (Creates Draft)
1. Navigate to the **[Actions](https://github.com/google/Xee/actions)** tab on
   GitHub.
2. Under **Workflows** on the left, select **`Release`** (or go to
   [`.github/workflows/release.yml`](https://github.com/google/Xee/actions/workflows/release.yml)).
3. Click **Run workflow**:
   * **Branch:** `main`
   * **Version number to release:** Enter the target version (e.g., `0.1.2` or
     `0.2.0`).
4. Click **Run workflow**.
5. The workflow will validate the version, tag `main` (e.g. `v0.1.2`), and
   create a **Draft Release** with auto-generated release notes.

### Step 2: Review and Publish the Release
1. Navigate to the **[Releases](https://github.com/google/Xee/releases)** page
   on GitHub.
2. Find the newly created **Draft Release** and click **Edit**.
3. **Review the draft against the following checklist:**
   * **Tag & Title:** Verify that the release title and tag match the target
     version exactly (e.g., `v0.1.2`) with no typos or extra prefixes.
   * **Changelog Scope:** Check the auto-generated **What's Changed** section
     and the **Full Changelog** compare link at the bottom (e.g.
     `compare/v0.1.1...v0.1.2`). Ensure the listed PRs and commits accurately
     span *only* the changes merged since the previous release.
   * **Release Notes:** Clean up any redundant entries or formatting issues and
     highlight key new features or breaking changes if applicable.
4. Click **Publish release** (ensure "Set as latest release" is checked).

### Step 3: Verify PyPI and Merge Conda-Forge PR
1. **PyPI Publishing (Automated)**:
   * Publishing the GitHub release in the web UI automatically triggers the
     **`Publish to PyPI`** workflow.
   * Check the
     **[Actions](https://github.com/google/Xee/actions/workflows/publish.yml)**
     tab to verify the build and upload succeed.
   * Verify the new version is live on [PyPI](https://pypi.org/project/xee).
2. **Conda-Forge Feedstock (Action Required)**:
   * Within ~1 hour of the PyPI publish, the `regro-cf-autotick-bot` will
     automatically open a version bump PR on the
     **[xee-feedstock](https://github.com/conda-forge/xee-feedstock/pulls)**
     repository.
   * **Releaser Action Checklist:**
     1. Open the new PR on the
        **[xee-feedstock](https://github.com/conda-forge/xee-feedstock/pulls)**
        repo.
     2. **If dependencies have not changed:** Wait for the CI checks (Linux,
        macOS, Windows) to turn green, then click **Merge**.
     3. **If dependencies changed in `pyproject.toml`:** Edit `recipe/meta.yaml`
        in the PR branch to add, remove, or pin the updated `host` and `run`
        dependencies before merging.
     4. Once merged, Conda-Forge will automatically build and distribute the new
        package to the `conda-forge` channel.

---

## 🛡️ Dependency Extras Guardrails

* `dataflow` is intended for production export workflows.
* `examples` may include `dataflow` dependencies plus any additional packages
  used by user-facing examples.
* Keep `examples` explicit (do not use self-referential extras like
  `xee[dataflow]` inside `project.optional-dependencies`).
* CI enforces that every dependency in `dataflow` is also present in `examples`
  (subset check in `.github/workflows/ci-build.yml`).
