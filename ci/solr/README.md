<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Solr compatibility with nightly cuVS

[solr-cuvs.yaml](../../.github/workflows/solr-cuvs.yaml) builds and tests Solr
against existing nightly cuVS artifacts. It runs from nightly `test.yaml` or by
manual dispatch. Compatibility failures are allowed by default; set
`continue_on_error=false` to fail the caller.

## Build and test

Both jobs use the same fixed CUDA/OS image defined in the workflow:

1. **CPU (`solr-build`):** download native, Java, and Lucene artifacts from one
   completed cuVS `build.yaml` run; apply the Solr patches; build the full
   distribution and compile the integration tests.
2. **GPU (`solr-test`):** restore the CPU job's bundle and exact Conda package
   versions, initialize CUDA, and run the precompiled `TestCuVSCodecSupportIT`
   offline. This job starts only after the bundle is uploaded.

The bundle contains the distribution, test classes and resources, JVM options,
Gradle distribution, native-library probe, and Conda package specification.
The GPU job downloads the native package channel from the same producer run.
`gpu-test.gradle` executes the exported tests without rebuilding Solr.

`artifacts.py` stages the plain cuVS Java JAR and the codec matching Solr's Lucene
minor version in a local Maven repository. `nightly.init.gradle` substitutes only
cuVS dependencies, preserving other dependency locks. Resolved cuVS JARs, the
packaged distribution, and the exported test classpath must match the staged
checksums.

The GPU probe must initialize successfully and `testIndexAndSearch` must pass.
Missing, skipped, failing, or CPU-fallback results fail validation. The patch and
GPU task enforce `solr.tests.cuvs.required=true`; the JUnit validator checks the
result independently. Test names and counts appear in the GPU step's log.

Diagnostics are uploaded as `solr-cuvs-build-cuda...` (CPU) and
`solr-cuvs-cuda...` (GPU); the transfer archive is `solr-cuvs-bundle-cuda...`.
They record source SHAs, producer run ID, package versions, JAR checksums, patch
and phase logs, and JUnit results. Job summaries report failures even when allowed.

## PR and manual validation

The temporary `solr-cuvs-validation` caller in
[pr.yaml](../../.github/workflows/pr.yaml) pins a successful nightly and makes
both jobs required through `pr-builder`. The CI scripts come from the PR revision;
the selected cuVS checkout supplies dependencies and artifact identity. This
tests the CI changes against nightly cuVS binaries. Remove the temporary caller
and its `pr-builder` dependency before merging the nightly-only integration.

For manual dispatch, select the workflow branch with `gh workflow run solr-cuvs.yaml
--ref <branch>` and supply these inputs with `-f`:

| Input | Purpose / default |
| --- | --- |
| `cuvs_commit` | Commit or branch with completed `build.yaml` artifacts; `main` |
| `branch` | Producer's branch; `main` |
| `date` | Nightly build date (`YYYY-MM-DD`); optional |
| `build_run_id` | Specific producer run; otherwise resolved from cuVS SHA |
| `solr_repository` / `solr_commit` | Downstream checkout; `apache/solr` / `main` |
| `continue_on_error` | Set `false` for required validation; default `true` |

Manual dispatch requires the workflow to exist on the default branch.
The selected producer must be a completed `build.yaml` run matching the cuVS SHA;
a PR's own run ID is not supported. All three artifacts must be available for
the workflow's CUDA version.

## Temporary Solr patches

[patches/solr.json](patches/solr.json) lists patches in application order:
`file` names a diff relative to the manifest and `issue` explains why it is needed
(with an upstream link when available). Every listed patch is required.

The `patch` phase uses `jq` to read the manifest and `git apply` to apply each
diff. A successful reverse check identifies an already-applied patch; other
application failures fail the phase. It needs no Conda environment, compiler,
or GPU. Results preserve the base Solr SHA, manifest, diffs, and `patch.log`,
including Git's error output on failure.

The current patch updates the vectors-format constructor to
`AcceleratedHNSWParams.Builder`, uses
`ThreadLocalCuVSResourcesProvider.isSupported()`, requires GPU support when
requested by CI, and updates the codec assertion for Lucene 10.4. Remove the
patch once equivalent changes are upstream.

## Local checks

Run the artifact regression tests without Java, CUDA, or network access:

```sh
python -m unittest discover -s ci/solr/tests -v
```

Check patch application against a disposable Solr checkout:

```sh
SOLR_SOURCE_DIR=/absolute/path/to/solr ci/test_solr.sh patch
```

To reproduce the jobs, use the workflow's RAPIDS CI image and environment
variables, with fresh work/results directories for each job:

* CPU: run `ci/test_solr.sh setup`, then `patch` and `build`.
  Generate the summary with `SOLR_REPORT_STAGE=build ci/test_solr.sh report`.
* GPU: transfer `SOLR_WORK_DIR/solr-test-bundle.tar.gz` into the GPU job's work
  directory, then run `ci/test_solr.sh restore`, `test`, and `report`.
  Only the CI checkout and bundle are needed on this runner.
