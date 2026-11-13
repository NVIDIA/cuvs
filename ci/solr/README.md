<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Solr compatibility with nightly cuVS

`.github/workflows/solr-cuvs.yaml` is callable from nightly `test.yaml` and can
also be dispatched manually. Like cuDF's `velox-cudf` workflow, it accepts a
downstream repository/ref and initially allows compatibility failures without
failing the nightly workflow. A CPU job builds the full Solr distribution and
compiles the integration tests. A separate GPU job runs the precompiled
`TestCuVSCodecSupportIT` on one amd64 configuration from the cuVS build matrix.

The workflow consumes **existing artifacts**, rather than rebuilding cuVS. It
resolves one completed `build.yaml` run for the selected cuVS commit, then takes
the native Conda package, Java bindings, and Lucene codecs from that run. An
explicit `build_run_id` can select a particular run; its SHA and workflow are
checked. Missing artifacts or unsupported Lucene versions fail the job.

The CI scripts come from the workflow revision. The cuVS checkout selected by
`cuvs_commit` supplies the dependency specification and artifact identity, so an
older build can be replayed without requiring it to contain this workflow.

## CPU build and GPU test jobs

`solr-build` runs on `linux-amd64-cpu16`. It installs the selected nightly
artifacts, applies the temporary Solr patches, builds the distribution, and
compiles the test classes. Setup does not require a GPU; CUDA's virtual package
version is supplied to Conda for dependency resolution on the CPU runner.

`export-tests.init.gradle` exports the configured Solr test task's runtime
classpath, compiled classes, JVM options (including its random seed), and test
resources into a portable bundle. The bundle also contains the Solr distribution,
the exact Gradle distribution, the compiled native-library probe, and an explicit
Conda package specification. It excludes build caches and the native package
channel, which the GPU job downloads from the original cuVS producer run.

`solr-test` starts on the GPU runner only after that bundle is uploaded. It
restores the same package versions, relocates the local native-channel URLs, and
verifies the cuVS JAR checksums in both the distribution and test classpath.
It then initializes CUDA and runs `gpu-test.gradle` offline. This standalone
project only has a test task: it has no Solr compilation tasks, source sets, or
remote dependency repositories. The GPU job does not re-checkout a moving Solr
branch or regenerate dependency versions.

Build diagnostics are uploaded as `solr-cuvs-build-cuda...`, the transfer archive
as `solr-cuvs-bundle-cuda...`, and GPU results as `solr-cuvs-cuda...`. An allowed
CPU failure does not schedule GPU work without an uploaded bundle. Set
`continue_on_error=false` to make failures in either job fail the caller.

## Manual runs

Once the workflow is available on the default branch, dispatch `solr-cuvs.yaml`
with the branch containing the implementation to test selected as the workflow
ref (`gh workflow run ... --ref <branch>`). Set:

* `cuvs_commit`: the SHA of a completed cuVS build (or a branch to resolve).
* `branch`: the producer's branch, normally `main`.
* `build_run_id`: optional, useful when reproducing a particular nightly.
* `solr_repository` / `solr_commit`: `apache/solr` and `main` by default, or
  another ref to reproduce a downstream change. CI applies the temporary patches
  listed in `patches/solr.json` to the selected checkout.
* `continue_on_error`: set to `false` for a hard-failing manual validation run.

The result artifact records source SHAs, producing run ID, Conda package URLs,
Maven coordinates and checksums, dependency resolution, patch manifest and diffs,
patch application logs, phase logs, and JUnit reports. The job summary reports
unsuccessful checks even when failure is allowed.

## Testing the initial PR

For initial validation, `pr.yaml` temporarily calls this workflow through
`solr-cuvs-validation`, with failures enabled and the job included in the required
`pr-builder` aggregate. It is pinned to the successful
[2026-11-13 main nightly](https://github.com/NVIDIA/cuvs/actions/runs/43373472245):

* cuVS SHA: `a93bbfad1f9c26efa113bc6f1b1427bfbb23c9ed`
* Build run ID: `43373472245`
* Build date: `2026-11-13`

The native, Java, and Lucene artifacts were verified available for CUDA 13.4.3.
The production `test.yaml` caller remains nightly-only. Manual dispatch of a
newly introduced workflow also has a [default-branch registration requirement](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

The existing PR workflow runs on the repository's `pull-request/<number>` push
branches. The local reusable-workflow reference and root checkout use that PR
revision, while the separate cuVS checkout and all three artifact downloads use
the selected nightly build. Solr defaults to `apache/solr@main`, with the
repository-managed temporary patches applied before building. This validates
the new CI implementation against nightly cuVS; it does not test cuVS binaries
built by the PR.

Check the GPU job and its `solr-cuvs-cuda...` result artifact: setup, distribution
build, and GPU indexing must all pass. `continue_on_error: false` exposes failures
in the workflow result and the `pr-builder` aggregate. Remove the temporary
caller and aggregate dependency before merging the nightly-only integration.

A PR's own run ID is not accepted as `build_run_id`: the downloader deliberately
requires a completed `build.yaml` producer with the selected cuVS SHA. Testing PR
binaries would require additional support for the `pr.yaml` producer, its matrix,
and downloading artifacts while the producer workflow is still running.

## Dependency substitution

`artifacts.py` reads Solr's `apache-lucene` version from its version catalog and
stages the matching `cuvs-lucene-X.Y` module plus the plain `cuvs-java` JAR and
their standalone POMs in an isolated file-based Maven repository. For example,
Solr on Lucene 10.4 uses `cuvs-lucene-10.4`; the legacy `cuvs-lucene` coordinate
must not select the relocated Lucene 10.2 variant.

`nightly.init.gradle` applies only to this checkout. It substitutes the cuVS
dependencies, ignores only their old locks, and restricts their groups to the
staged repository. All other dependency locks remain active. Every resolved
cuVS JAR and the JARs packaged in the distribution must match the staged hashes.
Dependency lockfiles are not rewritten. Source compatibility patches are applied
separately as described below.

## Temporary Solr patches

`patches/solr.json` uses the RAPIDS CMake package-override schema. Each entry names
its diff (`file`), explains why it is needed (`issue`), gives an upstream release
boundary when known (`fixed_in`), and marks whether application must succeed
(`required`). Keep `fixed_in` empty until an upstream release containing the fix
is known; add the upstream issue/PR link to `issue` when one is available.

The `patch` phase configures this directory's CMake project with no compiler or
CUDA detection. It uses cuVS's existing RAPIDS CMake bootstrap, registers the
manifest with `rapids_cpm_package_override`, and executes the generated patch
command against the Solr checkout on the CPU runner. Gradle performs the CPU
build and exports the test configuration used by the GPU runner.
RAPIDS CMake recognizes already-applied `.diff` files; required patches that
cannot apply fail the phase. Upstream drift therefore requires reviewing or
retiring the patch, rather than silently proceeding without the fix.

The initial patch updates the removed vectors-format constructor to
`AcceleratedHNSWParams.Builder`, uses
`ThreadLocalCuVSResourcesProvider.isSupported()`, makes missing GPU support fail
when CI requests it, and updates the codec-name assertion for Lucene 10.4.
It was checked against Solr `d8ae2a03785c78840ee8dc8bf7e600f9d7b57d01`.
Remove the entry and diff once equivalent changes are upstream, or set `fixed_in`
to the first fixed Solr version if older supported refs still need patching.

Results include the original Solr SHA, manifest, exact patch files, RAPIDS patch
logs, and resulting source diff. This records the modifications alongside the
base revision even when a patch fails. To validate patching alone:

```sh
cmake -S ci/solr -B /tmp/solr-patches -DSOLR_SOURCE_DIR=/absolute/path/to/solr
```

## GPU checks

The GPU test phase requires `nvidia-smi` and runs the CPU-compiled `CheckGpu`
class against the downloaded Java JAR and native library. The integration test must execute `testIndexAndSearch`
successfully: absent, skipped, failing, or CPU-fallback results are errors.
The test runs with one fork, native access enabled by Solr's test configuration,
and its Java SecurityManager disabled for this native-library integration.

The exported test configuration and GPU runner set
`solr.tests.cuvs.required=true` on test JVMs. A Solr test
supporting this option should assert GPU availability instead of skipping. The
JUnit validator additionally rejects skipped tests, including older Solr tests
that do not recognize the option.

## Local checks

Run the helper's regression tests without Java, CUDA, or network access:

```sh
python -m unittest discover -s ci/solr/tests -v
```

For the CPU job, use a RAPIDS CI container with the tools used by the existing
Java/Lucene jobs. Set `CUVS_SOURCE_DIR`, `SOLR_SOURCE_DIR`, `SOLR_WORK_DIR`,
`SOLR_RESULTS_DIR`, `RAPIDS_REPOSITORY`, `RAPIDS_REF_NAME`, `RAPIDS_CUDA_VERSION`,
`RAPIDS_BUILD_TYPE=nightly`, and optionally `SOLR_BUILD_RUN_ID`. Run `setup`,
`patch`, and `build`, followed by `SOLR_REPORT_STAGE=build ci/test_solr.sh report`.

Transfer `SOLR_WORK_DIR/solr-test-bundle.tar.gz` to a fresh GPU job's work directory
using the same CUDA/OS container configuration. Run `restore`, `test`, and `report`
with that job's work/results directories. The GPU job only needs the CI checkout
and bundle; it does not need cuVS or Solr source checkouts. Use fresh work and
results directories for each job.
