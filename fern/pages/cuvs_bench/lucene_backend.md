---
slug: user-guide/benchmarking-guide/cu-vs-bench-tool/lucene-backend
---

# Lucene Runtime

The optional Lucene runtime adapters embed the JVM through PyLucene. This page
covers runtime setup; registering a selectable benchmark backend is a separate
integration step.

## Runtime requirements

The optional runtime is not provisioned by the ordinary cuVS Bench installation.
It requires the custom PyLucene 10.2.0 build and JDK 22. Native cuVS integrations
also require matching base `cuvs-java` and thin `cuvs-lucene-10.2` JARs. Both
JAR versions must match the cuVS Bench release. Select the 10.2 artifact for
this PyLucene runtime, not `cuvs-lucene-10.3`. GPU execution additionally
requires compatible native cuVS, CUDA, and a supported NVIDIA GPU.

The runtime resolver discovers artifacts built by the current checkout or
installed in the local Maven repository. For nonstandard locations, set both
`CUVS_LUCENE_CUVS_JAVA_JAR` and `CUVS_LUCENE_JAR`; use `JAVA_LIBRARY_PATH` for
native libraries. An explicit native-library path must contain an unversioned
`libcuvs_c.so`.

Do not use the cuVS-Lucene JAR assembled with dependencies: PyLucene already
provides Lucene classes, and loading a second copy can make the embedded JVM
classpath inconsistent.

### Build PyLucene 10.2.0 from source

The ordinary cuVS Bench wheel, conda package, and container do not include the
custom PyLucene 10.2.0 runtime. The helper below is available only in a cuVS
source checkout.

Previous validation used Linux x86_64 with CPython 3.14 and JDK 22.
A PyLucene wheel is specific to its operating system, architecture, Python ABI,
and JDK toolchain; do not attach or redistribute this local wheel as a general
binary. No real ARM64 build or execution has been performed.

Start in the normal cuVS source-build environment described in the
[shared source-build prerequisites](/installation#build-from-source). Also
install the [Java build prerequisites](/installation/java#build-from-source),
CPython 3.12-3.14 with development headers and `venv` support, GNU Make, a C/C++
compiler, `awk`, `curl`, `flock`, `gzip`, `patch`, `tar`, GNU coreutils, and GNU
findutils. CUDA and a supported NVIDIA GPU are required for GPU execution.

For native cuVS integrations, start at the repository root and, before
activating the isolated PyLucene environment, build matching native cuVS,
base `cuvs-java`, and thin `cuvs-lucene` artifacts. Standalone CPU PyLucene does
not need native cuVS or either JAR, so CPU-only users can skip this command and
the native-library setup below, but should retain the documented cuVS build
environment for the editable install.

```bash
export JAVA_HOME=/absolute/path/to/jdk-22
./build.sh libcuvs java lucene
```

Choose a stable absolute PyLucene build location outside `/tmp`. Do not move
the completed directory or selected JDK, and do not remove the base Python
installation: JCC and the virtual environment retain absolute paths. Allow at
least 3 GB of disk space.

```bash
export PYLUCENE_BUILD_ROOT="$HOME/.local/share/cuvs/pylucene-10.2.0"

conda/recipes/cuvs-bench/build_pylucene_10_2.sh \
    --python python3 \
    --build-root "$PYLUCENE_BUILD_ROOT"

source "$PYLUCENE_BUILD_ROOT/activate.sh"
python -m pip install -e ./python/cuvs_bench
python -m pip check
```

The helper verifies checksum-pinned Apache PyLucene 10.0.0 scaffolding, Lucene
10.2.0 sources, and the Gradle distribution. It applies the tracked
compatibility patch, builds JCC 3.15 and a Python-ABI-specific PyLucene wheel in
an isolated virtual environment, runs the upstream PyLucene tests, and performs
a JVM class-loading smoke test. The Python packages requested by the helper are
version-pinned but not hash-locked, and Gradle dependencies remain
network-resolved, so this is not a hermetic or bit-for-bit-reproducible build.

Verify that the activated interpreter uses the expected runtime:

```bash
python - <<'PY'
import os
from pathlib import Path

import lucene

build_root = Path(os.environ["PYLUCENE_BUILD_ROOT"]).resolve()
module_path = Path(lucene.__file__).resolve()
assert lucene.VERSION == "10.2.0", lucene.VERSION
assert module_path.is_relative_to(build_root), module_path
print(f"PyLucene {lucene.VERSION} from {module_path}")
PY
```

The runtime does not require the `bench-ann` target. Make the
fresh source-build libraries visible to both Java and the ELF loader:

```bash
CUVS_NATIVE_BUILD="$(cd cpp/build && pwd -P)"
export JAVA_LIBRARY_PATH="$CUVS_NATIVE_BUILD/c:$CUVS_NATIVE_BUILD"
export LD_LIBRARY_PATH="$JAVA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Keep CUDA's runtime directory loader-visible through the normal cuVS build
environment. Some conda toolchains encode their active prefix as `DT_RPATH`,
which takes precedence over `LD_LIBRARY_PATH`. The standard `./build.sh`
command installs the freshly built libraries into that active prefix. If you
override `INSTALL_PREFIX`, verify that the C wrapper resolves matching cuVS
libraries and relocations rather than an older installation:

```bash
ldd -r "$CUVS_NATIVE_BUILD/c/libcuvs_c.so" \
    | grep -E 'libcuvs|librmm|librapids_logger|undefined symbol'
```

The runtime resolver discovers the base JAR under `java/cuvs-java/target/` and the
matching thin JAR under `java/cuvs-lucene/lucene-10.2/target/`. It also supports
the corresponding local Maven artifacts, including
`com.nvidia.cuvs.lucene:cuvs-lucene-10.2`. Do not select the native-classifier
`cuvs-java` JAR or the cuVS-Lucene `jar-with-dependencies`.

In each new shell, reactivate the normal cuVS source-build environment before
sourcing `activate.sh`, then restore the two native-library path variables
before starting Python. PyLucene's JVM is process-global, so its classpath and
JVM arguments cannot be changed after `lucene.initVM(...)`.
Let these runtime adapters initialize PyLucene; if PyLucene was initialized
outside them, use a fresh Python process.

### Java search timing

When initialized with a validated cuvs-java and cuvs-lucene artifact pair, the
runtime uses the thin JAR's `IndexSearcherTimingBridge`. It measures
`System.nanoTime()` around exactly `IndexSearcher.search(Query, int)` inside
the JVM, excluding Python/JCC dispatch and result conversion. The artifact-free
CPU runtime calls PyLucene directly and does not provide this Java measurement.
