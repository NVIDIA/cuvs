# Java Installation

Use this page when you need the NVIDIA cuVS Java API. The Java API uses Panama bindings and requires matching native NVIDIA cuVS libraries at runtime.

All NVIDIA cuVS routine implementations live in the C++ core. The Java bindings call into the native C and C++ libraries, so install both `libcuvs_c` and `libcuvs`.

## Install Native Dependencies

Install the native C and C++ libraries first. For most users, conda is the simplest option:

```bash
# CUDA 13
conda install -c rapidsai -c conda-forge libcuvs cuda-version=13.3

# CUDA 12
conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.9
```

If you use locally built native libraries, make sure the directory containing `libcuvs.so` and `libcuvs_c.so` is on `LD_LIBRARY_PATH`.

Java development also requires:

1. Maven 3.9.6 or newer.
2. JDK 22.
3. jextract for JDK 22. If it is not already installed, the build script downloads it.

## Build From Source

Before building from source, review the [shared C++ source-build prerequisites](/installation#build-from-source), including the recommended conda environment setup for build dependencies.

Build the native libraries and Java API from the repository root:

```bash
./build.sh libcuvs java
```

If matching native libraries are already built and installed, build only the Java API:

```bash
./build.sh java
```

You can also build from the `java` directory:

```bash
cd java
./build.sh
```

Run Java integration tests from the `java` directory:

```bash
./build.sh --run-java-tests
```

Run a focused test suite from `java/cuvs-java`:

```bash
mvn clean integration-test -Dit.test=com.nvidia.cuvs.CagraBuildAndSearchIT
```

If the C headers used by Java change, regenerate the Panama bindings:

```bash
java/panama-bindings/generate-bindings.sh
```

## cuVS Lucene

`cuvs-lucene` provides Apache Lucene codecs that offload vector index build, and optionally search, to the GPU. It is published as a separate artifact that builds on the Java API described above. For usage guidance, see the [Lucene Integration](/user-guide/lucene) guide.

`cuvs-lucene` supports Lucene 10.2 through 10.5 and requires Maven 3.9.6 or newer. It compiles and runs on JDK 21, the minimum of Lucene 10, but uses the GPU only on a JDK 22 or newer runtime: on JDK 21 the accelerated HNSW codecs build graphs on the CPU and the GPU search codec is unavailable. It inherits the NVIDIA cuVS [CUDA GPU requirements](/installation#cuda-gpu-requirements).

The accelerated HNSW codec falls back to CPU index construction when no GPU or native library is available, so an application using it works on GPU and non-GPU hosts alike. The other three codecs require a working cuVS installation. See [Lucene Integration](/user-guide/lucene) for details.

### Add the Maven Dependency

Lucene changes its codec APIs between minor releases, so `cuvs-lucene` is published as one artifact per supported Lucene minor release. Pick the one that matches the Lucene version your application uses:

| Lucene | Artifact |
| --- | --- |
| 10.2.x | `cuvs-lucene-10.2` |
| 10.3.x | `cuvs-lucene-10.3` |
| 10.4.x | `cuvs-lucene-10.4` |
| 10.5.x | `cuvs-lucene-10.5` |

For example, for Lucene 10.5, add the following dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene-10.5</artifactId>
  <version>26.12.0</version>
</dependency>
```

Against any other Lucene minor release, an artifact's codecs fail when they are used, with an error naming the artifact to use instead, and the problem is logged at `SEVERE` the first time Lucene looks up its codecs. Call `CuVSCodecs.checkLuceneVersion()` at startup to fail early instead. Releases before 26.12 published a single `cuvs-lucene` artifact, built for Lucene 10.2. Its coordinates are relocated to `cuvs-lucene-10.2`, so Maven builds that only bump its version to 26.12 or later keep working and print a warning; switch to the artifact matching your Lucene version. Build tools other than Maven may not follow the relocation; update the artifact name there.

The native NVIDIA cuVS libraries are not bundled with the artifact. Install a matching version of `libcuvs` and `libcuvs_c` as described above, and make sure the directory containing them is on `LD_LIBRARY_PATH` before starting the JVM.

### Build cuVS Lucene From Source

If the native libraries and the Java bindings have not been built yet, build everything from the repository root:

```bash
./build.sh libcuvs java lucene
```

If the native libraries are already installed and the `cuvs-java` artifact is already in your local Maven repository, build only the codecs:

```bash
./build.sh lucene
```

You can also build from the `java/cuvs-lucene` directory:

```bash
cd java/cuvs-lucene
./build.sh
```

This builds and installs every `cuvs-lucene-10.X` artifact into the local Maven repository. The resulting artifacts are written to `java/cuvs-lucene/lucene-10.X/target`.

Add `--run-java-tests` to any of these commands to run the test suite, and `--build-java-examples` to build the example projects against the jars that were just built. Each target builds only its own examples: the `lucene` target builds `examples/java/cuvs-lucene`, and the `java` target builds `examples/java/cuvs-java`.
