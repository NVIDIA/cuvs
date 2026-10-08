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

`cuvs-lucene` targets Lucene 10.2, requires JDK 22 and Maven 3.9.6 or newer, and inherits the NVIDIA cuVS [CUDA GPU requirements](/installation#cuda-gpu-requirements).

`Lucene101AcceleratedHNSWCodec` falls back to CPU index construction when no GPU or native library is available, so an application using it works on GPU and non-GPU hosts alike. The other three codecs require a working cuVS installation. See [Lucene Integration](/user-guide/lucene) for details.

### Add the Maven Dependency

To use `cuvs-lucene` in a Maven project, add the following dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene</artifactId>
  <version>26.10.0</version>
</dependency>
```

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

The resulting artifacts are written to `java/cuvs-lucene/target`.

Add `--run-java-tests` to any of these commands to run the test suite, and `--build-java-examples` to build the example projects against the jars that were just built. Each target builds only its own examples: the `lucene` target builds `examples/java/cuvs-lucene`, and the `java` target builds `examples/java/cuvs-java`.
