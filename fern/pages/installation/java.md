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
