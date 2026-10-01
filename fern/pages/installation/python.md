# Python Installation

Use this page when you need the NVIDIA cuVS Python package.

All NVIDIA cuVS routine implementations live in the C++ core. The Python bindings call into the native C and C++ libraries, so environments that use shared libraries need both `libcuvs_c` and `libcuvs` installed. The pip wheels bundle these native libraries for Python use.

## Install Pre-Compiled Packages

Install the Python package with conda:

```bash
# CUDA 13
conda install -c rapidsai -c conda-forge cuvs cuda-version=13.3

# CUDA 12
conda install -c rapidsai -c conda-forge cuvs cuda-version=12.9
```

You can also install through pip:

```bash
# CUDA 13
pip install cuvs-cu13

# CUDA 12
pip install cuvs-cu12
```

The pip packages statically link the C and C++ libraries, so `libcuvs` and `libcuvs_c` shared libraries are not readily available for use by external code.

## Build From Source

Before building from source, review the [shared C++ source-build prerequisites](/installation#build-from-source), including the recommended conda environment setup for build dependencies.

Build and install the Python package with:

```bash
./build.sh python
```

Uninstall the Python package with:

```bash
./build.sh python --uninstall
```

If the Python changes depend on native C or C++ changes, rebuild the native libraries first:

```bash
./build.sh libcuvs python
```
