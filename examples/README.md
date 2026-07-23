# cuVS C and C++ Examples

This template project provides a drop-in sample to either start building a new application with, or using CUVS in an existing CMake project.

First, please refer to our [installation docs](https://docs.rapids.ai/api/cuvs/stable/build.html#cuda-gpu-requirements) for the minimum requirements to use cuVS.

Once the minimum requirements are satisfied, this example template application can be built with the provided `build.sh` script. This is a bash script that calls the appropriate CMake commands, so you can look into it to see the typical CMake based build workflow.

The directories (`CUVS_SOURCE/examples/c`) or (`CUVS_SOURCE/examples/cpp`) can be copied directly in order to build a new application with cuVS.

cuVS can be integrated into an existing CMake project by copying the contents in the `configure rapids-cmake` and `configure cuvs` sections of the provided `CMakeLists.txt` into your project, along with `cmake/thirdparty/get_cuvs.cmake`.

Make sure to link against the appropriate CMake targets. Use `cuvs::c_api` and `cuvs::cuvs` to use the C and C++ shared libraries respectively.

```cmake
target_link_libraries(your_app_target PRIVATE cuvs::cuvs)
```

## Balanced k-means example

`BALANCED_KMEANS_EXAMPLE` partitions a vector database with cuVS balanced k-means. Specify the
dataset path with `-d`, its data type with `-t`, and the desired number of partitions with `-P`:

```bash
./cpp/build/BALANCED_KMEANS_EXAMPLE -d vectors.bin -t float -P 256 -I 20 -L 0.333,0.5 -U 2.0,3.0 -O 0.01
```

The supported data types are `float`, `half`, `int8`, and `uint8`. The dataset can use the BIGANN
format (`uint32` vector count, `uint32` dimension count, then row-major vectors) or the xvec format.
Use `-I` to set the number of k-means iterations; it defaults to 20. Use `-L` to set one or more
lower balance tolerances, `-U` to set one or more upper balance tolerances, and `-O` to set the
centroid offset used when splitting large partitions; they default to 0.333, 3.0, and 0.01. The
example runs balanced k-means for every `-L` and `-U` combination. The defaults target partitions
outside roughly one-third to three times the average partition size. Very strict upper tolerance
values around 1.4 or lower can be difficult for this heuristic rebalancing method to satisfy. The
example prints partition size statistics, underflow/overflow counts, and histograms comparing
regular k-means and balanced k-means for `float` input.
