---
slug: user-guide/field-guide/jit-compilation
---

# JIT Compilation

NVIDIA cuVS uses Just-in-Time (JIT) [Link-Time Optimization (LTO)](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/) compilation technology to compile certain kernels. When JIT compilation is triggered, NVIDIA cuVS compiles the kernel for your architecture and automatically caches it in memory and on disk.

The cache validity is:

1. In-memory cache: lifetime of the process.
2. On-disk cache: until a CUDA driver upgrade is performed. The cache can be shared between machines through network or cloud storage, and we recommend storing it in a persistent location. For more details on configuring the on-disk cache, see the CUDA documentation on [JIT Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#jit-compilation). The most relevant environment variables are `CUDA_CACHE_PATH` and `CUDA_CACHE_MAX_SIZE`.

JIT compilation is a one-time cost for a given kernel configuration. After the first compilation, you should not expect a steady-state performance loss. For latency-sensitive workflows, run a warmup step before the actual workload so the relevant kernels are compiled and cached ahead of time.

The following public NVIDIA cuVS C++ APIs currently trigger JIT compilation. The search entries include single-GPU overloads and multi-GPU overloads where those overloads are exposed.

- [cuvs::neighbors::cagra::search()](/api-reference/cpp-api-neighbors-cagra)
- [cuvs::neighbors::ivf_flat::search()](/api-reference/cpp-api-neighbors-ivf-flat)
- [cuvs::neighbors::ivf_pq::search()](/api-reference/cpp-api-neighbors-ivf-pq)
- [cuvs::neighbors::ivf_sq::search()](/api-reference/cpp-api-neighbors-ivf-sq)

Custom distance metrics (UDFs) for IVF-flat search also use JIT compilation. See [UDF Usage](/user-guide/field-guide/udf-usage).

For implementation details on building JIT LTO kernel fragments and linking them at runtime, see [Link-time Optimization](/developer-guide/advanced-topics/link-time-optimization).
