---
slug: developer-guide/coding-guidelines/cpp-guidelines
---

# C++ Guidelines

This page collects the engineering conventions that keep NVIDIA cuVS APIs stable, predictable, and easy to maintain. Start with the [Contributor Guide](/developer-guide/contributing), then use this page when designing public APIs, writing CUDA/C++ implementation code, or preparing a change for review.

## Local Development

Most NVIDIA cuVS changes can be developed directly in this repository. Cross-project CUDA/C++ work may also require a local RAFT build or temporary downstream pin.

If a consuming project supports source builds, pass `CPM_raft_SOURCE=/path/to/raft/source` to its CMake configuration. If the downstream project must pin a RAFT branch while related changes are under review, update the `FORK` and `PINNED_TAG` arguments to `find_and_configure_raft`, then revert that pin before the downstream change merges.

If source builds are not being used, install the local RAFT C++ artifacts into the consuming project's environment before testing the downstream change.

## Public Interface

### General Guidelines

Public C++ APIs should be stateless wrappers around implementation code in a private `detail` namespace.

Expose only lightweight, predictable types:

1. Plain data structs used for parameters or metadata.
2. `raft::resources`, because it owns execution resources rather than algorithm state.
3. `raft::span` and `raft::mdspan` views for single- and multi-dimensional data.
4. `std::optional` for optional values instead of sentinel pointers.

Prefer references for required inputs. Reserve pointers for established output patterns and avoid exposing temporary implementation classes in public headers.

### API Stability

Public APIs are consumed by multiple projects and should change carefully. Add new APIs before removing old ones, deprecate old entry points over a few releases, and avoid changing behavior in ways that downstream users cannot detect at compile time.

### Stateless C++ APIs

Avoid public APIs that store algorithm state in non-POD wrapper objects:

```cpp
class ivf_pq_float {
  ivf_pq::index_params params_;
  raft::resources const& res_;

 public:
  ivf_pq_float(raft::resources const& res);

  void train(raft::device_matrix_view<const float, int64_t, raft::row_major> dataset);

  void search(raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<float, int64_t, raft::row_major> distances);
};
```

Prefer stateless, instantiated overloads for the supported type combinations. Template implementations can still live in `detail`, but public entry points should be concrete:

```cpp
namespace cuvs::neighbors::ivf_pq {

auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> index<int64_t>;

void build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           index<int64_t>* idx);

void search(raft::resources const& res,
            search_params const& params,
            index<int64_t> const& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

// Add supported variants, such as half or int8_t, as separate overloads.
}
```

### Functions On State

When an API creates an index or model object, also expose stateless functions for persistence and transfer. Keep those functions in the same public namespace as the owning algorithm:

```cpp
namespace cuvs::neighbors::ivf_pq {

void serialize(raft::resources const& res, std::ostream& os, index<int64_t> const& index);

void deserialize(raft::resources const& res, std::istream& is, index<int64_t>* index);

}  // namespace cuvs::neighbors::ivf_pq
```

### Working With Dense Arrays

Use RAFT array types consistently in public C++ APIs and implementation code. For user-facing examples of passing dense arrays into NVIDIA cuVS APIs, see [Dense Arrays](/user-guide/api-guides/core-types/array-types/dense-arrays).

Keep data layout explicit. Most NVIDIA cuVS APIs expect row-major dense matrices unless the API says otherwise.

Prefer public API signatures that accept non-owning views, not owning arrays. This keeps NVIDIA cuVS functions flexible because callers can pass memory owned by RAFT, another RAPIDS library, or an application-specific allocator.

Prefer `device_matrix_view` and `device_vector_view` over raw pointers for public NVIDIA cuVS C++ APIs. Views carry shape, layout, memory-space, and constness information, making incorrect dimensions and accidental writes easier to catch.

Use owning arrays such as `raft::device_matrix`, `raft::host_matrix`, and `raft::pinned_matrix` when implementation code, tests, or examples need RAFT to allocate storage.

Synchronize appropriately before reading data on the host. Many NVIDIA cuVS operations enqueue GPU work asynchronously on the stream owned by `raft::device_resources`.

Use pinned arrays only when their transfer or coordination benefits matter. Ordinary host arrays are simpler and should be the default for CPU-only data.

## Common Design Considerations

1. Use `.hpp` for headers that can be compiled by `gcc` against the CUDA runtime. Use `.cuh` when a header requires `nvcc`.
2. Keep public types simple. They should store state, not perform computation.
3. Document every public API with a clear summary, parameter descriptions, and a short usage example when helpful.
4. Before adding a primitive, check whether an existing primitive can be extended cleanly. Add a new public API only when the behavior is genuinely distinct.

### Performance

Prefer small, explicit choices that avoid hidden overhead:

1. Use `cudaDeviceGetAttribute` instead of `cudaDeviceGetProperties` in performance-critical code. See the CUDA developer blog post on [fast device property queries](https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/).
2. Reuse the stream pool on the provided `raft::resources` object instead of creating one `raft::resources` object per stream. See [Threading Model](#threading-model) and [Resource Management](#resource-management).
3. Keep CPU work around GPU launches light. If host threads are used, they should coordinate CUDA streams, not perform heavy CPU computation.

### Threading Model

NVIDIA cuVS algorithms should be safe to call from multiple host threads when each thread uses its own `raft::resources` instance. Treat `raft::resources` as the boundary for CUDA streams, memory resources, communication handles, and library handles.

Inside an algorithm, host threads are acceptable only when they help keep CUDA streams busy. Keep them bounded, prefer OpenMP, and make sure the algorithm still works when OpenMP is disabled.

```cpp
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>

void run_batches(raft::resources const& res, int n_batches)
{
  auto main_stream = raft::resource::get_cuda_stream(res);
  raft::resource::sync_stream(res, main_stream);

#pragma omp parallel for
  for (int i = 0; i < n_batches; ++i) {
    auto stream = raft::resource::get_stream_from_stream_pool(res);

    // Keep host work here light. The thread exists to drive GPU work.
    preprocess_batch(i);
    my_kernel<<<blocks, threads, 0, stream>>>(i);
    postprocess_batch(i);
  }

  raft::resource::sync_stream_pool(res);
}
```

If there is no CPU work before the first kernel, make the internal streams wait on the main stream with CUDA events. If there is no CPU work after each batch, synchronize the stream pool once after the loop instead of synchronizing inside every iteration.

### Asynchronous Operations And Stream Ordering

NVIDIA cuVS algorithms should be asynchronous whenever possible and should avoid the default CUDA stream. For single-stream work, use the stream on `raft::resources`:

```cpp
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

void foo(raft::resources const& res)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);
}
```

When an algorithm uses internal streams, preserve ordering with the caller's stream:

1. Work already queued on `raft::resource::get_cuda_stream(res)` must complete before internal stream work starts.
2. Work queued by the caller after the API returns must wait until all internal stream work is complete.

Use CUDA events and `cudaStreamWaitEvent` to create those dependencies. This lets users compose NVIDIA cuVS operations with their own asynchronous copies and kernels without accidental races.

#### Using Thrust

Run Thrust algorithms on the intended stream and memory resource by using the execution policy from `raft::resources`:

```cpp
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>

void foo(raft::resources const& res)
{
  auto policy = raft::resource::get_thrust_policy(res);
  thrust::for_each(policy, first, last, op);
}
```

### Resource Management

Do not create reusable CUDA resources directly inside algorithm implementations. Reuse the handles, streams, events, allocators, and library resources attached to `raft::resources`. If a reusable resource is missing, file an issue or feature request instead of creating a local long-lived resource.

```cpp
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>

void foo(raft::resources const& res)
{
  cublasHandle_t cublas_handle = raft::resource::get_cublas_handle(res);
  auto stream                  = raft::resource::get_stream_from_stream_pool(res);
}
```

Users can configure the stream pool once and pass the same `raft::resources` object through the API:

```cpp
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <rmm/cuda_stream_pool.hpp>

int main()
{
  raft::resources res;
  raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(4));

  foo(res);
}
```

### Multi-GPU

NVIDIA cuVS multi-GPU APIs generally use one of two execution strategies:

1. Single-process multi-GPU: one host process owns and coordinates multiple GPUs. Use `raft::device_resources_snmg` in examples and public API documentation for this model. It owns the selected devices, streams, communication resources, and per-device execution state.
2. One-process-per-GPU: each process owns one GPU and participates as a communication rank. Use `raft::device_resources` or `raft::resources` with an initialized `raft::comms::comms_t`. Access communication through `raft::resource::get_comms`, and guard optional communication paths with `raft::resource::comms_initialized`.

APIs may support either strategy or both, but they should document which resource object users are expected to construct. Keep streams, stream pools, memory resources, and communicators attached to the supplied resource object instead of creating unmanaged process-wide state. Single-GPU APIs should not depend on communication libraries or require multi-GPU resources.

For one-process-per-GPU implementations, developers can assume:

1. `raft::comms::comms_t` has been initialized correctly.
2. All participating ranks call the multi-GPU algorithm cooperatively.

Access the communicator through `raft::resources`:

```cpp
#include <raft/core/resource/comms.hpp>
#include <raft/core/resources.hpp>

void foo(raft::resources const& res)
{
  auto const& comm = raft::resource::get_comms(res);
  int rank         = comm.get_rank();
  int size         = comm.get_size();
}
```

### Using Just-in-Time Link-Time Optimization

NVIDIA cuVS is moving new kernels toward JIT link-time optimization. Instead of compiling every kernel variant into the binary, JIT LTO compiles fragments and links the needed combination at runtime.

This helps reduce binary size and enables user-defined functions in NVIDIA cuVS CUDA kernels. For runtime and cache behavior, see [JIT Compilation](/user-guide/field-guide/jit-compilation). For implementation guidance, see [Link-time Optimization](/developer-guide/advanced-topics/link-time-optimization).

## Coding Style

### Formatting

NVIDIA cuVS uses [pre-commit](https://pre-commit.com/) to run formatting, linting, spelling, and copyright checks. Install it with conda:

```bash
conda install -c conda-forge pre-commit
```

Run checks before committing:

```bash
pre-commit run
```

Run the full suite across the repository when needed:

```bash
pre-commit run --all-files
```

You can also install the git hook:

```bash
pre-commit install
```

### Core Hooks

C++ and CUDA code are formatted with [clang-format](https://clang.llvm.org/docs/ClangFormat.html). NVIDIA cuVS follows the Google C++ style with a few local adjustments documented in [cpp/.clang-format](https://github.com/rapidsai/cuvs/blob/main/cpp/.clang-format):

1. Empty functions, records, and namespaces are not split.
2. Indentation is two spaces, including line continuations.
3. Comments are not reflowed automatically.

[Doxygen](https://doxygen.nl/) checks C++ and CUDA API documentation:

```bash
./ci/checks/doxygen.sh
```

[codespell](https://github.com/codespell-project/codespell) catches spelling issues. To apply suggested fixes interactively, run:

```bash
codespell -i 3 -w .
```

### Include Style

Use `#include "..."` only for local files in the same algorithm or nearby directory. Use `#include <...>` for dependencies, primitives, and headers from other algorithms.

To bulk-fix include style issues, run:

```bash
python ./cpp/scripts/include_checker.py --inplace cpp/include cpp/tests
```

### Copyright

RAPIDS pre-commit hooks check copyright headers on modified tracked files. To run that check manually:

```bash
pre-commit run -a verify-copyright
```

## Code Quality

### Testing

Public APIs need direct test coverage because downstream projects rely on their compile-time and runtime behavior. Prefer tests that exercise the public entry point, cover edge cases, and make the expected behavior visible without requiring downstream projects to catch regressions first.

### Performance Benchmarking

The most important implementation details in NVIDIA cuVS are written in C++ and CUDA, so performance-sensitive changes need benchmarks with clear baselines. Benchmarks should show that regressions have not occurred, intended improvements can be reproduced consistently, and the implementation scales as expected across problem sizes and hardware configurations. For relevant indexing APIs, it is often preferable to use the [cuVS Bench Tool](/user-guide/benchmarking-guide/cu-vs-bench-tool) for reproducible benchmarks.

Attach benchmark results to the relevant GitHub pull request so future reviewers and contributors have an audit trail. Pull requests that change performance-critical code should not be merged without proper benchmarks in place.

For multi-GPU APIs, include scaling measurements whenever the change affects communication, partitioning, synchronization, or resource use across ranks. A change that improves a single-GPU path should not silently reduce multi-GPU efficiency.

### Error Handling

Call CUDA and library APIs through the RAFT helper macros, such as `RAFT_CUDA_TRY`, `RAFT_CUBLAS_TRY`, and `RAFT_CUSOLVER_TRY`. They check return values and throw on failure.

Use the `_NO_THROW` variants only where throwing is unsafe, such as destructors. Those variants log errors without throwing.

### Documentation

Public C++ and CUDA APIs require user-facing [Doxygen](https://doxygen.nl/) documentation. Document the purpose, parameters, return values, relevant template or overload behavior, and any constraints that affect correct use.
