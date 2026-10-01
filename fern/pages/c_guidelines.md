---
slug: developer-guide/coding-guidelines/c-guidelines
---

# C Guidelines

This page collects the engineering conventions that keep NVIDIA cuVS C APIs stable, predictable, and easy to use from downstream projects and language bindings. Start with the [Contributor Guide](/developer-guide/contributing), then use this page when designing public C APIs, C wrappers, or C-facing documentation.

## Local Development

Most C API changes can be developed directly in this repository. Cross-project work may also require a local RAFT build or a downstream project that consumes the installed NVIDIA cuVS C headers and shared libraries.

If source builds are not being used, install the local NVIDIA cuVS C artifacts into the consuming project's environment before testing the downstream change.

## Public Interface

### General Guidelines

Public C APIs should be thin, ABI-stable wrappers around NVIDIA cuVS implementation code. Keep C headers free of C++ types, templates, namespaces, exceptions, and RAII-only ownership patterns.

Expose only C-compatible types:

1. Opaque handles for resources, indexes, models, and parameter objects.
2. Plain C structs for lightweight metadata and simple value groups.
3. `cuvsError_t` return values for API status.
4. Explicit pointer outputs for objects created by the API.

Prefer explicit create and destroy functions for every opaque object that owns memory or other resources.

### API Stability

The C API is the stable boundary used by downstream integrations and NVIDIA cuVS language bindings. Add new functions or fields before removing old ones, avoid changing the meaning of existing parameters, and keep [ABI compatibility](/developer-guide/advanced-topics/abi-stability) in mind when changing public structs or exported symbols.

### Stateless C APIs

Prefer stateless functions that take all required state explicitly:

```c
cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
                           cuvsIvfPqIndexParams_t params,
                           DLManagedTensor* dataset,
                           cuvsIvfPqIndex_t index);

cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
                            cuvsIvfPqSearchParams_t params,
                            cuvsIvfPqIndex_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances);
```

Avoid APIs that hide global state, allocate persistent internal state without an owning handle, or require callers to understand C++ object lifetimes.

### Functions On State

When a C API creates an index, model, resource, or parameter object, expose matching operations for lifecycle and persistence in the same API family:

```c
cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index);
cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index);

cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res,
                               const char* filename,
                               cuvsIvfPqIndex_t index);

cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res,
                                 const char* filename,
                                 cuvsIvfPqIndex_t index);
```

## Common Design Considerations

1. Use `.h` for public C headers and keep them consumable from both C and C++.
2. Wrap declarations with `extern "C"` guards when a header can be included from C++.
3. Mark exported public functions with `CUVS_EXPORT`.
4. Keep ownership explicit: the API that creates an object should document the matching destroy function.
5. Use DLPack tensors for array inputs and outputs where possible so callers can pass data across language and framework boundaries.

### Performance

Keep C wrappers thin. Validate inputs and translate handles at the boundary, but leave expensive work in the underlying NVIDIA cuVS implementation.

Avoid hidden host-device copies and hidden synchronization. If a wrapper needs to synchronize, document that behavior clearly.

### Threading Model

C APIs should be safe to call from multiple host threads when each thread uses its own `cuvsResources_t` instance. Treat `cuvsResources_t` as the boundary for streams, memory resources, communication handles, and library handles.

Avoid mutable process-wide state in C wrappers. If shared state is unavoidable, make ownership and synchronization explicit.

### Asynchronous Operations And Stream Ordering

C APIs should preserve the stream-ordering behavior of the underlying NVIDIA cuVS implementation. Do not add hidden synchronization only to simplify wrapper code.

When a C function accepts `cuvsResources_t`, use the stream and resources associated with that handle. Work queued by the caller before the API should complete before internal work starts, and work queued by the caller after the API returns should wait for internal work that affects the result.

### Resource Management

Every successful create call that returns an owning handle should have a matching destroy call in examples and tests:

```c
cuvsResources_t res;
cuvsIvfPqIndexParams_t params;
cuvsIvfPqIndex_t index;

cuvsResourcesCreate(&res);
cuvsIvfPqIndexParamsCreate(&params);
cuvsIvfPqIndexCreate(&index);

/* Use res, params, and index. */

cuvsIvfPqIndexDestroy(index);
cuvsIvfPqIndexParamsDestroy(params);
cuvsResourcesDestroy(res);
```

Destroy functions should tolerate cleanup after partial setup when practical, and examples should release resources in the reverse order they were acquired.

### Multi-GPU

Multi-GPU C APIs generally use one of two execution strategies:

1. Single-process multi-GPU: one host process owns and coordinates multiple GPUs. Use `cuvsMultiGpuResourcesCreate` or `cuvsMultiGpuResourcesCreateWithDeviceIds` for this model. These functions return a `cuvsResources_t` backed by `raft::device_resources_snmg`; release it with `cuvsMultiGpuResourcesDestroy`.
2. One-process-per-GPU: each process owns one GPU and participates as a communication rank. APIs that wrap this model should accept the process-local `cuvsResources_t` for the calling rank and document how communicator setup, rank ownership, and synchronization are provided by the underlying implementation.

APIs may support either strategy or both, but they should document which resource handle users are expected to create. Use `cuvsMultiGpuResourcesSetMemoryPool` to configure an RMM memory pool across devices managed by a single-process multi-GPU resource. Continue to use `cuvsStreamSet`, `cuvsStreamGet`, and `cuvsStreamSync` for stream ownership and synchronization through the supplied opaque resource handle.

Single-GPU C APIs should not require communication libraries or multi-GPU setup.

### Using Just-in-Time Link-Time Optimization

C APIs may call implementations that use JIT link-time optimization, but the C wrapper should not duplicate JIT LTO policy or expose C++ implementation details. Keep runtime behavior documented at the API level when JIT compilation can affect first-call latency or cache behavior.

For runtime and cache behavior, see [JIT Compilation](/user-guide/field-guide/jit-compilation). For implementation guidance, see [Link-time Optimization](/developer-guide/advanced-topics/link-time-optimization).

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

C headers and C wrapper implementation files are checked by the same formatting, spelling, Doxygen, and copyright hooks used by the rest of NVIDIA cuVS.

Run Doxygen checks for public C API documentation:

```bash
./ci/checks/doxygen.sh
```

[codespell](https://github.com/codespell-project/codespell) catches spelling issues. To apply suggested fixes interactively, run:

```bash
codespell -i 3 -w .
```

### Include Style

Use `#include <cuvs/...>` for public NVIDIA cuVS C headers. Keep public C headers minimal and avoid including private C++ implementation headers from the public C interface.

### Copyright

RAPIDS pre-commit hooks check copyright headers on modified tracked files. To run that check manually:

```bash
pre-commit run -a verify-copyright
```

## Code Quality

### Testing

Public C APIs need direct test coverage because downstream projects and language bindings rely on their runtime and ABI behavior. Prefer tests that exercise public entry points, lifecycle functions, error paths, and resource cleanup.

### Error Handling

C APIs should return `cuvsError_t` and should not let C++ exceptions cross the C boundary. Translate implementation failures into C error values and make sure callers can retrieve useful diagnostic information when available.

Destroy functions should avoid throwing or failing in ways that make cleanup unsafe.

### Documentation

Public C APIs require user-facing Doxygen documentation. Document the purpose, parameters, return values, ownership rules, matching destroy functions, and any constraints that affect correct use.
