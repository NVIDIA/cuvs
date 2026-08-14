---
slug: developer-guide/coding-guidelines/java-guidelines
---

# Java Guidelines

This page collects the conventions that keep the NVIDIA cuVS Java APIs predictable, resource-safe, and aligned with the native NVIDIA cuVS API. Start with the [Contributor Guide](/developer-guide/contributing), then use this page when changing Java APIs, Panama bindings, native library loading, Java packaging, or Java-facing documentation.

## Local Development

Most Java changes can be developed directly in this repository. The Java APIs depend on the native NVIDIA cuVS C library, so local testing usually requires matching native libraries to be built first.

Use `./build.sh libcuvs java` from the repository root when the native libraries need to be rebuilt. If matching native libraries are already available, use `./build.sh java` from the repository root or `./build.sh` from the `java` directory.

The Java package uses Maven, JDK 22 for the Panama-backed implementation, and a multi-release JAR. If the C headers used by Java change, regenerate the Panama bindings with `java/panama-bindings/generate-bindings.sh` before testing Java changes.

## Public Interface

### General Guidelines

Public Java APIs should be small Java-friendly wrappers around the NVIDIA cuVS C API. The Java layer should translate Java objects into native handles, arrays, matrices, streams, and parameters; heavy algorithmic work should stay in the native NVIDIA cuVS implementation.

Expose predictable Java types:

1. Interfaces for public resources, indexes, matrices, and query/result objects.
2. Builder APIs for objects that need several required and optional inputs.
3. Parameter classes and enums that map clearly to native C parameters.
4. `AutoCloseable` for every object that owns native memory, native handles, or other external resources.

Keep generated Panama bindings and native `MemorySegment` details out of the public `com.nvidia.cuvs` API. Public users should interact with stable Java abstractions such as `CuVSResources`, `CuVSMatrix`, `CagraIndex`, and `SearchResults`.

### API Stability

Java APIs are consumed by downstream applications and should change carefully. Add new methods or overloads before removing old ones, preserve existing builder behavior where possible, and avoid changing defaults in ways that silently alter search quality, memory use, or native resource ownership.

The Java bindings should call the NVIDIA cuVS C APIs rather than C++ or CUDA implementation details directly. The C layer is the ABI-stable boundary for bindings, so changes that require new native behavior should usually start with the C API. See [ABI Stability](/developer-guide/advanced-topics/abi-stability) for more detail.

### Resource Lifecycle

Use `AutoCloseable` for resources, matrices, indexes, temporary native allocations, and native handle wrappers. Examples and tests should use try-with-resources whenever practical:

```java
try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix();
    CagraIndex index =
        CagraIndex.newBuilder(resources)
            .withDataset(dataset)
            .withIndexParams(indexParams)
            .build()) {
  SearchResults results = index.search(query);
}
```

When an implementation creates a native handle, make the matching destroy call visible in `close()`. Close resources in the reverse order they were acquired, and guard against double-close or use-after-close when an object can outlive a native allocation.

### Native Boundary

Keep native calls behind implementation classes in `com.nvidia.cuvs.internal`. Public API classes should not expose generated Panama classes, raw `MemorySegment` instances, or raw native pointer arithmetic.

Use confined `Arena` instances for temporary native memory that is only needed during one native call. Use dedicated closeable wrappers for allocations that must live across calls, such as native index handles, RMM allocations, pinned host buffers, and matrix storage.

Check every native return code through a shared helper such as `checkCuVSError`. Include the native function name in error paths so failures are useful to users and maintainers.

## Common Design Considerations

### Performance

Avoid hidden copies between Java heap, host memory, and device memory. Prefer `CuVSMatrix` builders or device-backed matrix APIs when callers need to control where data lives. When a convenience API accepts Java arrays, document that it may allocate and copy data into native memory.

Keep Java-side work around native calls light. The Java layer should validate inputs, prepare native views, call the NVIDIA cuVS C API, and return Java objects that own their native results. Do not reimplement vector search, clustering, or preprocessing logic in Java.

### Threading Model

Treat `CuVSResources` as the boundary for native streams, device state, temporary buffers, and allocations. The native resources object is not safe for concurrent scoped access; callers must close a `CuVSResources.ScopedAccess` before another thread uses the same resource handle.

Use `SynchronizedCuVSResources` or another explicit synchronization strategy when a Java-facing API needs to share one resource handle across threads. Do not add hidden process-wide locks unless the ownership and performance impact are documented.

### Native Memory And Matrices

Use `CuVSMatrix`, `CuVSHostMatrix`, and `CuVSDeviceMatrix` for matrix-shaped data instead of ad hoc arrays or pointer pairs. Matrix APIs should preserve size, column count, data type, memory kind, and stride constraints clearly enough that native calls can validate inputs before launching work.

Prefer explicit temporary files or caller-provided paths for APIs that need intermediate storage during serialization, deserialization, or native interop. Use `CuVSResources.tempDirectory()` for defaults that need to stay associated with the resource owner.

### Native Library Loading

Keep provider discovery and native library loading behind `CuVSProvider` and `CuVSServiceProvider`. Public APIs should not require callers to know how the native libraries are located, except for documented deployment requirements such as `LD_LIBRARY_PATH` when using locally built libraries.

When adding a native dependency, update packaging, provider loading, tests, and documentation together so source builds and packaged JARs continue to behave consistently.

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

### Java Formatting

Java code is formatted by the Maven Spotless plugin with Google Java Format. The generated Panama bindings under `src/**/panama/*.java` are excluded from formatting and should not be edited by hand.

Run Java formatting and checks through the Java build:

```bash
./build.sh java
```

For focused Maven work, run commands from `java/cuvs-java`:

```bash
mvn spotless:apply
```

### Package Structure

Place public API types in `com.nvidia.cuvs`, service-provider types in `com.nvidia.cuvs.spi`, and implementation details in `com.nvidia.cuvs.internal`. Avoid exporting implementation packages from `module-info.java`.

Keep public interfaces and builders readable. Prefer clear Java names over native abbreviations unless the abbreviation is already part of the NVIDIA cuVS API surface.

### Copyright

RAPIDS pre-commit hooks and the Java Spotless configuration check copyright headers on modified tracked files. Use the repository Java license header and avoid hand-editing generated binding files unless the generator is being updated.

## Code Quality

### Testing

Java APIs need direct integration test coverage because downstream applications rely on their resource ownership, native loading, and runtime behavior. Prefer tests that exercise public entry points, common data shapes, close behavior, error paths, serialization, and expected search result shapes.

Run the Java integration tests from the `java` directory:

```bash
./build.sh --run-java-tests
```

To run a focused integration test from `java/cuvs-java`, use Maven:

```bash
mvn clean integration-test -Dit.test=com.nvidia.cuvs.CagraBuildAndSearchIT
```

When a randomized test fails, reproduce it with the reported `tests.seed`.

### Error Handling

Validate Java inputs before creating native tensors or launching native work. Use Java exceptions such as `NullPointerException`, `IllegalArgumentException`, `IllegalStateException`, or `LibraryException` where they make the caller problem clear.

Translate native NVIDIA cuVS failures consistently through shared helpers. Do not ignore native return codes, and do not let partially constructed native handles escape without a matching cleanup path.

### Documentation

Public Java APIs require user-facing Javadocs. Document the purpose, parameters, return values, ownership rules, close requirements, default values, and any native-library or temporary-file constraints that affect correct use.

Keep examples small and complete enough to show resource ownership. Prefer examples that use `try`-with-resources and public API types rather than internal implementation classes.
