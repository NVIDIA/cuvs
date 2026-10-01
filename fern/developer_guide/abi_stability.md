# ABI Stability Guarantees

## Overview

ABI stands for **Application Binary Interface**. It defines how compiled programs interact with a shared library at runtime.

For cuVS, ABI stability means that an application built against one supported version of the **cuVS C library** can run with a compatible newer cuVS runtime without needing to be rebuilt.

For example:

```text
Application built against: cuVS 26.04
Runtime installed by user: cuVS 26.06

Result: works, as long as both versions use the same ABI major version
```

ABI stability is different from API stability:

| Concept | Meaning |
|---|---|
| API stability | Source code continues to compile. |
| ABI stability | Already-compiled binaries continue to run. |

A source-compatible change may still break ABI if it changes the binary layout, symbol names, function signatures, or expected runtime behavior.

---

## Why ABI Stability Is Needed

cuVS is used by downstream projects and language bindings such as Java, Rust, Go, and database integrations.

These users often want to build their application once and allow the final runtime environment to provide cuVS separately. Without ABI stability, downstream consumers may need to rebuild or repackage every time cuVS changes.

ABI stability enables this model:

```text
Vendor application
  built once against a supported cuVS version

User environment
  provides a compatible cuVS shared library

Result
  the application can load and run without being rebuilt
```

This is especially important for database vendors and other software providers that do not want to vendor or bundle a private copy of cuVS with every release.

ABI stability helps provide:

- Predictable runtime compatibility
- Smaller downstream packages
- Easier system-level deployment
- Safer upgrades for users
- Clear rules for when breaking changes are allowed

---

## ABI Compatibility Rule

A cuVS runtime is ABI-compatible with an application when both of the following are true:

1. The runtime has the **same ABI major version** as the version used at build time.
2. The runtime cuVS version is the **same or newer** than the version used at build time.

Example compatibility matrix:

| Built With | Runtime | Compatible? | Reason |
|---|---:|---:|---|
| cuVS 26.04, ABI 1.1 | cuVS 26.06, ABI 1.2 | Yes | Same ABI major, newer runtime |
| cuVS 26.04, ABI 1.1 | cuVS 26.02, ABI 1.0 | No | Runtime is older |
| cuVS 26.04, ABI 1.1 | cuVS 26.08, ABI 2.0 | No | ABI major changed |

---

## Shared Library Naming

The cuVS C shared library follows this pattern:

```text
libcuvs_c.so.<abi-major>.<abi-minor>
```

For example:

```text
libcuvs_c.so.1.2
```

Where:

```text
1 = ABI major version
2 = ABI minor version
```

The SONAME uses only the ABI major version:

```text
libcuvs_c.so.1
```

Applications that dynamically load cuVS should load the ABI-major versioned name:

```text
libcuvs_c.so.1
```

They should not load the fully specified file name:

```text
libcuvs_c.so.1.2
```

Loading the ABI-major versioned name allows compatible ABI-minor updates to work without relinking or rebuilding the application.

---

## Scope of the ABI Guarantee

The ABI stability guarantee applies to the public **C ABI**.

It applies to public C interface items that meet these conditions:

- They are installed under `include/cuvs/`
- They are declared in `.h` header files
- They are inside an `extern "C"` block
- They are part of the public cuVS C interface

The guarantee covers public C functions, enums, and structs that are exposed through the stable C interface.

It does not generally apply to internal implementation details or to the general C++ implementation ABI.

---

## Struct Stability

Structs require special care.

A struct is only ABI-stable when it is allocated, initialized, or managed by ABI-stable cuVS functions. User code should not directly depend on the internal layout of cuVS structs.

Recommended pattern:

```c
cuvsHnswIndex_t index;
cuvsHnswIndexCreate(&index);

cuvsHnswIndexParams_t params;
cuvsHnswIndexParamsCreate(&params);

cuvsHnswFromCagra(res, params, cagra_index, index);
```

Avoid this pattern:

```c
cuvsHnswIndex index = { ... };
cuvsHnswIndexParams params = { ... };

cuvsHnswFromCagra(res, &params, cagra_index, &index);
```

The second example is risky because it depends on struct layout. If cuVS later adds, removes, reorders, or changes fields, compiled applications may break.

---

# Developer Guide: How to Avoid Breaking ABI

## General Rule

When changing public C headers, assume that existing applications may already be compiled against the current ABI.

Use this rule of thumb:

```text
Adding a new symbol is usually safe.
Changing an existing symbol is usually unsafe.
Removing an existing symbol is an ABI break.
```

---

## Safe Changes in ABI-Compatible Releases

The following changes are generally safe during ABI-compatible releases:

- Add a new public C function
- Add a new enum value when it does not change existing values or behavior
- Add new functionality behind a new symbol
- Add fields to structs that are fully allocated and managed by cuVS
- Add new optional behavior without changing existing function signatures
- Add new APIs while leaving old APIs intact

Example of a safe additive change:

```c
cuvsStatus_t cuvsFoo(cuvsHandle_t handle, int n);

/* New symbol added later */
cuvsStatus_t cuvsFooWithOptions(
    cuvsHandle_t handle,
    int n,
    cuvsFooOptions_t options
);
```

The original function remains unchanged, so existing binaries continue to work.

---

## Unsafe Changes in ABI-Compatible Releases

Do not make these changes in ABI-compatible releases:

- Remove a public function
- Rename a public function
- Change a function return type
- Add, remove, or reorder function arguments
- Change the type of a function argument
- Change the size or layout of a public struct that users may construct directly
- Remove or rename struct fields
- Change the type of a struct field
- Remove or renumber enum values
- Change the meaning of an existing enum value
- Change behavior in a way that violates existing runtime expectations

Example of an ABI break:

```c
/* Original */
cuvsStatus_t cuvsFoo(cuvsHandle_t handle, int n);

/* ABI-breaking change */
cuvsStatus_t cuvsFoo(cuvsHandle_t handle, int64_t n);
```

Even though the function name is the same, the binary signature changed.

---

## How to Make an Incompatible Change Safely

If a function needs an incompatible signature, do not change the existing function directly.

Instead, add a new suffixed function.

Example:

```c
/* Existing ABI-stable function */
cuvsStatus_t cuvsFoo(cuvsHandle_t handle, int n);

/* New replacement function */
cuvsStatus_t cuvsFoo_v1(
    cuvsHandle_t handle,
    int64_t n,
    float threshold
);
```

The old function remains available for existing binaries. New applications can use the new suffixed function.

The old function should be documented as deprecated or superseded, and release notes should explain which replacement should be used.

---

## Consolidating During ABI-Breaking Releases

cuVS has planned releases where ABI-breaking changes are allowed. These are the releases where accumulated compatibility work can be consolidated.

During normal ABI-compatible releases, developers may accumulate suffixed replacement APIs:

```c
cuvsStatus_t cuvsFoo(...);
cuvsStatus_t cuvsFoo_v1(...);
cuvsStatus_t cuvsFoo_v5(...);
```

During an ABI-breaking release, the highest replacement can become the canonical unsuffixed API:

```c
cuvsStatus_t cuvsFoo(...);
```

For example:

```text
ABI 1.x:
  cuvsFoo()
  cuvsFoo_v1()
  cuvsFoo_v5()

ABI 2.0:
  cuvsFoo() now uses the cuvsFoo_v5 signature
```

At that point, older variants can be removed because the ABI major version has changed.

---

## ABI-Breaking Release Checklist

When performing ABI consolidation during a planned ABI-breaking release:

- Bump the ABI major version
- Update the SONAME
- Remove obsolete symbols that are no longer supported
- Promote the latest suffixed replacement to the canonical unsuffixed API
- Update public headers
- Update documentation
- Update release notes
- Update the compatibility matrix
- Update ABI baseline or ABI checking data
- Ensure packaging uses the correct shared library version
- Clearly document old signatures and their replacements

Example transition:

```text
Before:
  libcuvs_c.so.1
  cuvsFoo()
  cuvsFoo_v1()
  cuvsFoo_v5()

After:
  libcuvs_c.so.2
  cuvsFoo()  // uses the cuvsFoo_v5 behavior/signature
```

---

# Practical Checklist for Developers

Before modifying a public C header, ask:

- Is this header installed under `include/cuvs/`?
- Is this declaration inside `extern "C"`?
- Could downstream code already be compiled against this symbol?
- Am I changing a function name, return type, or parameter list?
- Am I changing an enum value or its meaning?
- Am I changing a struct layout that user code may depend on?
- Would an already-compiled application still load and call this correctly?
- Should this be a new suffixed symbol instead of a direct change?

Use this decision guide:

| Change | Allowed in ABI-compatible release? | Recommended approach |
|---|---:|---|
| Add a new function | Yes | Add a new symbol |
| Change a function signature | No | Add a suffixed replacement |
| Remove a function | No | Wait for ABI-breaking release |
| Rename a function | No | Add new function, keep old one |
| Add managed struct fields | Usually | Only if users cannot construct the struct directly |
| Change public struct layout | Risky | Avoid, or wait for ABI-breaking release |
| Remove enum value | No | Wait for ABI-breaking release |

---

# Summary

ABI stability lets applications built against one cuVS C library version run with compatible newer cuVS runtimes without being rebuilt.

Developers should preserve existing public C ABI symbols during ABI-compatible releases. Add new symbols instead of changing existing ones. When an incompatible change is required, introduce a suffixed replacement such as `_v1`, keep the old symbol available, and document the migration path.

During planned ABI-breaking releases, developers can consolidate these suffixed replacements by promoting the newest version to the canonical unsuffixed API, removing obsolete variants, and incrementing the ABI major version.
