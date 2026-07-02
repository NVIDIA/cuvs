# Compatibility

ABI stability means that an application built with one compatible version of NVIDIA cuVS can continue to run with a newer compatible NVIDIA cuVS runtime without being rebuilt.

ABI stands for **Application Binary Interface**. It is the runtime contract between a compiled application and the shared library it loads. Most users do not need to know the low-level details. The important point is that ABI stability helps already-built applications keep working across compatible NVIDIA cuVS upgrades.

## Why ABI Stability Matters

Many applications use NVIDIA cuVS through another product, such as a database, search system, language binding, or packaged application.

Without ABI stability, every NVIDIA cuVS update could require those applications to be rebuilt or repackaged. ABI stability allows application vendors to build against a supported NVIDIA cuVS version while end users install a compatible NVIDIA cuVS runtime separately.

This helps provide:

- Easier runtime upgrades
- Smaller application packages
- More predictable compatibility
- Safer deployments
- Clearer rules for breaking changes

## Versions You May See

There are two related version numbers:

| Version | Example | Meaning |
| --- | --- | --- |
| NVIDIA cuVS version | `26.04`, `26.06`, `26.08` | The normal NVIDIA cuVS release version |
| ABI version | `1.1`, `1.2`, `2.0` | The binary compatibility version |

The NVIDIA cuVS version tells you which release you are using. The ABI version tells you whether a compiled application can run with that runtime.

## Release Versioning

NVIDIA cuVS follows the [Calendar Versioning](https://calver.org/overview.html) scheme used by RAPIDS. An NVIDIA cuVS version such as `26.04` means the release was made in April 2026.

RAPIDS releases are scheduled every even-numbered month. The current release schedule is published in the [RAPIDS Maintainers Docs](https://docs.rapids.ai/maintainers/), and completed release schedules are archived on the [RAPIDS release schedule page](https://docs.rapids.ai/releases/schedule/).

NVIDIA cuVS currently guarantees ABI compatibility for each six-month release window. ABI-breaking changes may happen between the `YY.12` and following `YY.02` releases, and between the `YY.06` and `YY.08` releases. In practice, the stable ABI windows are `YY.02` through `YY.06`, and `YY.08` through `YY.12`.

## ABI Versioning Scheme

The ABI version uses `<abi-major>.<abi-minor>`.

For example, ABI version `1.2` means:

- `1` is the ABI major version.
- `2` is the ABI minor version.

The ABI major version identifies a compatibility family. For example, ABI `1.x` and ABI `2.x` are different compatibility families.

## Compatibility Rule

An NVIDIA cuVS runtime is ABI-compatible with an application when both of these are true:

1. The runtime has the same ABI major version as the version used when the application was built.
2. The runtime NVIDIA cuVS version is the same as or newer than the version used when the application was built.

For example:

| Built With | Runtime | Compatible? | Reason |
| --- | --- | --- | --- |
| NVIDIA cuVS `26.04`, ABI `1.1` | NVIDIA cuVS `26.06`, ABI `1.2` | Yes | Same ABI major, newer runtime |
| NVIDIA cuVS `26.04`, ABI `1.1` | NVIDIA cuVS `26.02`, ABI `1.0` | No | Runtime is older |
| NVIDIA cuVS `26.04`, ABI `1.1` | NVIDIA cuVS `26.08`, ABI `2.0` | No | ABI major changed |

## ABI Minor Version

The ABI minor version may increase when NVIDIA cuVS adds ABI-compatible functionality.

For example, a runtime with ABI `1.2` can usually run an application built against ABI `1.1`, as long as the NVIDIA cuVS runtime version is also new enough.

The reverse is not guaranteed. An application built against ABI `1.2` may depend on symbols that do not exist in an older ABI `1.1` runtime.

## ABI Major Version

The ABI major version changes when binary compatibility may break.

For example, an application built for ABI `1.x` should not assume it can run with ABI `2.x` unless it is rebuilt or the application vendor explicitly supports that combination.

## Shared Library Naming

The NVIDIA cuVS C shared library includes the ABI major version in its name.

| Library name | Meaning |
| --- | --- |
| `libcuvs_c.so.1` | ABI major version `1` |
| `libcuvs_c.so.2` | ABI major version `2` |

This helps the system load a runtime from the correct compatibility family.

## What Is Covered

The ABI stability guarantee applies to the NVIDIA cuVS C shared library ABI that downstream applications and bindings load at runtime.

This includes:

- Exported C symbols
- C function signatures
- Public C structs and enums that are part of the ABI contract
- Shared library naming and ABI versioning

## Why C++ ABI Stability Is Challenging

<Note>The NVIDIA cuVS ABI guarantee is focused on the C shared library, not the C++ API.</Note>

C is the right boundary for long-lived binary compatibility because exported function names, struct layouts, and library versions can be controlled explicitly.

C++ exposes many more implementation details at the binary boundary. A compiled C++ caller may depend on compiler-specific name mangling, template instantiations, inline functions, object layout, virtual tables, exception handling, standard library types, allocator behavior, CUDA compiler behavior, and the exact versions of dependencies such as RAFT and RMM. Even a source-compatible change can alter the binary symbols or layouts that an already-built application expects.

For that reason, guaranteeing C++ ABI stability across NVIDIA cuVS releases, compilers, CUDA versions, and build options is not practical. Downstream projects that need a stable binary boundary should use the NVIDIA cuVS C API and shared library. C++ integrations should generally be rebuilt against the target NVIDIA cuVS, CUDA, compiler, and dependency stack.

## What Is Not Covered

ABI stability is not the same as promising that every behavior or source-level API detail will never change.

The guarantee does not necessarily cover:

- Experimental APIs
- Private implementation details
- Internal symbols
- Source compatibility for every header-only or C++ template detail
- Behavior changes that do not break the binary interface

When in doubt, use the NVIDIA cuVS runtime version recommended by your application vendor or package manager.

## Summary

Compatibility is easiest to reason about using the ABI major version.

Use a runtime with the same ABI major version as the application was built against, and make sure the runtime is the same NVIDIA cuVS version or newer. For example, an application built with ABI `1.1` can run with ABI `1.2`, but it should not assume compatibility with ABI `2.0`.
