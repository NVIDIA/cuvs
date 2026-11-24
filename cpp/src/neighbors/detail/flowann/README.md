# FlowANN implementation

FlowANN is described in "Disentangling Graph Dependencies for Efficient Billion-Scale GPU Vector Search" (OSDI 2026).

These are internal implementation headers. The integration is introduced in separate PRs; only the interfaces present in the current revision are available.


## Host service thread lifetime

The experimental queue runtime uses `std::jthread` for persistent CPU pollers. They must
continue servicing device requests while the calling thread waits for CUDA completion,
and an explicit search session can keep them alive between calls. An OpenMP parallel
region scoped to one search call cannot provide that lifetime. Threads select the
context's CUDA device and are stopped and joined before queue storage is released.
This threading choice needs an explicit exception to the repository's OpenMP-only host
threading guideline during maintainer review; this rationale does not claim that approval.

Pollers inherit the caller's CPU affinity by default. Internal callers may opt into
physical-core pinning with `queue_params::pin_pollers`; pinned contexts share a process-wide
allocation counter rather than all starting at CPU zero.


## Deploying a GDRCopy-enabled library

GDRCopy remains an external runtime dependency; installing cuVS does not bundle `libgdrapi`.
The runtime environment must provide a compatible `libgdrapi` and its SONAME links. Place them
beside the installed cuVS libraries, install them in the system loader paths, or set
`LD_LIBRARY_PATH` to the runtime GDRCopy directory before loading cuVS. The kernel driver is
also required for GDRCopy transfers.

The installed RPATH defaults to `$ORIGIN`, so moving an installation does not retain the
build host's GDRCopy prefix. Packagers with a fixed runtime layout may set
`CUVS_FLOWANN_GDRCOPY_INSTALL_RPATH` to their deployment search paths. The same setting applies
to `libcuvs_c` when `CUVSC_STATIC_CUVS_LIBRARY=ON`, since that library then carries the GDRCopy
dependency directly. A standalone C build accepts the same cache setting independently of the
C++ build's feature flags. Set it in each separate build when using a custom deployment path.
Compile-time package discovery still uses `GDRCopy_ROOT` or the usual
CMake search paths and does not establish the runtime layout.
