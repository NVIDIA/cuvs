# warpReduce / raft::add_op ADL ambiguity — Bug Report & Reproducer

## What you saw when building cuvs

A build of `libcuvs` against CUDA 13.2+ (CCCL 3.4+) fails with:

```
error: more than one instance of function template "warpReduce" matches:
  function template "T raft::warpReduce(T, ReduceLambda)"
  function template "Tp cub::detail::scan::warpReduce(Tp, ScanOpT &)"
  argument types are: (uint32_t, raft::add_op)
```

The error points into the device-code instantiation of the CUB scan kernel
triggered during compilation of `ivf_pq_build.cuh`.

## Root cause

### The triggering call site

`cpp/src/neighbors/ivf_pq/ivf_pq_build.cuh`, function
`calculate_offsets_and_indices` (~line 233), before fix `99b38018`:

```cpp
thrust::inclusive_scan(
    exec_policy,
    cluster_sizes, cluster_sizes + n_lists,
    cluster_offsets + 1,
    raft::add_op{});   // <-- this argument is the root trigger
```

### Why this blows up in CCCL 3.4+ / CUDA 13.2+

CCCL 3.4 introduced a new high-performance scan implementation
(`cub/device/dispatch/kernels/kernel_scan_warpspeed.cuh`) that defines its
own warp-level reduction helper:

```cpp
// cub::detail::scan namespace (kernel_scan_warpspeed.cuh, CCCL 3.4+)
template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API Tp warpReduce(const Tp input, ScanOpT& scan_op) { ... }
```

When `thrust::inclusive_scan` is called with `raft::add_op{}` as the binary
op, the compiler instantiates this kernel with `ScanOpT = raft::add_op`.
Inside the kernel body, line 455 makes an **unqualified** call:

```cpp
regWarpSum = warpReduce(regThreadSum, scan_op);  // scan_op is raft::add_op
```

C++ name lookup resolves this with two independent searches:

| Lookup kind | What it finds |
|---|---|
| Normal (unqualified) | `cub::detail::scan::warpReduce<Tp, ScanOpT>` — defined in the same namespace |
| ADL on `raft::add_op` | `raft::warpReduce<T, ReduceLambda>` — `raft::add_op` lives in `raft::` |

Both are function templates with identical match quality for
`(uint32_t, raft::add_op)`. Neither is more specialized. The call is
ambiguous and the compiler errors out.

### Why raft::warpReduce is in scope at all

`ivf_pq_build.cuh` includes `<raft/stats/histogram.cuh>`, which includes
`<raft/stats/detail/histogram.cuh>`, which includes
`<raft/util/cuda_utils.cuh>`, which includes `<raft/util/reduction.cuh>`.
That header declares `raft::warpReduce`. Even though the call site is inside
`cub::detail::scan`, ADL on `raft::add_op` reaches across into the `raft::`
namespace and finds it.

### Include chain

```
ivf_pq_build.cuh
  └─ <raft/stats/histogram.cuh>
       └─ <raft/stats/detail/histogram.cuh>
            └─ <raft/util/cuda_utils.cuh>
                 └─ <raft/util/reduction.cuh>   ← declares raft::warpReduce
```

## Fixes applied

### Fix 1 — raft (zbrad/raft@cu132, commit d1345188)

Added an explicit non-template overload to `raft/util/reduction.cuh`:

```cpp
// Before: only the generic template existed
template <typename T, typename ReduceLambda>
DI T warpReduce(T val, ReduceLambda reduce_op);   // matches anything

// After: explicit overload added
template <typename T>
DI T warpReduce(T val, raft::add_op reduce_op);   // more specialized -> wins
```

When both candidates are considered, partial ordering rules select the
`raft::add_op` overload as more specialized than `ReduceLambda`. The
ambiguity disappears: `raft::warpReduce<T, raft::add_op>` wins and the CUB
candidate is never chosen.

### Fix 2 — cuvs (commit 99b38018)

Changed the call site in `ivf_pq_build.cuh` to avoid `raft::add_op`
entirely:

```cpp
// Before
thrust::inclusive_scan(..., raft::add_op{});

// After
thrust::inclusive_scan(..., thrust::plus<>{});
```

`thrust::plus` lives in `thrust::`, so ADL on it does not search `raft::`.
The unqualified `warpReduce(input, scan_op)` inside the CUB kernel has only
one candidate — `cub::detail::scan::warpReduce` — and compiles without
ambiguity.

Fix 2 is sufficient on its own for cuvs. Fix 1 is the general raft-level
guard that prevents any future caller from hitting the same ADL trap when
using `raft::add_op` with CUB/Thrust algorithms.

## Minimal reproducer

`cpp/tests/regression/repro_warp_reduce_ambiguity.cu` isolates the collision
without the full build:

```bash
# Strip the explicit add_op overload from a copy of reduction.cuh, then:
nvcc -std=c++17 -arch=sm_121 repro_warp_reduce_ambiguity.cu \
    -I<raft>/cpp/include \
    -I<cccl>/thrust -I<cccl>/libcudacxx/include -I<cccl>/cub \
    -I<rmm>/cpp/include \
    -o /dev/null
```

## Regression test

`cpp/tests/regression/warp_reduce_add_op.cu` compiles and runs the kernel
using `raft::warpReduce(val, raft::add_op{})` directly, verifying that the
explicit overload resolves correctly and produces the right numeric result.
