---
slug: api-reference/cpp-api-neighbors-gpipnn
---

# Gpipnn

_Source header: `cuvs/neighbors/gpipnn.hpp`_

## GPiPNN graph build parameters

<a id="neighbors-experimental-gpipnn-index-params"></a>
### neighbors::experimental::gpipnn::index_params

Parameters for the GPiPNN proximity graph builder.

GPiPNN builds a navigable proximity graph without performing any graph search during construction. It partitions the dataset into small overlapping leaves via flattened randomized ball carving, computes a bi-directed 2-NN graph inside each leaf, streams the resulting candidate edges into a bounded per-point reservoir (HashPrune), and sparsifies the resulting reservoirs with DirectEdgePrune.

This is a prototype: scoring runs in fp16 with fp32 accumulation, and a single partition round is performed (no replication). It ships under `cuvs::neighbors::experimental` for its first release, so the parameter set and the entry points may change.

Metrics supported:

- L2Expanded
- CosineExpanded (note: the parameter defaults are tuned for L2Expanded and may not be optimal here)

Requires a GPU of compute capability 8.0 (Ampere) or newer.

```cpp
struct index_params : cuvs::neighbors::index_params {
  uint32_t graph_degree;
  uint32_t max_candidates;
  uint32_t leaf_degree;
  uint32_t hash_bits;
  uint32_t max_leaf_size;
  uint32_t min_leaf_size;
  float leader_fraction;
  uint32_t max_leaders;
  uint32_t fanout_level0;
  uint32_t fanout_level1;
  uint32_t max_rounds;
  float alpha;
  uint32_t leaf_batch_mib;
  float dataset_max_abs;
  uint64_t seed;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `graph_degree` | `uint32_t` | Degree of the output graph. |
| `max_candidates` | `uint32_t` | Capacity of the per-point HashPrune reservoir. Must be at least `graph_degree` and at most `kMaxCandidates`, which is 1024.<br /><br />Setting this equal to `graph_degree` disables DirectEdgePrune. The default of 128 is twice the default `graph_degree`, which is an empirically good value. It is not derived from `graph_degree`, so it should be updated when `graph_degree` changes. |
| `leaf_degree` | `uint32_t` | Neighbours kept per member of each leaf self-join. At most 32.<br /><br />This is what bounds the candidate supply, and therefore whether the reservoir has more members than the output has slots. Peak memory during the leaf phase is linear in it.<br /><br />The paper uses 2; 4 gives a better recall-for-build-time trade. |
| `hash_bits` | `uint32_t` | Number of LSH hyperplanes used for HashPrune residual hashing. At most 16.<br /><br />More bits mean finer direction buckets and therefore less pruning. The paper uses 12; recall is still rising at 14 for a minimal build-time cost. |
| `max_leaf_size` | `uint32_t` | Target maximum leaf size. Leaves are guaranteed not to exceed this. |
| `min_leaf_size` | `uint32_t` | Leaves smaller than this are merged with adjacent leaves from the same top-level bucket. A merge never crosses between buckets, so the last leaf of a bucket may stay below it. |
| `leader_fraction` | `float` | Fraction of a subproblem sampled as leaders, capped by `max_leaders`. |
| `max_leaders` | `uint32_t` | Hard cap on the number of leaders sampled per subproblem. |
| `fanout_level0` | `uint32_t` | Number of nearest leaders each point is assigned to at the top level. At most 32. |
| `fanout_level1` | `uint32_t` | Number of nearest sub-leaders each point is assigned to at the second level. At most 32. |
| `max_rounds` | `uint32_t` | Maximum number of levels of the partition tree (assign-then-sort rounds). |
| `alpha` | `float` | Alpha parameter of the DirectEdgePrune occlusion rule. Larger retains more edges. |
| `leaf_batch_mib` | `uint32_t` | Device memory, in MiB, the candidate buffer can occupy. This dictates how many leaves are processed per batch. Affects only peak memory and build time, not graph quality. 0 selects an automatic budget of 256 bytes per row, clamped to 512 MiB-6 GiB. |
| `dataset_max_abs` | `float` | Largest absolute value the dataset contains, or 0 to measure it.<br /><br />Read only for `float` input under L2: `int8` and `uint8` are bounded by 255, `half` is representable by definition, and cosine normalizes each row to unit norm instead.<br /><br />Scoring runs on half operands, so a float dataset peaking outside the range half holds usefully is first rescaled by one shared power of two, which leaves the neighbour ordering unchanged. Measuring the peak is one streaming pass over the dataset, a fraction of a percent of the build, so speed is rarely the reason to set this. Splitting a build across partitions is: the rescale preserves ordering only because a single factor applies to the whole dataset, so every partition has to be given the same value rather than deriving one from its own shard.<br /><br />A value below the true peak is rejected rather than tolerated -- the conversion would turn the elements above half's range into infinities and every distance touching them into a NaN. Overestimating is safe and costs some precision, since the factor places the stated peak just below the top of the range. |
| `seed` | `uint64_t` | Seed for leader sampling and LSH hyperplane generation. |

## GPiPNN graph build

<a id="neighbors-experimental-gpipnn-is-supported"></a>
### neighbors::experimental::gpipnn::is_supported

Whether this build of the library can run GPiPNN on the current device.

```cpp
auto is_supported() -> bool;
```

False when the device predates compute capability 8.0, and also when the device is new enough but libcuvs was compiled without an sm_80-or-newer image for the leaf join kernel.

**Returns**

`bool`

<a id="neighbors-experimental-gpipnn-build-graph"></a>
### neighbors::experimental::gpipnn::build_graph

Build a GPiPNN proximity graph over `dataset`.

```cpp
void build_graph(raft::resources const& res,
const index_params& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);
```

The output graph is a fully populated `[n_rows, graph_degree]` adjacency matrix. Wrap it with `raft::make_const_mdspan` and hand it to a CAGRA index to search it.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-gpipnn#neighbors-experimental-gpipnn-index-params) | build parameters |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | row-major device matrix [n_rows, dim]; `dim` must be even |
| `graph` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | row-major device matrix [n_rows, params.graph_degree] |

**Returns**

`void`

**Additional overload:** `neighbors::experimental::gpipnn::build_graph`

Build a GPiPNN proximity graph over a half `dataset`.

```cpp
void build_graph(raft::resources const& res,
const index_params& params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);
```

Used directly as the scoring operand, so this overload avoids the fp16 conversion copy.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-gpipnn#neighbors-experimental-gpipnn-index-params) | build parameters |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | row-major device matrix [n_rows, dim]; `dim` must be even |
| `graph` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | row-major device matrix [n_rows, params.graph_degree] |

**Returns**

`void`

**Additional overload:** `neighbors::experimental::gpipnn::build_graph`

Build a GPiPNN proximity graph over an int8 `dataset`.

```cpp
void build_graph(raft::resources const& res,
const index_params& params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);
```

Converted to fp16 for scoring, exactly: half represents every integer up to 2048.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-gpipnn#neighbors-experimental-gpipnn-index-params) | build parameters |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | row-major device matrix [n_rows, dim]; `dim` must be even |
| `graph` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | row-major device matrix [n_rows, params.graph_degree] |

**Returns**

`void`

**Additional overload:** `neighbors::experimental::gpipnn::build_graph`

Build a GPiPNN proximity graph over a uint8 `dataset`.

```cpp
void build_graph(raft::resources const& res,
const index_params& params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);
```

Converted to fp16 for scoring, exactly, same as the int8 overload.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-gpipnn#neighbors-experimental-gpipnn-index-params) | build parameters |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | row-major device matrix [n_rows, dim]; `dim` must be even |
| `graph` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | row-major device matrix [n_rows, params.graph_degree] |

**Returns**

`void`

## Types

<a id="gpipnn-params"></a>
### gpipnn_params

Parameters for a standalone GPiPNN build, spelled without the `experimental` namespace. Despite the enclosing namespace, CAGRA does not accept this type: it is absent from `cagra::graph_build_params_t`, so a GPiPNN graph is built on its own via `experimental::gpipnn::build_graph` and handed to a CAGRA index as a prebuilt graph.

```cpp
using gpipnn_params = cuvs::neighbors::experimental::gpipnn::index_params;
```
