---
slug: api-reference/cpp-api-neighbors-common
---

# Common

_Source header: `cuvs/neighbors/common.hpp`_

## Approximate Nearest Neighbors Types

<a id="neighbors-index"></a>
### neighbors::index

The base for approximate KNN index structures.

```cpp
struct index;
```

<a id="neighbors-index-params"></a>
### neighbors::index_params

The base for KNN index parameters.

```cpp
struct index_params {
  cuvs::distance::DistanceType metric;
  float metric_arg;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Distance type. |
| `metric_arg` | `float` | The argument used by some distance metrics. |

<a id="neighbors-mergestrategy"></a>
### neighbors::MergeStrategy

Strategy for merging indices.

This enum is declared separately to avoid namespace pollution when including common.hpp. It provides a generic merge strategy that can be used across different index types.

```cpp
enum class MergeStrategy {
  MERGE_STRATEGY_PHYSICAL = 0,
  MERGE_STRATEGY_LOGICAL = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_STRATEGY_PHYSICAL` | `0` |
| `MERGE_STRATEGY_LOGICAL` | `1` |

## Types

<a id="neighbors-dataset"></a>
### neighbors::dataset

Tags selecting dataset representation for `dataset` / `dataset_view`.

Each container defines nested `owning_storage` then `view_storage` (aliases into `detail::*` storage types shared by device/host). Accessibility (device vs host) is selected by the `Accessor` template parameter on `dataset` / `dataset_view`, not by duplicating containers. Layout kinds: empty, padded, standard, VPQ. `dataset` / `dataset_view` only express ownership vs view.

```cpp
template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct dataset;
```

<a id="math-type"></a>
### math_type

Floating-point type used for VQ/PQ codebooks (rows are still uint8 codes).

```cpp
using math_type = MathT;
```

<a id="neighbors-dataset-view-kind-of"></a>
### neighbors::dataset_view_kind_of

Primary template returns `unknown` so traits safely return `false` for non-dataset-view types.

```cpp
template <typename V>
struct dataset_view_kind_of {
  static constexpr dataset_view_kind value;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `value` | `static constexpr dataset_view_kind` |  |

<a id="neighbors-dataset-view-is-device-accessible"></a>
### neighbors::dataset_view_is_device_accessible

True when the dataset view accessor is device-accessible.

```cpp
template <typename V>
struct dataset_view_is_device_accessible;
```

<a id="neighbors-with-accessor"></a>
### neighbors::with_accessor

Generic accessor retargeting while preserving the dataset tag/layout and value/index types:

`dataset&lt;Tag, DataT, IdxT, OldAccessor&gt;      -&gt; dataset&lt;Tag, DataT, IdxT, NewAccessor&gt;` `dataset_view&lt;Tag, DataT, IdxT, OldAccessor&gt; -&gt; dataset_view&lt;Tag, DataT, IdxT, NewAccessor&gt;`

```cpp
template <typename DatasetLikeT, typename NewAccessor>
struct with_accessor;
```

<a id="neighbors-to-device-accessor"></a>
### neighbors::to_device_accessor

Map any host accessor to its device counterpart (same payload policy).

```cpp
template <typename Accessor>
struct to_device_accessor;
```

<a id="neighbors-device-counterpart"></a>
### neighbors::device_counterpart

Maps a host dataset view type to its device-resident counterpart.

```cpp
template <typename HostViewT>
struct device_counterpart;
```

<a id="neighbors-cagra-view-element-type"></a>
### neighbors::cagra_view_element_type

Element type `T` for `cagra::build(res, params, dataset_view)` (deduced, not a template arg).

```cpp
template <typename V, typename = void>
struct cagra_view_element_type;
```

<a id="neighbors-ivf-list-base"></a>
### neighbors::ivf::list_base

Abstract base class for IVF list data.

This allows polymorphic access to list data regardless of the underlying layout.

TODO: Make this struct internal (tracking issue: https://github.com/nvidia/cuvs/issues/1726)

```cpp
template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
struct list_base;
```

<a id="neighbors-ivf-list"></a>
### neighbors::ivf::list

The data for a single IVF list.

```cpp
template <template <typename, typename...> typename SpecT,
typename SizeT,
typename... SpecExtraArgs>
struct list : public list_base<typename SpecT<SizeT, SpecExtraArgs...>::value_type,
typename SpecT<SizeT, SpecExtraArgs...>::index_type,
SizeT> {
  raft::device_mdarray<value_type, list_extents, raft::row_major> data;
  raft::device_mdarray<index_type, raft::extent_1d<size_type>, raft::row_major> indices;
  std::atomic<size_type> size;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `data` | [`raft::device_mdarray<value_type, list_extents, raft::row_major>`](/api-reference/cpp-api-neighbors-ivf-pq#list-extents) | Possibly encoded data; it's layout is defined by `SpecT`. |
| `indices` | `raft::device_mdarray<index_type, raft::extent_1d<size_type>, raft::row_major>` | Source indices. |
| `size` | `std::atomic<size_type>` | The actual size of the content. |

## Filtering for ANN Types

<a id="neighbors-filtering-filtertype"></a>
### neighbors::filtering::FilterType

Filtering for ANN Types

```cpp
enum class FilterType : int {
  None = 0,
  Bitmap = 1,
  Bitset = 2,
  Bloom = 3,
  UDF = 100
};
```

**Values**

| Name | Value |
| --- | --- |
| `None` | `0` |
| `Bitmap` | `1` |
| `Bitset` | `2` |
| `Bloom` | `3` |
| `UDF` | `100` |

<a id="neighbors-filtering-none-sample-filter-operator"></a>
### neighbors::filtering::none_sample_filter::operator

```cpp
constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
// query index
const uint32_t query_ix,
// the current inverted list index
const uint32_t cluster_ix,
// the index of the current sample inside the current inverted list
const uint32_t sample_ix) const;
```

**Returns**

`constexpr __forceinline__ _RAFT_HOST_DEVICE bool`

<a id="neighbors-filtering-none-sample-filter-get-filter-type"></a>
### neighbors::filtering::none_sample_filter::get_filter_type

```cpp
FilterType get_filter_type() const override;
```

**Returns**

[`FilterType`](/api-reference/cpp-api-neighbors-common#neighbors-filtering-filtertype)

<a id="neighbors-filtering-ivf-to-sample-filter"></a>
### neighbors::filtering::ivf_to_sample_filter

Filter used to convert the cluster index and sample index of an IVF search into a sample index. This can be used as an intermediate filter.

```cpp
template <typename index_t, typename filter_t>
struct ivf_to_sample_filter : public base_filter {
  const index_t* const* inds_ptrs_;
  const filter_t next_filter_;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `inds_ptrs_` | `const index_t* const*` |  |
| `next_filter_` | `const filter_t` |  |

<a id="neighbors-filtering-ivf-to-sample-filter-operator"></a>
### neighbors::filtering::ivf_to_sample_filter::operator

If the original filter takes three arguments, then don't modify the arguments.

```cpp
inline _RAFT_HOST_DEVICE bool operator()(
// query index
const uint32_t query_ix,
// the current inverted list index
const uint32_t cluster_ix,
// the index of the current sample inside the current inverted list
const uint32_t sample_ix) const;
```

If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample index.

**Returns**

`inline _RAFT_HOST_DEVICE bool`

<a id="neighbors-filtering-bitmap-filter"></a>
### neighbors::filtering::bitmap_filter

Filter an index with a bitmap

```cpp
template <typename bitmap_t, typename index_t>
struct bitmap_filter : public base_filter {
  const view_t bitmap_view_;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `bitmap_view_` | `const view_t` |  |

<a id="neighbors-filtering-bitmap-filter-operator"></a>
### neighbors::filtering::bitmap_filter::operator

```cpp
inline _RAFT_HOST_DEVICE bool operator()(
// query index
const uint32_t query_ix,
// the index of the current sample
const uint32_t sample_ix) const;
```

**Returns**

`inline _RAFT_HOST_DEVICE bool`

<a id="neighbors-filtering-bitmap-filter-get-filter-type"></a>
### neighbors::filtering::bitmap_filter::get_filter_type

```cpp
FilterType get_filter_type() const override;
```

**Returns**

[`FilterType`](/api-reference/cpp-api-neighbors-common#neighbors-filtering-filtertype)

<a id="neighbors-filtering-bitset-filter"></a>
### neighbors::filtering::bitset_filter

Filter an index with a bitset

This filter holds a non-owning view of the bitset; it does not allocate or copy the underlying device buffer. The library performs no caching of the bitset across search calls. Allocating and populating the device bitset may be more expensive than a single filtered search, so callers that issue repeated searches against the same filter (e.g. many queries over one index) should build the bitset once and reuse it across those calls rather than rebuild it per search. Reusing the bitset is essential for realizing the full throughput of filtered search.

```cpp
template <typename bitset_t, typename index_t>
struct bitset_filter : public base_filter {
  const view_t bitset_view_;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `bitset_view_` | `const view_t` |  |

<a id="neighbors-filtering-bitset-filter-bitset-filter"></a>
### neighbors::filtering::bitset_filter::bitset_filter

```cpp
_RAFT_HOST_DEVICE bitset_filter(const view_t bitset_for_filtering);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `bitset_for_filtering` |  | `const view_t` |  |

**Returns**

`_RAFT_HOST_DEVICE`

<a id="neighbors-filtering-bitset-filter-get-filter-type"></a>
### neighbors::filtering::bitset_filter::get_filter_type

```cpp
FilterType get_filter_type() const override;
```

**Returns**

[`FilterType`](/api-reference/cpp-api-neighbors-common#neighbors-filtering-filtertype)

<a id="neighbors-filtering-bloom-filter"></a>
### neighbors::filtering::bloom_filter

Filter CAGRA candidates with a global `cuvs::core::bloom_filter` over the index.

Build the filter once on the host with bulk `add`() over the allowed dataset row ids and pass the owning `cuvs::core::bloom_filter` to this wrapper. CAGRA internals build/cache the device payload, similar to `bitset_filter`, and the linked JIT-LTO fragment probes the same filter for every query and candidate with probabilistic membership tests.

Bloom filters have no false negatives: if a row was inserted, `contains` returns `true`. False positives are possible, so highly selective predicates may still need a bitset or UDF for exact filtering.

This adapter is non-owning. The referenced `cuvs::core::bloom_filter` must outlive the adapter and any searches that use it, and must not be moved or mutated concurrently with a search.

```cpp
struct bloom_filter : public base_filter {
  void* filter_data;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `filter_data` | `void*` |  |

<a id="neighbors-filtering-udf-filter"></a>
### neighbors::filtering::udf_filter

JIT-LTO user-defined filter predicate.

The source must define a device function named by `function_name` with signature:

Return `true` to allow a source vector to appear in the results and `false` to reject it. UDF dereferences it. CAGRA currently provides `source_index_t` as `uint32_t` in the generated JIT fragment.

```cpp
struct udf_filter : public base_filter {
  std::string source;
  void* filter_data;
  float filtering_rate;
  std::string function_name;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `source` | `std::string` | CUDA C++ source containing the device predicate. |
| `filter_data` | `void*` | Opaque device-accessible pointer passed to the predicate. |
| `filtering_rate` | `float` | Estimated fraction of rows rejected by the predicate, or negative if unknown. |
| `function_name` | `std::string` | Device function name to call from the generated CAGRA sample filter. |

## ANN MG index build parameters

<a id="neighbors-distribution-mode"></a>
### neighbors::distribution_mode

```cpp
enum distribution_mode;
```

<a id="neighbors-mg-index-params"></a>
### neighbors::mg_index_params

```cpp
template <typename Upstream>
struct mg_index_params : public Upstream {
  cuvs::neighbors::distribution_mode mode;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `mode` | [`cuvs::neighbors::distribution_mode`](/api-reference/cpp-api-neighbors-common#neighbors-distribution-mode) | Distribution mode |

## ANN MG search parameters

<a id="neighbors-replicated-search-mode"></a>
### neighbors::replicated_search_mode

```cpp
enum replicated_search_mode {
  LOAD_BALANCER,
  ROUND_ROBIN
};
```

**Values**

| Name | Value |
| --- | --- |
| `LOAD_BALANCER` | `` |
| `ROUND_ROBIN` | `` |

<a id="neighbors-sharded-merge-mode"></a>
### neighbors::sharded_merge_mode

```cpp
enum sharded_merge_mode {
  MERGE_ON_ROOT_RANK,
  TREE_MERGE
};
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_ON_ROOT_RANK` | `` |
| `TREE_MERGE` | `` |

<a id="neighbors-mg-search-params"></a>
### neighbors::mg_search_params

```cpp
template <typename Upstream>
struct mg_search_params : public Upstream {
  cuvs::neighbors::replicated_search_mode search_mode;
  cuvs::neighbors::sharded_merge_mode merge_mode;
  int64_t n_rows_per_batch;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `search_mode` | [`cuvs::neighbors::replicated_search_mode`](/api-reference/cpp-api-neighbors-common#neighbors-replicated-search-mode) | Replicated search mode |
| `merge_mode` | [`cuvs::neighbors::sharded_merge_mode`](/api-reference/cpp-api-neighbors-common#neighbors-sharded-merge-mode) | Sharded merge mode |
| `n_rows_per_batch` | `int64_t` | Number of rows per batch |
