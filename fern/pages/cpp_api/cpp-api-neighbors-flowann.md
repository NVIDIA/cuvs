---
slug: api-reference/cpp-api-neighbors-flowann
---

# Flowann

_Source header: `cuvs/neighbors/flowann.hpp`_

## Types

<a id="cuvs-neighbors-cagra-experimental-flowann-queue-params"></a>
### cuvs::neighbors::cagra::experimental::flowann::queue_params

Parameters controlling the host-serviced cross-graph queues.

```cpp
struct queue_params {
  std::uint32_t num_queues;
  std::uint32_t empty_pause;
  bool collect_statistics;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `num_queues` | `std::uint32_t` | Number of independent GPU-to-CPU command queues. |
| `empty_pause` | `std::uint32_t` | Number of CPU pause instructions issued after an empty poll. |
| `collect_statistics` | `bool` | Collect aggregate queue and polling statistics. |

<a id="cuvs-neighbors-cagra-experimental-flowann-queue-statistics"></a>
### cuvs::neighbors::cagra::experimental::flowann::queue_statistics

Aggregate counters collected by the host-serviced cross-graph queues.

```cpp
struct queue_statistics {
  std::uint64_t polls;
  std::uint64_t empty_polls;
  std::uint64_t processed_commands;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `polls` | `std::uint64_t` |  |
| `empty_polls` | `std::uint64_t` |  |
| `processed_commands` | `std::uint64_t` |  |

<a id="cuvs-neighbors-cagra-experimental-flowann-search-params"></a>
### cuvs::neighbors::cagra::experimental::flowann::search_params

FlowANN search parameters.

```cpp
struct search_params : cuvs::neighbors::cagra::search_params {
  std::uint32_t num_seeds;
  float sync_window_scale;
  std::uint32_t sync_drop_threshold;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `num_seeds` | `std::uint32_t` | Maximum number of preloaded medoid seed nodes used to initialize each query. |
| `sync_window_scale` | `float` | Scale used by the adaptive deferred-cross-edge synchronization window. |
| `sync_drop_threshold` | `std::uint32_t` | Parent-position drop that forces synchronization for a submitted cross-edge request. |

<a id="cuvs-neighbors-cagra-experimental-flowann-search-context"></a>
### cuvs::neighbors::cagra::experimental::flowann::search_context

Owns FlowANN command queues and their CPU polling threads.

The cross-graph storage passed to the constructor must outlive this context. A context can be reused across search calls for the same index. Construction and search must use the same current CUDA device; CPU pollers select that device before servicing requests.

```cpp
class search_context;
```

<a id="cuvs-neighbors-cagra-experimental-flowann-search-session"></a>
### cuvs::neighbors::cagra::experimental::flowann::search_session

Keep a search context's pollers running for the lifetime of this session.

Use a session for repeated low-latency search calls. The search context must outlive the session and must not be moved while the session is active.

```cpp
class search_session;
```

<a id="cuvs-neighbors-cagra-experimental-flowann-index"></a>
### cuvs::neighbors::cagra::experimental::flowann::index

FlowANN tiered-graph index.

The index stores a compressed inner graph in device memory and the cross-graph rows in host memory. The dataset is a non-owning view and its backing storage must outlive the index.

```cpp
template <typename T,
typename IdxT,
ann_dataset_view DatasetViewT = device_padded_dataset_view<T, int64_t>>
class index;
```

<a id="cuvs-neighbors-cagra-experimental-flowann-index-bundle"></a>
### cuvs::neighbors::cagra::experimental::flowann::index_bundle

Owns the dataset backing storage required by a FlowANN index.

```cpp
template <typename DatasetT, typename IndexT>
struct index_bundle {
  std::unique_ptr<DatasetT> dataset;
  IndexT index;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `std::unique_ptr<DatasetT>` |  |
| `index` | `IndexT` |  |
