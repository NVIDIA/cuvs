---
slug: api-reference/cpp-api-neighbors-dynamic-batching
---

# Dynamic Batching

_Source header: `cuvs/neighbors/dynamic_batching.hpp`_

## Dynamic Batching index parameters

<a id="neighbors-dynamic-batching-index-params"></a>
### neighbors::dynamic_batching::index_params

Dynamic Batching index parameters

```cpp
struct index_params : cuvs::neighbors::index_params {
  int64_t k;
  int64_t max_batch_size;
  size_t n_queues;
  bool conservative_dispatch;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `k` | `int64_t` | The number of neighbors to search is fixed at construction time. |
| `max_batch_size` | `int64_t` | Maximum size of the batch to submit to the upstream index. |
| `n_queues` | `size_t` | The number of independent request queues.<br /><br />Each queue is associated with a unique CUDA stream and IO device buffers. If the number of concurrent requests is high, using multiple queues allows to fill-in data and prepare the batch while the other queue is busy. Moreover, the queues are submitted concurrently; this allows to better utilize the GPU by hiding the kernel launch latencies, which helps to improve the throughput. |
| `conservative_dispatch` | `bool` | By default (`conservative_dispatch = false`) the first CPU thread to commit a query to a batch dispatches the upstream search function as soon as possible (before the batch is full). In that case, it does not know the final batch size at the time of calling the upstream search and thus runs the upstream search with the maximum batch size every time, even if only one valid query is present in the batch. This reduces the latency at the cost of wasted GPU resources.<br /><br />The alternative behavaior (`conservative_dispatch = true`) is more conservative: the dispatcher thread starts the kernel that gathers input queries, but waits till the batch is full or the waiting time is exceeded. Only then it acquires the actual batch size and launches the upstream search. As a result, less GPU resources are wasted at the cost of exposing upstream search latency.<br /><br />*Rule of Thumb*: for a large `max_batch_size` set `conservative_dispatch = true`, otherwise keep it disabled. |

## Dynamic Batching search parameters

<a id="neighbors-dynamic-batching-search-params"></a>
### neighbors::dynamic_batching::search_params

Dynamic Batching search parameters

```cpp
struct search_params : cuvs::neighbors::search_params {
  double dispatch_timeout_ms;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dispatch_timeout_ms` | `double` | How long a request can stay in the queue (milliseconds).<br />Note, this only affects the dispatch time and does not reflect full request latency; the latter depends on the upstream search parameters and the batch size. |

## Dynamic Batching index type

<a id="neighbors-dynamic-batching-index"></a>
### neighbors::dynamic_batching::index

Lightweight dynamic batching index wrapper

One lightweight dynamic batching index manages a single index and a single search parameter set. This structure should be shared among multiple users via copy semantics: access to the underlying implementation is managed via a shared pointer, and concurrent search among the participants is thread-safe.

__Usage example__

__Priority queues__

The dynamic batching index has a limited support for prioritizing individual requests. There's only one pool of queues in the batcher and no functionality to prioritize one bach over the other. The `search_params::dispatch_timeout_ms` parameters passed in each request are aggregated internally and the batch is dispatched no later than any of the timeouts is exceeded. In this logic, a high-priority request can never be processed earlier than any lower-priority requests submitted earlier.

However, dynamic batching indexes are lightweight and do not contain any global or static state. This means it's easy to combine multiple batchers. As an example, you can construct one batching index per priority class:

```cpp
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
  std::shared_ptr<detail::batch_runner<T, IdxT>> runner;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `runner` | `std::shared_ptr<detail::batch_runner<T, IdxT>>` |  |

<a id="neighbors-dynamic-batching-index-index"></a>
### neighbors::dynamic_batching::index::index

Construct a dynamic batching index by wrapping the upstream index.

```cpp
template <typename Upstream>
index(const raft::resources& res,
const cuvs::neighbors::dynamic_batching::index_params& params,
const Upstream& upstream_index,
const typename Upstream::search_params_type& upstream_params,
const cuvs::neighbors::filtering::base_filter* sample_filter = nullptr);
```

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `Upstream` | `` | the upstream index type |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resources |
| `params` | in | [`const cuvs::neighbors::dynamic_batching::index_params&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index-params) | dynamic batching parameters |
| `upstream_index` | in | `const Upstream&` | the original index to perform the search (the reference must be alive for the lifetime of the dynamic batching index) |
| `upstream_params` | in | `const typename Upstream::search_params_type&` | the original index search parameters for all queries in a batch (the parameters are captured by value for the lifetime of the dynamic batching index) |
| `sample_filter` | in | `const cuvs::neighbors::filtering::base_filter*` | filtering function, if any, must be the same for all requests in a batch (the pointer must be alive for the lifetime of the dynamic batching index)<br />Default: `nullptr`. |

**Returns**

`void`

## Dynamic Batching search

<a id="neighbors-dynamic-batching-search"></a>
### neighbors::dynamic_batching::search

Search ANN using a dynamic batching index.

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<float, uint32_t> const& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

The search parameters of the upstream index and the optional filtering function are configured at the dynamic batching index construction time.

Like with many other indexes, the dynamic batching search has the stream-ordered semantics: the host function may return the control before the results are ready. Synchronize with the main CUDA stream in the given resource object to wait for arrival of the search results.

Dynamic batching search is thread-safe: call the search function with copies of the same index in multiple threads to increase the occupancy of the batches.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) | query-specific batching parameters, such as the maximum waiting time |
| `index` | in | [`dynamic_batching::index<float, uint32_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) | a dynamic batching index |
| `queries` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_queries, dim] |
| `neighbors` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | a device matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | a device matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

**Additional overload:** `neighbors::dynamic_batching::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<half, uint32_t> const& index,
raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) |  |
| `index` |  | [`dynamic_batching::index<half, uint32_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) |  |
| `queries` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::dynamic_batching::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<int8_t, uint32_t> const& index,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) |  |
| `index` |  | [`dynamic_batching::index<int8_t, uint32_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) |  |
| `queries` |  | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::dynamic_batching::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<uint8_t, uint32_t> const& index,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) |  |
| `index` |  | [`dynamic_batching::index<uint8_t, uint32_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) |  |
| `queries` |  | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::dynamic_batching::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<float, int64_t> const& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) |  |
| `index` |  | [`dynamic_batching::index<float, int64_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) |  |
| `queries` |  | `raft::device_matrix_view<const float, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::dynamic_batching::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<half, int64_t> const& index,
raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) |  |
| `index` |  | [`dynamic_batching::index<half, int64_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) |  |
| `queries` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::dynamic_batching::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<int8_t, int64_t> const& index,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) |  |
| `index` |  | [`dynamic_batching::index<int8_t, int64_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) |  |
| `queries` |  | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::dynamic_batching::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<uint8_t, int64_t> const& index,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`cuvs::neighbors::dynamic_batching::search_params const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-search-params) |  |
| `index` |  | [`dynamic_batching::index<uint8_t, int64_t> const&`](/api-reference/cpp-api-neighbors-dynamic-batching#neighbors-dynamic-batching-index) |  |
| `queries` |  | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`
