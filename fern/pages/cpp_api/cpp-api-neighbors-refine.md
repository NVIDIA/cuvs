---
slug: api-reference/cpp-api-neighbors-refine
---

# Refine

_Source header: `cuvs/neighbors/refine.hpp`_

## Approximate Nearest Neighbors Refinement

<a id="neighbors-refine"></a>
### neighbors::refine

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | device matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | device matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::device_matrix_view<const int64_t, int64_t, raft::row_major>` | indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | device matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | device matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> neighbor_candidates,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> indices,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | device matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | device matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>` | indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | device matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | device matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | device matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | device matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::device_matrix_view<const int64_t, int64_t, raft::row_major>` | indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | device matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | device matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | device matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | device matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::device_matrix_view<const int64_t, int64_t, raft::row_major>` | indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | device matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | device matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | device matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | device matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::device_matrix_view<const int64_t, int64_t, raft::row_major>` | indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | device matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | device matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::host_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | host matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | host matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::host_matrix_view<const int64_t, int64_t, raft::row_major>` | host matrix with indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::host_matrix_view<int64_t, int64_t, raft::row_major>` | host matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | host matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
raft::host_matrix_view<const uint32_t, int64_t, raft::row_major> neighbor_candidates,
raft::host_matrix_view<uint32_t, int64_t, raft::row_major> indices,
raft::host_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | host matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | host matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::host_matrix_view<const uint32_t, int64_t, raft::row_major>` | host matrix with indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::host_matrix_view<uint32_t, int64_t, raft::row_major>` | host matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | host matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
raft::host_matrix_view<const half, int64_t, raft::row_major> queries,
raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::host_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | host matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | host matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::host_matrix_view<const int64_t, int64_t, raft::row_major>` | host matrix with indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::host_matrix_view<int64_t, int64_t, raft::row_major>` | host matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | host matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::host_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | host matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | host matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::host_matrix_view<const int64_t, int64_t, raft::row_major>` | host matrix with indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::host_matrix_view<int64_t, int64_t, raft::row_major>` | host matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | host matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::refine`

Refine nearest neighbor search.

```cpp
void refine(raft::resources const& handle,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
raft::host_matrix_view<float, int64_t, raft::row_major> distances,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

The k nearest neighbors and distances are returned.

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | host matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | host matrix of the queries [n_queris, dims] |
| `neighbor_candidates` | in | `raft::host_matrix_view<const int64_t, int64_t, raft::row_major>` | host matrix with indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `indices` | out | `raft::host_matrix_view<int64_t, int64_t, raft::row_major>` | host matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | host matrix that stores the refined distances [n_queries, k] |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`
