---
slug: api-reference/cpp-api-stats-silhouette-score
---

# Silhouette Score

_Source header: `cuvs/stats/silhouette_score.hpp`_

## Silhouette Score

<a id="stats-silhouette-score"></a>
### stats::silhouette_score

main function that returns the average silhouette score for a given set of data and its clusterings

```cpp
float silhouette_score(
raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> X_in,
raft::device_vector_view<const int, int64_t> labels,
std::optional<raft::device_vector_view<float, int64_t>> silhouette_score_per_sample,
int64_t n_unique_labels,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `X_in` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | input matrix Data in row-major format (nRows x nCols) |
| `labels` | in | `raft::device_vector_view<const int, int64_t>` | the pointer to the array containing labels for every data sample (length: nRows) |
| `silhouette_score_per_sample` | out | `std::optional<raft::device_vector_view<float, int64_t>>` | optional array populated with the silhouette score for every sample (length: nRows) |
| `n_unique_labels` | in | `int64_t` | number of unique labels in the labels array |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`float`

The silhouette score.

<a id="stats-silhouette-score-batched"></a>
### stats::silhouette_score_batched

function that returns the average silhouette score for a given set of data and its clusterings

```cpp
float silhouette_score_batched(
raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> X,
raft::device_vector_view<const int, int64_t> labels,
std::optional<raft::device_vector_view<float, int64_t>> silhouette_score_per_sample,
int64_t n_unique_labels,
int64_t batch_size,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `X` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | input matrix Data in row-major format (nRows x nCols) |
| `labels` | in | `raft::device_vector_view<const int, int64_t>` | the pointer to the array containing labels for every data sample (length: nRows) |
| `silhouette_score_per_sample` | out | `std::optional<raft::device_vector_view<float, int64_t>>` | optional array populated with the silhouette score for every sample (length: nRows) |
| `n_unique_labels` | in | `int64_t` | number of unique labels in the labels array |
| `batch_size` | in | `int64_t` | number of samples per batch |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | the numerical value that maps to the type of distance metric to be used in the calculations<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`float`

The silhouette score.

**Additional overload:** `stats::silhouette_score`

main function that returns the average silhouette score for a given set of data and its clusterings

```cpp
double silhouette_score(
raft::resources const& handle,
raft::device_matrix_view<const double, int64_t, raft::row_major> X_in,
raft::device_vector_view<const int, int64_t> labels,
std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample,
int64_t n_unique_labels,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `X_in` | in | `raft::device_matrix_view<const double, int64_t, raft::row_major>` | input matrix Data in row-major format (nRows x nCols) |
| `labels` | in | `raft::device_vector_view<const int, int64_t>` | the pointer to the array containing labels for every data sample (length: nRows) |
| `silhouette_score_per_sample` | out | `std::optional<raft::device_vector_view<double, int64_t>>` | optional array populated with the silhouette score for every sample (length: nRows) |
| `n_unique_labels` | in | `int64_t` | number of unique labels in the labels array |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | the numerical value that maps to the type of distance metric to be used in the calculations<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`double`

The silhouette score.

**Additional overload:** `stats::silhouette_score_batched`

function that returns the average silhouette score for a given set of data and its clusterings

```cpp
double silhouette_score_batched(
raft::resources const& handle,
raft::device_matrix_view<const double, int64_t, raft::row_major> X,
raft::device_vector_view<const int, int64_t> labels,
std::optional<raft::device_vector_view<double, int64_t>> silhouette_score_per_sample,
int64_t n_unique_labels,
int64_t batch_size,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `X` | in | `raft::device_matrix_view<const double, int64_t, raft::row_major>` | input matrix Data in row-major format (nRows x nCols) |
| `labels` | in | `raft::device_vector_view<const int, int64_t>` | the pointer to the array containing labels for every data sample (length: nRows) |
| `silhouette_score_per_sample` | out | `std::optional<raft::device_vector_view<double, int64_t>>` | optional array populated with the silhouette score for every sample (length: nRows) |
| `n_unique_labels` | in | `int64_t` | number of unique labels in the labels array |
| `batch_size` | in | `int64_t` | number of samples per batch |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | the numerical value that maps to the type of distance metric to be used in the calculations<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`double`

The silhouette score.
