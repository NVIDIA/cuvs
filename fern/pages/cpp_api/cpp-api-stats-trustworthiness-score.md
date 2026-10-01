---
slug: api-reference/cpp-api-stats-trustworthiness-score
---

# Trustworthiness Score

_Source header: `cuvs/stats/trustworthiness_score.hpp`_

## Trustworthiness

<a id="stats-trustworthiness-score"></a>
### stats::trustworthiness_score

Compute the trustworthiness score

```cpp
double trustworthiness_score(
raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> X,
raft::device_matrix_view<const float, int64_t, raft::row_major> X_embedded,
int n_neighbors,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtUnexpanded,
int batch_size                      = 512);
```

**Note:** The constness of the data in X_embedded is currently casted away and the data is slightly<br />modified.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `X` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | Data in original dimension |
| `X_embedded` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | Data in target dimension (embedding) |
| `n_neighbors` | in | `int` | Number of neighbors considered by trustworthiness score |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Distance metric to use. Euclidean (L2) is used by default<br />Default: `cuvs::distance::DistanceType::L2SqrtUnexpanded`. |
| `batch_size` | in | `int` | Batch size<br />Default: `512`. |

**Returns**

`double`

Trustworthiness score
