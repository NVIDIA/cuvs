---
slug: api-reference/cpp-api-neighbors-ball-cover
---

# Ball Cover

_Source header: `cuvs/neighbors/ball_cover.hpp`_

## Types

<a id="neighbors-ball-cover-index"></a>
### neighbors::ball_cover::index

Stores raw index data points, sampled landmarks, the 1-nns of index points

to their closest landmarks, and the ball radii of each landmark. This class is intended to be constructed once and reused across subsequent queries.

```cpp
template <typename idx_t, typename value_t>
struct index : cuvs::neighbors::index {
  raft::resources const& handle;
  int64_t m;
  int64_t n;
  int64_t n_landmarks;
  raft::device_matrix_view<const float, idx_t, raft::row_major> X;
  cuvs::distance::DistanceType metric;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `handle` | `raft::resources const&` |  |
| `m` | `int64_t` |  |
| `n` | `int64_t` |  |
| `n_landmarks` | `int64_t` |  |
| `X` | `raft::device_matrix_view<const float, idx_t, raft::row_major>` |  |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |

## Random Ball Cover algorithm

<a id="neighbors-ball-cover-build"></a>
### neighbors::ball_cover::build

Builds and populates a previously unbuilt cuvs::neighbors::ball_cover::index

```cpp
void build(raft::resources const& handle, index<int64_t, float>& index);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | library resource management handle |
| `index` | inout | [`index<int64_t, float>&`](/api-reference/cpp-api-neighbors-ball-cover#neighbors-ball-cover-index) | an empty (and not previous built) instance of cuvs::neighbors::ball_cover::index |

**Returns**

`void`
