---
slug: api-reference/cpp-api-selection-select-k
---

# Select K

_Source header: `cuvs/selection/select_k.hpp`_

## Batched-select k smallest or largest key/values

<a id="selection-select-k"></a>
### selection::select_k

Select k smallest or largest key/values from each row in the input data.

```cpp
void select_k(
raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> in_val,
std::optional<raft::device_matrix_view<const int64_t, int64_t, raft::row_major>> in_idx,
raft::device_matrix_view<float, int64_t, raft::row_major> out_val,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> out_idx,
bool select_min,
bool sorted                                                           = false,
SelectAlgo algo                                                       = SelectAlgo::kAuto,
std::optional<raft::device_vector_view<const int64_t, int64_t>> len_i = std::nullopt);
```

If you think of the input data `in_val` as a row-major matrix with `len` columns and `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills in the row-major matrix `out_val` of size (batch_size, k).

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | container of reusable resources |
| `in_val` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | inputs values [batch_size, len]; these are compared and selected. |
| `in_idx` | in | `std::optional<raft::device_matrix_view<const int64_t, int64_t, raft::row_major>>` | optional input payload [batch_size, len]; typically, these are indices of the corresponding `in_val`. If `in_idx` is `std::nullopt`, a contiguous array `0...len-1` is implied. |
| `out_val` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | output values [batch_size, k]; the k smallest/largest values from each row of the `in_val`. |
| `out_idx` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | output payload (e.g. indices) [batch_size, k]; the payload selected together with `out_val`. |
| `select_min` | in | `bool` | whether to select k smallest (true) or largest (false) keys. |
| `sorted` | in | `bool` | whether to make sure selected pairs are sorted by value<br />Default: `false`. |
| `algo` | in | `SelectAlgo` | the selection algorithm to use<br />Default: `SelectAlgo::kAuto`. |
| `len_i` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional array of size (batch_size) providing lengths for each individual row<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `selection::select_k`

Select k smallest or largest key/values from each row in the input data.

```cpp
void select_k(
raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> in_val,
std::optional<raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>> in_idx,
raft::device_matrix_view<float, int64_t, raft::row_major> out_val,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> out_idx,
bool select_min,
bool sorted                                                            = false,
SelectAlgo algo                                                        = SelectAlgo::kAuto,
std::optional<raft::device_vector_view<const uint32_t, int64_t>> len_i = std::nullopt);
```

If you think of the input data `in_val` as a row-major matrix with `len` columns and `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills in the row-major matrix `out_val` of size (batch_size, k).

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | container of reusable resources |
| `in_val` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | inputs values [batch_size, len]; these are compared and selected. |
| `in_idx` | in | `std::optional<raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>>` | optional input payload [batch_size, len]; typically, these are indices of the corresponding `in_val`. If `in_idx` is `std::nullopt`, a contiguous array `0...len-1` is implied. |
| `out_val` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | output values [batch_size, k]; the k smallest/largest values from each row of the `in_val`. |
| `out_idx` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | output payload (e.g. indices) [batch_size, k]; the payload selected together with `out_val`. |
| `select_min` | in | `bool` | whether to select k smallest (true) or largest (false) keys. |
| `sorted` | in | `bool` | whether to make sure selected pairs are sorted by value<br />Default: `false`. |
| `algo` | in | `SelectAlgo` | the selection algorithm to use<br />Default: `SelectAlgo::kAuto`. |
| `len_i` | in | `std::optional<raft::device_vector_view<const uint32_t, int64_t>>` | optional array of size (batch_size) providing lengths for each individual row<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `selection::select_k`

Select k smallest or largest key/values from each row in the input data.

```cpp
void select_k(
raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> in_val,
std::optional<raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>> in_idx,
raft::device_matrix_view<half, int64_t, raft::row_major> out_val,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> out_idx,
bool select_min,
bool sorted                                                            = false,
SelectAlgo algo                                                        = SelectAlgo::kAuto,
std::optional<raft::device_vector_view<const uint32_t, int64_t>> len_i = std::nullopt);
```

If you think of the input data `in_val` as a row-major matrix with `len` columns and `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills in the row-major matrix `out_val` of size (batch_size, k).

Example usage

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | container of reusable resources |
| `in_val` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | inputs values [batch_size, len]; these are compared and selected. |
| `in_idx` | in | `std::optional<raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>>` | optional input payload [batch_size, len]; typically, these are indices of the corresponding `in_val`. If `in_idx` is `std::nullopt`, a contiguous array `0...len-1` is implied. |
| `out_val` | out | `raft::device_matrix_view<half, int64_t, raft::row_major>` | output values [batch_size, k]; the k smallest/largest values from each row of the `in_val`. |
| `out_idx` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | output payload (e.g. indices) [batch_size, k]; the payload selected together with `out_val`. |
| `select_min` | in | `bool` | whether to select k smallest (true) or largest (false) keys. |
| `sorted` | in | `bool` | whether to make sure selected pairs are sorted by value<br />Default: `false`. |
| `algo` | in | `SelectAlgo` | the selection algorithm to use<br />Default: `SelectAlgo::kAuto`. |
| `len_i` | in | `std::optional<raft::device_vector_view<const uint32_t, int64_t>>` | optional array of size (batch_size) providing lengths for each individual row<br />Default: `std::nullopt`. |

**Returns**

`void`
