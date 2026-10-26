---
slug: api-reference/cpp-api-core-dataset-binary
---

# Binary

_Source header: `cuvs/core/dataset/binary.hpp`_

## Types

<a id="binary-dictionary-view"></a>
### binary_dictionary_view

Non-owning view of the dictionary that binarizes a dataset: the per-dimension thresholds of a binary quantizer. An empty threshold vector stands for the zero threshold.

```cpp
template <typename MathT>
struct binary_dictionary_view {
  raft::device_vector_view<const MathT, int64_t> threshold;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `threshold` | `raft::device_vector_view<const MathT, int64_t>` |  |

<a id="device-binary-dataset-view"></a>
### device_binary_dataset_view

Non-owning view of a binary-quantized dataset.

The data are bit-packed codes, `ceil(dim / 8)` bytes per row; bit `i` of a row is stored in bit `i % 8` of byte `i / 8`, and the unused bits of the last byte are zero. The dictionary is the quantizer that produced the codes, so the logical dimension `dim` (the number of bits) never has to be guessed from the padded byte width.

```cpp
template <typename MathT, typename IdxT>
struct device_binary_dataset_view;
```
