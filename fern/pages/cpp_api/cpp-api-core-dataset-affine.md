---
slug: api-reference/cpp-api-core-dataset-affine
---

# Affine

_Source header: `cuvs/core/dataset/affine.hpp`_

## Types

<a id="affine-data"></a>
### affine_data

Rows with a per-row `scale` and a per-row `shift`: the building block of the data of affine

datasets. The rows are the values themselves for a plain affine dataset, and an encoding of the values for a quantized one (BBQ has one such group, RaBitQ has two); the `scale` and the `shift` are the per-row terms the rows are mapped through.

It holds arrays for an owning dataset and spans for a view of it, and it is an aggregate, so it is made as `affine_data{rows, scale, shift}`.

```cpp
template <typename RowsT, typename VectorT>
struct affine_data {
  RowsT rows;
  VectorT scale;
  VectorT shift;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `rows` | `RowsT` |  |
| `scale` | `VectorT` |  |
| `shift` | `VectorT` |  |

<a id="affine-spec"></a>
### affine_spec

Affine dataset specification: the rows of values (not quantized), each with its own `scale` and

`shift`. The data is one `affine_data`; there is no manifest.

```cpp
template <typename MathT, typename RowsPolicy>
struct affine_spec;
```
