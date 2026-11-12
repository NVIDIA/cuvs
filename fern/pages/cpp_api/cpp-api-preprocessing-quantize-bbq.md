---
slug: api-reference/cpp-api-preprocessing-quantize-bbq
---

# Bbq

_Source header: `cuvs/preprocessing/quantize/bbq.hpp`_

## Better Binary Quantization utilities

<a id="preprocessing-quantize-bbq-params"></a>
### preprocessing::quantize::bbq::params

```cpp
struct params {
  bbq_code_layout doc_layout;
  bbq_code_layout query_layout;
  cuvs::distance::DistanceType metric;
  float lambda;
  int iters;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `doc_layout` | `bbq_code_layout` | The code layout of the encoded rows on the document side of a distance |
| `query_layout` | `bbq_code_layout` | The code layout of the encoded rows on the query side of a distance. When it differs from doc_layout, the dataset is encoded a second time in this layout. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Used for selecting the per-row additional correction: \|\|x - c\|\|^2 for L2, x.c otherwise. |
| `lambda` | `float` | Relative interval weight of the quantization error against the norm-preserving term of the loss. |
| `iters` | `int` | Coordinate descent iterations per row. |

<a id="preprocessing-quantize-bbq-helpers-resolve-dequant-factors"></a>
### preprocessing::quantize::bbq::helpers::resolve_dequant_factors

Derives dequant_delta from lower/upper_intervals and the layout's code width, and

```cpp
void resolve_dequant_factors(
raft::resources const& res,
raft::device_vector_view<float, int64_t> dequant_delta,
raft::device_vector_view<float, int64_t> dequant_sum_delta,
raft::device_vector_view<const float, int64_t> lower_intervals,
raft::device_vector_view<const float, int64_t> upper_intervals,
raft::device_vector_view<const int32_t, int64_t> quantized_component_sums,
bbq_code_layout layout);
```

dequant_sum_delta from that delta and quantized_component_sums.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dequant_delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `dequant_sum_delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `lower_intervals` |  | `raft::device_vector_view<const float, int64_t>` |  |
| `upper_intervals` |  | `raft::device_vector_view<const float, int64_t>` |  |
| `quantized_component_sums` |  | `raft::device_vector_view<const int32_t, int64_t>` |  |
| `layout` |  | `bbq_code_layout` |  |

**Returns**

`void`

<a id="preprocessing-quantize-bbq-build"></a>
### preprocessing::quantize::bbq::build

Builds a BBQ quantizer from a dataset: computes the centroid shared by every row, and its squared norm, into `quantizer`.

```cpp
void build(raft::resources const& res,
raft::device_matrix_view<const float, int64_t> dense_dataset,
quantizer_view<float> quantizer);
```

Its dimension has to match the one of `dense_dataset`.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `dense_dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `quantizer` | out | `quantizer_view<float>` | a view of the BBQ quantizer to write the centroid into |

**Returns**

`void`

<a id="preprocessing-quantize-bbq-make-device-bbq-dataset"></a>
### preprocessing::quantize::bbq::make_device_bbq_dataset

Allocates a BBQ dataset of the shape `transform` writes, leaving its contents uninitialized.

```cpp
cuvs::device_bbq_dataset<float, int64_t> make_device_bbq_dataset(raft::resources const& res,
const params params,
int64_t n_rows,
uint32_t dim);
```

The rows are allocated in `params.doc_layout` and, when `params.query_layout` differs, a second time in that layout; the metric is recorded from `params.metric`, and the quantizer holds room for a centroid of `dim` components.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-params) | configuration for BBQ encoding; only doc_layout, query_layout and metric are used |
| `n_rows` | in | `int64_t` | number of rows |
| `dim` | in | `uint32_t` | number of components of each row before encoding |

**Returns**

`cuvs::device_bbq_dataset<float, int64_t>`

an uninitialized BBQ dataset

<a id="preprocessing-quantize-bbq-transform"></a>
### preprocessing::quantize::bbq::transform

Encodes a dataset into a BBQ dataset, against the centroid in the manifest of the BBQ dataset.

```cpp
void transform(raft::resources const& res,
const params params,
raft::device_matrix_view<const float, int64_t> dense_dataset,
cuvs::device_bbq_dataset<float, int64_t>::view_type out);
```

`params.query_layout` format as well. The manifest of `out` has to hold the quantizer already, from a previous call to `build`.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-params) | configuration for BBQ encoding, e.g. doc_layout, query_layout, metric, lambda, iters |
| `dense_dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `out` | inout | `cuvs::device_bbq_dataset<float, int64_t>::view_type` | a view of the BBQ dataset whose manifest holds the quantizer, to write the encoding into |

**Returns**

`void`
