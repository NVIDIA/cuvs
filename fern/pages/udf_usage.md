# UDF Usage

> **Caution:** Custom distance metrics for IVF-flat search are **experimental**. They live under the `cuvs::neighbors::ivf_flat::experimental::udf` namespace and the associated `CUVS_METRIC` macro. APIs and behavior may change without a major release.

## What this feature does

You can supply **your own CUDA device code** that defines how distance accumulates between a query vector and database vectors **inside the IVF-flat interleaved scan** (the fine search over lists). Technical background on compilation and linking is in [Link-time Optimization](/developer-guide/advanced-topics/link-time-optimization).

## Available via C++ APIs

- IVF-flat — [search](/api-reference/cpp-api-neighbors-ivf-flat) (`search_params.metric_udf` / `CUVS_METRIC`).

## Requirements and tips

- Include `<cuvs/neighbors/ivf_flat.hpp>` and define a metric with `CUVS_METRIC(MyName, { ... })`. Set `search_params.metric_udf` to the string returned by `MyName_udf()`.
- Prefer the [helpers](#helpers-in-cuvs_metric-bodies) when combining lanes so one body works for scalar and packed `int8_t` / `uint8_t` as well as wider element types.
- Custom UDF is **not supported for fp16** (`__half` / `half`) indices at this time; the headers enforce this with a static assertion when applicable.
- The scan assumes **ascending** distance order for top-*k* selection; metrics that do not behave like a distance in that sense need careful validation.
- The first search with a new metric string may pay a one-time compilation cost; reuse the same string (and run a warmup) to benefit from the caches described in [JIT Compilation](jit_compilation.md).

## Example

```cpp
#include <cuvs/neighbors/ivf_flat.hpp>

namespace ivf = cuvs::neighbors::ivf_flat;

// L∞ (Chebyshev): per dimension, acc = max(acc, |x - y|); acc starts at 0 in the scan kernel.
CUVS_METRIC(my_chebyshev, {
  auto d = abs_diff(x, y);
  acc    = (d > acc) ? d : acc;
})

void run_search(raft::resources const& res,
                ivf::index<float, int64_t> const& index,
                raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  ivf::search_params params;
  params.metric_udf = my_chebyshev_udf();

  ivf::search(res, params, index, queries, neighbors, distances);
}
```

## Helpers in `CUVS_METRIC` bodies

Inside `CUVS_METRIC(MyName, { ... })` you write the body of `operator()(AccT& acc, point_type x, point_type y)`. In scope: `acc`, `x`, `y`, template parameters `T`, `AccT`, `Veclen`, and the helpers below. The macro's full argument list and notes live beside `CUVS_METRIC` in `<cuvs/neighbors/ivf_flat.hpp>`.

| Helper | Role |
| --- | --- |
| `point` (`x`, `y`) | Element view: `raw()`, `operator[](i)`, `size()`, `is_packed()`. |
| `squared_diff(x, y)` | Squared difference; typical building block for L2-style energy. |
| `abs_diff(x, y)` | Absolute difference per lane. |
| `dot_product(x, y)` | Dot product / packed-byte dot where applicable. |
| `product(x, y)` | Element-wise product. |
| `sum(x, y)` | Element-wise sum. |
| `max_elem(x, y)` | Element-wise maximum. |

More examples: `cpp/tests/neighbors/ann_ivf_flat/test_udf.cu`.

## Further reading

- C++ API reference: [neighbors::ivf_flat](/api-reference/cpp-api-neighbors-ivf-flat)
- JIT LTO architecture and IVF-flat fragments: [Link-time Optimization](/developer-guide/advanced-topics/link-time-optimization)
