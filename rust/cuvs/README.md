# Experimental tiered CAGRA storage

Build the native library with `CUVS_ENABLE_FLOWANN_SEARCH=ON` and
`CUVS_ENABLE_FLOWANN_BUILD=ON`. `CUVS_FLOWANN_USE_GDRCOPY=OFF` selects the
CUDA-copy transport. Ordinary CAGRA keeps its default device graph storage.

Select tiered storage through the existing CAGRA parameter builder:

```rust,ignore
use cuvs::neighbors::cagra::{GraphStorage, Index, IndexParams, TieredGraphParams};

let params = IndexParams::builder()
    .nn_descent()
    .graph_storage(GraphStorage::Tiered)
    .tiered(TieredGraphParams {
        device_graph_budget_bytes: 1 << 30,
        ..Default::default()
    })
    .build()?;
let index = Index::build_tiered(&resources, &params, &device_padded_dataset)?;
```

`build_tiered` owns its reordered vectors, so the input can be released after
construction. `build_compressed` accepts a host standard dataset view and
`CompressionParams` to produce an owning VPQ-F16 index. Use
`DatasetView::standard` to request standard layout explicitly, including when
rows already meet CAGRA's alignment rule.

`SearchParams::builder().tiered(TieredSearchParams { ..Default::default() })`
configures queues and synchronization. Keeping pollers running between calls
consumes CPU while the index is idle. Use the CUDA device that created the index.

Serialize tiered indexes with `include_dataset=true`. After
`Index::deserialize_graph_and_dataset`, `graph_storage()` reports `Tiered` and
`has_dataset()` is false because the native index owns the vectors internally.
The loaded index is ready for search. Ordinary graph-only files still require
an attached searchable dataset.

Run the native FlowANN lifecycle tests against a library built with both
FlowANN options enabled:

```sh
cargo test -p cuvs tiered_dense_build_search_and_round_trip -- --ignored
cargo test -p cuvs tiered_compressed_build_search_and_round_trip -- --ignored
```
