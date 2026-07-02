# Dynamic Batching

Dynamic batching wraps an existing ANN index and combines many concurrent small search requests into larger GPU batches. It is useful for serving systems where requests arrive from many CPU threads, each request may contain only a few queries, and launching one GPU search per request would leave throughput on the table.

The dynamic batching API does not build a new ANN index. It owns batching queues and temporary buffers, then calls the upstream index search function with a larger batch. The upstream index, upstream search parameters, and optional filter are chosen when the dynamic batching index is created.

## Example API Usage

[C++ API](/api-reference/cpp-api-neighbors-dynamic-batching)

### Wrapping an upstream index

The example below wraps a CAGRA index. The same pattern can wrap supported C++ upstream indexes such as CAGRA, IVF-Flat, IVF-PQ, and brute-force.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/dynamic_batching.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();

cagra::index_params cagra_build_params;
auto upstream_index = cagra::build(res, cagra_build_params, dataset.view());

cagra::search_params upstream_search_params;
upstream_search_params.itopk_size = 128;

int64_t k = 10;
dynamic_batching::index_params batch_params;
batch_params.k = k;
batch_params.max_batch_size = 128;
batch_params.n_queues = 4;
batch_params.conservative_dispatch = true;

dynamic_batching::index<float, uint32_t> batch_index{
    res,
    batch_params,
    upstream_index,
    upstream_search_params};
```

</Tab>
</Tabs>

### Searching through the batcher

Dynamic batching search is thread-safe. Share copies of the lightweight dynamic batching index across request threads, and call `dynamic_batching::search` from each thread.

<Tabs>
<Tab title="C++">

```cpp
dynamic_batching::search_params request_params;
request_params.dispatch_timeout_ms = 1.0;

auto queries = load_request_queries();
auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(
    res, queries.extent(0), k);
auto distances = raft::make_device_matrix<float, int64_t>(
    res, queries.extent(0), k);

dynamic_batching::search(res,
                         request_params,
                         batch_index,
                         queries.view(),
                         neighbors.view(),
                         distances.view());

raft::resource::sync_stream(res);
```

</Tab>
</Tabs>

### Using separate priority classes

There is one queue pool inside each dynamic batching index. To separate latency-sensitive requests from throughput-oriented requests, create more than one dynamic batching index over the same upstream index.

<Tabs>
<Tab title="C++">

```cpp
dynamic_batching::index_params low_priority_params;
low_priority_params.k = k;
low_priority_params.max_batch_size = 128;
low_priority_params.n_queues = 2;
low_priority_params.conservative_dispatch = true;

dynamic_batching::index_params high_priority_params;
high_priority_params.k = k;
high_priority_params.max_batch_size = 16;
high_priority_params.n_queues = 4;
high_priority_params.conservative_dispatch = false;

dynamic_batching::index<float, uint32_t> low_priority_index{
    res, low_priority_params, upstream_index, upstream_search_params};
dynamic_batching::index<float, uint32_t> high_priority_index{
    res, high_priority_params, upstream_index, upstream_search_params};
```

</Tab>
</Tabs>

## How Dynamic Batching works

Each dynamic batching index owns a fixed set of request queues. Each queue has its own CUDA stream and temporary device buffers for queries, neighbors, and distances. CPU request threads commit their query rows into a queue. The batcher gathers those query rows into a contiguous device batch, calls the upstream search function, and scatters the output back to the request-owned output matrices.

The upstream index is not copied. The dynamic batching index keeps a reference to the upstream index, so the upstream index must outlive the dynamic batching wrapper. The upstream search parameters are captured by value when the wrapper is constructed, which means all requests submitted to the same dynamic batching index use the same upstream search configuration.

If a request already contains at least `max_batch_size` query rows, dynamic batching bypasses the queue and calls the upstream search directly.

## When to use Dynamic Batching

Use dynamic batching for high-concurrency serving workloads where many request threads submit small search batches. It can improve GPU utilization by turning many small searches into fewer larger upstream searches and by using multiple queues to overlap queue filling, kernel launch overhead, and search work.

Avoid dynamic batching when requests already arrive in large batches, when a single thread submits work serially, or when each request needs different upstream search parameters. In those cases, call the upstream index search API directly.

Dynamic batching is currently a C++ API. Use the index-specific guides for normal build and search workflows, then add dynamic batching only when serving concurrency makes request aggregation useful.

## Configuration parameters

### Index parameters

| Name | Default | Description |
| --- | --- | --- |
| `k` | Required | Number of neighbors returned for every request submitted to this dynamic batching index. This is fixed when the wrapper is constructed. |
| `max_batch_size` | `100` | Maximum number of query rows in a batch passed to the upstream index. Larger values can improve throughput but increase latency and temporary memory. |
| `n_queues` | `3` | Number of independent request queues. More queues can improve throughput under high concurrency by letting one queue fill while another queue searches. |
| `conservative_dispatch` | `false` | Controls when the upstream search is launched. `false` favors lower latency and may launch before the batch is full. `true` waits for a full batch or timeout, reducing wasted GPU work at the cost of exposing more upstream search latency. |

### Search parameters

| Name | Default | Description |
| --- | --- | --- |
| `dispatch_timeout_ms` | `1.0` | Maximum time, in milliseconds, that a request can wait in the queue before dispatch. This affects dispatch timing only; total latency also includes gather, upstream search, scatter, synchronization, and any waiting in the caller. |

## Tuning

Start by tuning the upstream index normally. For example, tune CAGRA, IVF-Flat, or IVF-PQ search parameters on representative batches before adding the dynamic batching wrapper.

Set `max_batch_size` to the largest request group that still meets latency and memory targets. Larger batches usually improve throughput, but they also increase temporary buffers and can delay small requests.

Increase `n_queues` when many CPU threads submit work concurrently and one queue is often busy. More queues can hide launch overhead and keep the GPU fed, but each queue allocates its own query and output buffers.

Use `conservative_dispatch=false` for latency-sensitive small batches. Use `conservative_dispatch=true` when `max_batch_size` is large and it is too expensive to run upstream search on mostly empty batches.

Lower `dispatch_timeout_ms` for latency-sensitive traffic. Raise it when throughput matters more than tail latency and request arrivals are dense enough to fill larger batches.

## Memory footprint

Dynamic batching memory has two parts: the upstream index memory and the batching buffers. Use the upstream index guide to estimate index memory. This section estimates the extra buffers owned by the dynamic batching wrapper.

Variables:

- `Q`: Number of independent request queues. This is `n_queues`.
- `B`: Maximum number of query rows per queue. This is `max_batch_size`.
- `D`: Vector dimension, or number of values in each query vector.
- `K`: Number of neighbors returned per query. This is `k`.
- `S_q`: Bytes per query value. Use `4` for `float`, `2` for `half`, and `1` for `int8_t` or `uint8_t`.
- `S_i`: Bytes per output index. Use `4` for `uint32_t` or `8` for `int64_t`.
- `S_d`: Bytes per output distance. Distances are stored as `float`, so use `4`.
- `M_upstream`: Device memory used by the upstream index.
- `M_scratch`: Temporary memory used by the upstream search implementation, CUDA libraries, memory-resource padding, and allocator overhead.

The query staging buffers use:

$$
\text{batch\_query\_buffers}
  = Q \cdot B \cdot D \cdot S_q
$$

The output staging buffers use:

$$
\text{batch\_output\_buffers}
  = Q \cdot B \cdot K \cdot (S_i + S_d)
$$

The peak memory is approximately:

$$
\begin{aligned}
\text{dynamic\_batching\_peak}
  \approx&\ M_{\text{upstream}}
   + \text{batch\_query\_buffers} \\
  &+ \text{batch\_output\_buffers}
   + M_{\text{scratch}}
\end{aligned}
$$

There are also small pinned host-memory structures for request pointers, queue tokens, events, and timeout bookkeeping. These are usually much smaller than the device query and output buffers.

### Scratch and maximum batch size

The upstream search path may allocate temporary buffers based on `max_batch_size`, especially when `conservative_dispatch=false` because the upstream search can be launched with the maximum batch shape even when fewer request rows are valid. Plan memory as if the upstream search can see `B` query rows.

To estimate the largest `max_batch_size` that fits, first reserve memory for the upstream index and other application buffers:

$$
M_{\text{usable}}
  = (M_{\text{free}} - M_{\text{upstream}} - M_{\text{other}}) \cdot (1 - H)
$$

where:

- `M_free`: Free device memory before constructing the dynamic batching wrapper.
- `M_other`: Device memory reserved for application buffers that are not included in this formula.
- `H`: Headroom fraction for scratch and allocator overhead. Start with `0.10` to `0.20`, then replace it with measured peak overhead from a representative run.
- `M_usable`: Device memory available for dynamic batching buffers after reservations and headroom.

Then solve:

$$
B_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}}}
         {Q \cdot (D \cdot S_q + K \cdot (S_i + S_d))}
  \right\rfloor
$$

Use the result as a planning estimate, then validate with the actual upstream index and request mix.
