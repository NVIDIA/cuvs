# FlowANN implementation

FlowANN is described in "Disentangling Graph Dependencies for Efficient Billion-Scale GPU Vector Search" (OSDI 2026).

These are internal implementation headers. The integration is introduced in separate PRs; only the interfaces present in the current revision are available.


## Host service thread lifetime

The experimental queue runtime uses `std::jthread` for persistent CPU pollers. They must
continue servicing device requests while the calling thread waits for CUDA completion,
and an explicit search session can keep them alive between calls. An OpenMP parallel
region scoped to one search call cannot provide that lifetime. Threads select the
context's CUDA device and are stopped and joined before queue storage is released.
This threading choice needs an explicit exception to the repository's OpenMP-only host
threading guideline during maintainer review; this rationale does not claim that approval.

Pollers inherit the caller's CPU affinity by default. Internal callers may opt into
physical-core pinning with `queue_params::pin_pollers`; pinned contexts share a process-wide
allocation counter rather than all starting at CPU zero.

## C++ compatibility

The tiered fields change the layouts of `cagra::index`, `cagra::index_params`, and
`cagra::search_params`, including builds with FlowANN disabled. Rebuild consumers with
headers from the same cuVS version as the linked library. `index::graph()` throws for
tiered indexes because their device and host edges do not form one device adjacency matrix.
Use `graph_storage()` to inspect the representation before requesting a device graph.
