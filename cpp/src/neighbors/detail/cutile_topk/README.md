# cuTile fused small Top-K

Physical K2 serves logical K2; physical K4 serves K3 and K4. The producer
computes a tile of distances and retains a small candidate list without
materializing the full query-by-corpus distance matrix. Retained candidates
are recomputed from the original stored coordinates in FP32 and ordered by
returned metric value and corpus index. K3 refines all four physical
candidates before compaction. Corpus-split launches merge the split-local
refined lists using the same ordering.

## Numerical/API contract

FP32 inputs use one-pass TF32 candidate scoring with corpus norms computed from
the same TF32-rounded operands as the dot product. Query-only ranking constants
are omitted: L2 ranks by half the corpus squared norm minus the dot product;
cosine ranks by the negative dot product divided by the corpus norm. This is
not the returned-distance formula. FP16 inputs use native half Tensor Core
products with FP32 accumulation; norm and retained-value arithmetic widens
stored half coordinates before multiplication/subtraction.

Both routes have **approximate membership**. Candidate scoring can omit a
neighbor that direct FP32 or FP64 ranking would retain. Refinement improves
returned values and ordering among retained IDs; it cannot recover an omitted
ID. There is no universal recall floor, exact-membership guarantee, or bitwise
equivalence promise to another reduction implementation.

Supported metrics are squared L2 (`L2Expanded`), Euclidean
(`L2SqrtExpanded`), cosine (`CosineExpanded`) and inner product
(`InnerProduct`). Returned distances/similarities are FP32. Final order is
ascending distance or descending inner product, then ascending corpus index,
including ties introduced by the Euclidean square root. Signed zeros compare
equal. As in Top1, a cosine comparison involving a zero-norm operand has
distance one. Invalid slots use index -1 and positive infinity; these sentinels
are not valid results and are excluded from descending-inner-product ordering.
PK8's existing wide split selection route has an L2-only producer image for
L2Expanded/L2SqrtExpanded. It preserves the generic scalar launch ABI, rounded
selection norms, candidate capacity and partitioning. Cosine/inner product keep
the generic image; missing specialization falls back to that image before any
kernel is enqueued. Refinement, merge and the approximate-membership contract
are unchanged. Ordinary exports remain generic; the L2 image has a distinct
symbol, fragment registration and launcher cache.

Inputs must be finite with representable metric intermediates; the search does
not perform a device-wide finite/overflow scan.

`brute_force::search_params::allow_approximate` defaults to false. True permits,
but does not force, this backend. Ordinary `all_neighbors::build` forwards the
option through `graph_build_params::brute_force_params::search_params`.
Mutual-reachability paths are unchanged. The named all-neighbors API remains
FP32-only; FP16 all-neighbors graphs can use brute-force self-search. Self
points are eligible in brute-force self-search. Deprecated overloads remain
default-off. C/Python exposure of the option is a separate integration task.
Adding a search-parameter field requires rebuilding consumers; this is not an
old-header/new-library binary compatibility promise.

## Public dispatch and scheduling

The public policy is capability-based, not a list of benchmark shapes.
On SM120, approximate opt-in calls can use PK2/PK4 for positive int32-sized
query/corpus extents and feature widths, including tails and unequal
query/corpus row counts. Supported metrics and both input types use the same
capability rule. There are no exact-D, exact-row-count or self-search-only
performance exceptions.

When query tiles alone underfill the GPU, a bounded corpus split adds
parallelism. Split count uses query/corpus tile counts and SM count; it is not
an exact-shape dispatch table. The following SM120 scheduling separates
selection from retained-value refinement where this reduces resource pressure:

| Call | Selection | Refinement |
| --- | --- | --- |
| Split FP32 PK2 | TM64 selection-only producer | TM32 retained-value kernel |
| Split FP32 PK4 | TM64 selection-only producer | TM32 retained-value kernel |
| Split FP16 PK2/PK4 | TM64 selection-only producer | TM32 retained-value kernel |
| Unsplit FP16 PK4 | TM64 producer with a selection witness | TM32 retained-value kernel |
| Other calls | Original fused producer/refinement | Within producer |

Split FP16 PK2 retains the fused producer when its actual padded width is at
most 64. Strict staged selection is used only when the corresponding image
and descriptors are compatible. Both stage handles are checked before any
stage is launched, and the same stream orders selection, refinement and
merge/cast. Unavailable or incompatible staged handles retain the original
fused implementation; this is runtime compatibility fallback, not permission
to omit link-time registrations. Unexpected CUDA/allocation errors propagate.

Nonaligned widths use fresh zero-padded storage rounded to a 16-byte row
pitch when representable and supported by the device copy-pitch limit.
Same-pointer query/corpus data shares one packed allocation; independent
search packs both inputs. Allocation, zero-fill, copy, producer,
refinement/merge and release belong to the public call. There is no hidden
cross-call cache. This is a general layout optimization; the relaxed launcher
remains available when preparation limits prevent padding. Zero extension
preserves the metric, while different kernel lowering may change approximate
membership. Padding can expand very narrow rows substantially.

Approximation permission, row-major query/corpus storage, one corpus
partition, an available backend, and valid launcher buffers remain required.
Filtered calls, unsupported metrics/layouts/devices, default-off calls and
launcher ABI limits retain existing cuVS dispatch. The Top1 path is unchanged.

## Compilation and launch contracts

FP32 strict/relaxed cubins use Top1's architecture metadata for SM80, SM86/89,
SM90, SM100 and SM120. Automatic public dispatch is initially enabled only on
SM120. Other exported architectures require runtime validation before enabling
public dispatch. FP16 and staged exports are initially SM120-only. Original
FP16 images use TM64/TN128/TK32; staged images use the stage-specific geometry
above. Strict pitch requires D divisible by 4 for
float or 8 for half, with 16-byte base alignment. `strict` and `relaxed` label
alignment ABIs, not numerical accuracy modes.

Dtype-specific symbols and caches prevent cross-type reuse. Producer IDs are
int32, independent of output index width. Native launch/merge tests cover
int32 and int64 result indices; the public brute-force API returns int64
indices. Merge uses 64-bit row/offset arithmetic and a bounded overflow-safe
grid. Unsupported try-launch calls return false.

## Tests

C++ tests are part of the existing `NEIGHBORS_TEST` target. They cover all four
metrics, retained values, deterministic ordering and ties, duplicates, split
and tile boundaries, short corpora, tails, both input types and index widths,
invalid calls, staged launches, and public opt-in and fallback behavior.
Independent-query tests use separate query/corpus storage and nondefault
streams. Backend-specific tests skip when the required cuTile backend is
unavailable; public fallback tests still run.

CPU-only exporter tests live in `cpp/tests/python/test_fused_topk_export.py`,
which is collected by the existing C++ Python-utilities CI test invocation.
They check typed signatures, descriptor alignment, supported input types,
export matrices, and export argument forwarding without compiling or launching
a kernel. They skip when the optional cuTile Python dependency is unavailable.

```sh
pytest cpp/tests/python
cpp/build/gtests/NEIGHBORS_TEST --gtest_filter='*FusedTopk*'
```

## Physical K8 extension

PK8 serves logical K5 through K8 using a worst-retained-candidate reduction.
All eight candidates are refined and sorted before returning a logical prefix.
Fused single-split calls write the requested prefix directly, with int32 or
int64 indices. Staged calls use physical-width scratch before logical-prefix
output. Two through 128 splits use a warp merge of sorted retained lists.
The long-scan variant skips a corpus tile only if no valid query row can admit
a candidate; equality always enters selection. Its capacity-based guard uses
at least 512 corpus tiles per split with TM32/TN128/TK32, not exact shapes.

For split SM120 calls with padded feature width at least 512, PK8 separates
TM32 selection from TM4 refinement. The long-scan guard above takes precedence
for split calls.
Both stage handles are resolved before either launch. This is a broad resource
policy, not an exact-shape benchmark lookup. Every physical candidate is still
refined before the logical prefix is selected.

Large-batch, single-split SM120 L2/sqrt-L2 calls can also separate selection
from TM4 refinement. This requires at least 10,000 queries, padded feature
width at least 128, and the existing long-scan threshold (512 TN128 corpus
tiles). Selection uses TM64/TN128 for at least 32,768 queries or FP32 padded
width at least 512; otherwise it uses TM32/TN256. Both use TK32, occupancy two,
and the long-scan admission guard. These broad scheduling rules increase
query or corpus reuse without changing candidate capacity or the score formula.
All eight candidates are refined before logical-prefix output and index casting.
Both optional stage handles must resolve before the direct fused shortcut
declines or writes any output; unavailable images retain the fused path.
Small-query calls and cosine/inner product retain their prior scheduling.

The separate strict refinement ABI states the actual padded feature width and
physical scratch width to the compiler: D is divisible by 4 for float or 8 for
half, and scratch has width `splits * 8`. It does not promise aligned row counts
or distinct query/corpus pointers. These promises do not apply to the fused
logical-prefix output. Compiler reassociation may change low FP32 bits or
near-tie ordering; this is not an additional numerical-accuracy guarantee.

PK8 nonaligned input preparation copies stored coordinates and zeroes padding
in one pass; it performs no coordinate arithmetic. Allocation is still fresh
per public call. Packing tests verify stored bits, signed zeros, subnormals,
padding, redzones, input preservation and nondefault-stream execution.

The same approximate membership contract and four metrics apply. PK8 adds no
strict/certified promise, architecture qualification or universal speed claim.
PK2/4 staging, packing and public defaults remain unchanged. Performance must
be reported separately for small search, large search and all-neighbors graphs,
with same-input-dtype baselines and explicit query/corpus layouts.

## Physical K16 extension

PK16 serves logical K9 through K16. On SM120, selection uses a sorted retained
list and bitonic merges, with a separate TM4 direct-FP32 refiner. All 16 retained
candidates are refined and sorted before a logical prefix is returned. The
ordinary selector uses FP32 TM32/TN128/TK32 or native-half TM64/TN128/TK64.
Long strict-layout partitions (at least 512 corpus tiles per split) use
conservative witness checkpoints: 1/2/4/8/12/16 for FP32, 1/4/8/12/16 for half.
Equality still enters selection. Two through 128 splits use a warp list merge.

Wide strict FP32 split scans (padded D at least512 and more than one split)
may use an optional fixed sorting network to order the16 refined values.
The ordinary refiner remains for half, relaxed layouts, narrower inputs and
single-split work. Both retain/refine all16 candidates with the same arithmetic
and returned-value/index key. Missing optional images keep the ordinary
refiner before any launch; this is not an exactness or new precision mode.

PK16 stage registrations are distinct from the PK2/4 stage registrations.
Both stage handles are resolved before enqueue. The API defaults, numerical
contract and metric/index ordering remain the same as the lower families.
These are capability/partition rules, not per-shape benchmark dispatch.
Historical timings are not fresh certification of this combined branch.

The strict SM120 int32 refinement ABI describes physical scratch width
`splits * 16`, plus padded input D divisibility by 4 for float. Half retains
its existing input shape descriptor: adding the redundant shape promise
changed near-tie accumulation ordering in graph validation. Both types retain
their actual padded input stride constraints. Query/corpus row counts remain
unconstrained, and same-pointer self-search remains legal. These promises do
not turn approximate membership into a strict certificate.
