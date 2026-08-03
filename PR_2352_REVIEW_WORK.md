# PR #2352 review response: experiments and worktrees

Working notes for [NVIDIA/cuvs#2352](https://github.com/NVIDIA/cuvs/pull/2352) ("[WIP] Fast CAGRA
Index Merge"), covering dantegd's `CHANGES_REQUESTED` review of 2026-07-28 (10 inline comments plus
an API note in the review body).

This document records, per comment: what was asked, whether it needs a code change or a
measurement, which worktree holds the work, and what any experiment actually found.

---

## Worktree layout

One worktree per comment that requests a change, all branched from the PR head `5e3f436d`, so each
thread can be developed, reviewed, and pushed independently rather than as one mixed diff.

**Branch state as of 2026-07-30.** The PR branch has moved to `e788608e`
("Merge branch 'main' into landrumb/cagra-fastener-merge"), and C1, C3, C4 and C5 have been landed
upstream by the author as `bf219633` ("round-robin leaf splitting + more tests") and `bf331a2f`
("Device OOM fallback for AUTO"), including all the tests written here. C6 and C8 were rebased onto
the new head and are still outstanding.

| Comment | Worktree | Status |
|---|---|---|
| C1 `scaffold:818` | `c1-leaf-origin-mixing` | **Landed upstream** in `bf219633` |
| C2 `merge:374` | `c2-candidate-headroom` | **Experiment done — no change needed** |
| C3 `merge:427` | `c3-auto-oom-fallback` | **Landed upstream** in `bf331a2f` |
| C4 `merge:330` | `c4-preflight-sort-degree` | **Landed upstream** |
| C5 `test:568` | `c5-recall-coverage` | **Landed upstream** |
| C6 `scaffold:606` | `c6-tile-rows-derivation` | In the working tree, uncommitted; also `bb128d06` in the worktree |
| C7 `scaffold:391` | `c7-aligned-consolidation` | Analysed — deferral recommended, see below |
| C8 `scaffold:1249` | `c8-redundant-sync` | In the working tree, uncommitted |

C6 and C8 are validated together on `e788608e` at **23/23**.

**C6 was briefly lost.** It was committed as `488ab76e` ("inferred tile_rows"), then dropped when the
branch was reset back to `e788608e`; `leader_bytes` was absent from both HEAD and the working tree.
Restored into the working tree. `488ab76e` remains reachable and `bb128d06` holds it in the worktree,
so nothing was actually lost — but it would not have shipped.

### C8: a lifetime "fix" was proposed, then withdrawn after checking rmm

I initially flagged the bare sync removal as unsafe — `scaffold` is a lambda-local temporary that
the copy kernels read, and the sync was what guaranteed they finished before it was destroyed — and
hoisted it to function scope. **That was wrong, and the hoist was reverted.**

Checking rmm rather than reasoning from first principles,
`stream_ordered_memory_resource::do_deallocate` (`stream_ordered_memory_resource.hpp:142-152`) does:

```cpp
auto stream_event = get_event(strm);
RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, strm.value()));
stream_free_blocks_[stream_event].insert(block);
```

Freeing records a CUDA event on the freeing stream and files the block under that stream. Same-stream
reuse is ordered behind the already-enqueued kernels; cross-stream reuse waits on the event. The
scaffold's stream and the copy kernels' stream both come from the same `res`, so the block cannot be
handed out until the kernels complete. This is rmm's documented contract, not incidental behaviour.

**And the hoist had a real cost.** The scaffold is `rows * spill * leaf_degree` uint32s, i.e.
`rows * 96` bytes with the default controls:

| Dataset | Scaffold |
|---|---:|
| wiki-1m | 96 MB |
| openai-2.3m | 223 MB |
| yfcc-10m | **960 MB** |

Scoped to the lambda it is freed *before* `sort_knn_graph_device_inplace`, `cap_sorted_graph` and
`optimize` allocate their own graphs. Hoisting kept it alive across all three, adding up to ~960 MB
to peak device memory on a 10M-row merge — directly against C7, which is dantegd's peak-memory
thread. Trading a gigabyte of peak to avoid depending on a documented allocator guarantee is a bad
deal.

**Final shape of C8:** drop the sync, keep the tight lambda scope, and document the stream-ordered
reliance at both ends. Exactly what dantegd proposed. The staged edit had also left a
trailing-whitespace line at `cagra_merge_scaffold.cuh:1301`, which would have failed the pre-commit
hook; that is fixed.

Verified present on the branch: round-robin striding, the `bad_alloc` fallback, the
`kMaxSortDegree` preflight rejection, `count_connected_components`,
`IdenticalInputsAlignedToLeafSizeStayConnected`, `MembershipsRemainAscendingWithinEachPartition`,
`PreflightRejectsCandidateDegreeAboveSortLimit` and the three `run_fastener_merge_recall` cases.
Verified absent: the `tile_rows` derivation (C6) and the sync removal (C8).

Changes were validated in batches in the integration checkout rather than one build per worktree,
since a fresh per-worktree CMake configure plus libcuvs build costs ~10 minutes each. C4 and C8 were
validated together on top of C1; C3, C4's test and C5 together. They touch disjoint code, but each
branch should still get a solo CI run before merge.

**Build caveats — two ways a "passing" validation lied.**

1. One run reported "19/19 passed" against a *stale* binary: the build had failed with
   `sccache: error: Timed out waiting for server startup`, and ninja's failure was hidden by the
   output filter. Fixed with `sccache --start-server`. Check the test *count*, not just the pass
   status — it rose to 23 once C4's and C5's tests were added.
2. Another reported nothing at all because the shell's working directory was a *different worktree*,
   so `cd cpp/build` failed and neither ninja nor the test binary ran. The filtered output showed
   only the echo between them, which reads like success at a glance.

Both share a root cause: filtering build output to `error:`/`FAILED:` hides the case where the build
never happened. Prefer absolute paths, and confirm the expected test count on every run.

Two comments (`scaffold:758`, `scaffold:657`) ask questions rather than request changes, so they get
measurements instead of worktrees; a worktree follows only if the numbers justify one.

`~/cuvs-fastener-pr` stays the integration checkout. It holds the configured CUDA build
(`cpp/build`, arch 90-real, sccache) and is where experiment arms are compiled, since a fresh
per-worktree configure plus libcuvs build costs ~10 minutes each. Experiment-only code is always
gated behind an environment variable so a single matched binary can run every arm, and is stripped
before the change lands in its worktree.

### Running the built test binary

The build's rmm must be preloaded or the binary picks up conda's and dies on an ABI mismatch
(`undefined symbol: _ZN3rmm10_RMM_26_1013device_bufferD1Ev`):

```
cd ~/cuvs-fastener-pr/cpp/build
LD_PRELOAD=$PWD/_deps/rmm-build/librmm.so ./gtests/NEIGHBORS_ANN_CAGRA_MERGE_TEST
```

---

## Benchmark harness

All merge-time / recall numbers come from `/raid/blandrum/fastener-leaf-ab/LEAF_BENCH`, adapted from
the existing `fastener-tf32-ab/recall/recall_bench.cu` with a stream-synced timer added around
`cagra::merge` (the original measured recall only). It deserializes pre-built fan-in 8 shards from
`/raid/blandrum/fastener-pivot-results/{wiki-1m,openai-2m,yfcc-10m}-parts`, so partition build time
is excluded and only the merge is timed, then computes Recall@12 against brute-force ground truth
derived from the merged index's own dataset.

- `run_sweep.sh` — leaf-construction arms (C1). `ITOPK` defaults to 160.
- `run_cap_sweep.sh` — candidate-cap arms (C2).

Environment-variable switches used by the experiments, all stripped before anything lands:

| Variable | Values | Comment |
|---|---|---|
| `CUVS_FASTENER_LEAF_FALLBACK` | `baseline` / `roundrobin` / `resplit` / `big512` / `big1024` | C1 |
| `CUVS_FASTENER_LEAF_STATS` | `1` | C1 diagnostics |
| `CUVS_FASTENER_CAP_MODE` | `headroom` | C2 |
| `CUVS_FASTENER_SPLIT_STATS` | `1` | Q1 / Q2 |

### itopk matters more than expected

The first C1 sweep used the library-default itopk and showed recall gains of +0.0012 to +0.0060.
Re-running at **itopk=160** shrank those by 3–20x, because a stronger search routes around missing
cross-input edges. Any recall claim from this PR should state its itopk; the default-itopk numbers
overstate the benefit of scaffold-quality changes. All results in this document are itopk=160 unless
noted.

### Reproducibility caveat

Absolute merge times drift between sweeps — Wiki1M baseline measured 545.6 ms, 514.5 ms and
512.9 ms in three different sweeps of the same code. Only within-sweep comparisons are meaningful,
which is why every arm is re-measured in every sweep rather than compared against a stored baseline.

Both write to a per-configuration results directory and append, so each run gets a fresh directory.
Arms are interleaved within each dataset and repetition so machine drift hits them equally, and
every cell is repeated 3x with the median reported. Absolute times drift a few percent between
sweeps, so arms are only ever compared **within** a sweep.

Datasets: Wiki1M (768d, float), OpenAI 2.3M (1536d, float), YFCC10M (192d, uint8), all at fan-in 8.

---

## C1 — `cagra_merge_scaffold.cuh:818`, single-origin leaves *(resolved)*

> "Since the sorts are stable, rows within a partition can remain grouped by their source index.
> Slicing an oversized partition into consecutive chunks can then produce leaves containing rows
> from only one input, and the leaf kernel skips every same-origin pair. [...] Could we either keep
> splitting or deterministically mix origins before slicing? A regression test for that case would
> also be helpful."

### Confirmed mechanism

Four facts compose into the bug:

1. Root memberships are the identity in ascending row order (`scaffold:286`).
2. Every regroup is a `thrust::stable_sort_by_key` (`scaffold:765`) and tiles emit in parent input
   order (`scaffold:261`, `:432`), so by induction **every partition's membership list stays
   ascending in consolidated row id**.
3. `copy_input_datasets` concatenates inputs in order (`merge:36-59`) and `initialize_origins_kernel`
   labels `[offsets[p], offsets[p+1])` with `p` (`scaffold:1048`), so origins are contiguous row-id
   blocks — ascending row id *is* origin-sorted order.
4. Positional slicing (`scaffold:818`) therefore yields single-origin leaves, and
   `manyway_leaf_gram_knn_kernel:925` skips every same-origin pair. Unfilled slots keep their
   self-ID prefill, which `deduplicate_graph_prefix_kernel:1278` strips — net zero cross-input edges.

### This fires on the shipped defaults

Instrumented run, Wiki1M at fan-in 8, default knobs:

```
ranges=40147  oversized_ranges=6597  oversized_rows=3097996
leaves=49733  single_origin_leaves=3007  single_origin_rows=316036
```

16% of final partitions overflow `leaf_size`, and **6% of all leaves are single-origin** — 316k
membership slots contributing no cross-input edges. Not merely a degenerate-input corner case.

### Arms evaluated

| Arm | Description |
|---|---|
| `baseline` | Current consecutive slicing |
| `roundrobin` | Deal members round-robin across the same number of leaves |
| `resplit` | Extra fanout-1 geometric levels until nothing is oversized, then slice |
| `big512` / `big1024` | Widen the cap for oversized partitions only: take them whole up to the cap, round-robin at that wider cap beyond it |

`resplit` uses fanout 1 so the spill width — and therefore scaffold degree and the `uint8_t`
candidate-width bound — is unchanged and the arm stays comparable.

### Results, itopk=160, 3 reps, medians

Sweep 1 (`results-itopk160-3arm/`):

| Dataset | Arm | Merge ms | Δ | Recall@12 | Δ | Spread |
|---|---|---:|---:|---:|---:|---:|
| wiki-1m | baseline | 545.6 | — | 0.991683 | — | 1.1e-4 |
| | roundrobin | 546.6 | +0.18% | 0.991875 | +0.000192 | 4.1e-5 |
| | resplit | 857.7 | +57.2% | 0.992083 | +0.000400 | 2.5e-5 |
| openai-2m | baseline | 2271.5 | — | 0.958383 | — | 7.5e-5 |
| | roundrobin | 2272.1 | +0.03% | 0.958758 | +0.000375 | 8.4e-5 |
| | resplit | 3292.8 | +45.0% | 0.958158 | −0.000225 | 1.2e-4 |
| yfcc-10m | baseline | 2744.5 | — | 0.981367 | — | 7.5e-5 |
| | roundrobin | 2734.6 | −0.36% | 0.983583 | +0.002216 | 3.4e-5 |
| | resplit | 3915.5 | +42.7% | 0.982783 | +0.001416 | 4.2e-5 |

Sweep 2, adding the widened-cap arms (`results-itopk160/`):

| Dataset | Arm | Merge ms | Δ | Recall@12 | vs baseline | vs roundrobin | Spread |
|---|---|---:|---:|---:|---:|---:|---:|
| wiki-1m | baseline | 514.5 | — | 0.991775 | — | +0.000008 | 2.5e-5 |
| | roundrobin | 516.2 | +0.35% | 0.991767 | −0.000008 | — | 1.0e-4 |
| | big512 | 542.1 | +5.37% | 0.992825 | +0.001050 | +0.001058 | 8e-6 |
| | big1024 | 554.0 | +7.68% | 0.993075 | +0.001300 | +0.001308 | 1.7e-5 |
| openai-2m | baseline | 2203.4 | — | 0.958417 | — | −0.000308 | 6.7e-5 |
| | roundrobin | 2206.6 | +0.15% | 0.958725 | +0.000308 | — | 1.6e-5 |
| | big512 | 2273.0 | +3.16% | 0.960408 | +0.001991 | +0.001683 | 1.0e-4 |
| | big1024 | 2287.2 | +3.80% | 0.960958 | +0.002541 | +0.002233 | 4.2e-5 |
| yfcc-10m | baseline | 2419.1 | — | 0.981433 | — | −0.002125 | 5.9e-5 |
| | roundrobin | 2430.8 | +0.49% | 0.983558 | +0.002125 | — | 1.2e-4 |
| | big512 | 2528.8 | +4.54% | 0.984625 | +0.003192 | +0.001067 | 1.7e-5 |
| | big1024 | 2594.4 | +7.25% | 0.985058 | +0.003625 | +0.001500 | 6.7e-5 |

### Findings

- **`roundrobin` shipped.** Merge time is free (−0.9%..+0.5% across both sweeps, inside the 1%
  significance threshold). ~30 lines confined to `make_leaves`, no new kernel, no extra buffer.
- **Its recall benefit is real only on yfcc** (+0.0021, reproduced in both sweeps). Wiki measured
  +0.000192 then −0.000008 against a ~1e-4 spread, i.e. nothing; openai is a consistent but small
  +0.0003. The much larger gains seen at the library-default itopk do not survive at itopk=160,
  because a stronger search routes around missing cross-input edges. **The PR reply should lead with
  correctness, not recall.**
- **"Keep splitting" is the wrong lever.** `resplit` costs +43..57% merge time, *increases*
  single-origin leaves (Wiki1M 5,956 vs baseline 3,007 — smaller leaves are likelier to be
  single-origin), and loses recall on openai. Worth telling dantegd explicitly, since it was one of
  his two suggestions.
- **Widening the cap is the only arm that gains on all three datasets**, +0.0011..+0.0022 over
  roundrobin for +3..5% merge time. Deliberately *not* shipped: it needs a second KNN kernel plus a
  bucketing driver (~200 lines) on a PR with ten open threads. Recorded as a possible follow-up.

Partition size distribution, which bounds any whole-partition pass:

| Dataset | Oversized | p50 | p90 | p99 | max | Σn² |
|---|---:|---:|---:|---:|---:|---:|
| wiki-1m | 6,597 | 380 | 753 | 1,547 | 4,929 | 1.98e9 |
| openai-2m | 15,315 | 365 | 654 | 1,163 | 3,173 | 3.36e9 |
| yfcc-10m | 65,614 | 369 | 697 | 1,525 | 9,536 | 1.83e10 |

### Change shipped

Commit `3a31c44b` on `landrumb/fastener-c1-leaf-origin-mixing` (252 insertions / 23 deletions):

- `leaf_set` becomes strided views (`starts`/`counts`/`strides`) instead of `[start, end)`, so
  origin mixing needs no permutation buffer — relevant to C7's peak-memory concern.
- `make_leaves` deals each oversized partition round-robin; a range that already fits yields stride
  1 and is bit-identical to before.
- The two consuming kernels index `start + i*stride`.
- Side benefit: removes the tiny trailing leaf (e.g. 257 rows → `{256, 1}`), where the one-row leaf
  hits the `leaf_n <= 1` early-out and contributes nothing. Round-robin gives `{129, 128}`.

### Tests added

- `IdenticalInputsAlignedToLeafSizeGetCrossOriginScaffoldEdges` — dantegd's exact case at the
  scaffold level; asserts every row gets a cross-origin neighbor.
- `IdenticalInputsAlignedToLeafSizeStayConnected` — end-to-end, asserts one weakly-connected
  component via a new `count_connected_components` union-find helper (`expect_valid_graph` only
  checks in-range and no self-loop, so it could not catch this).
- `MembershipsRemainAscendingWithinEachPartition` — pins the ordering invariant the fix relies on,
  so a future non-stable sort trips a test instead of silently weakening the guarantee.

Verified red-to-green: on `baseline` all 256 rows get zero cross-input edges and the merged graph
has **2 disconnected components**; on the fix, 1. Full suite 19/19 on every fix arm.

---

## C2 — `cagra_merge.cuh:374`, candidate headroom before `optimize()`

> "Are we capping this a little too early? [...] this nearest prefix cap can remove every cross
> index edge before `graph::optimize()` gets a chance to consider it. It also leaves the optimizer
> with the same input and output degree, so there's no candidate headroom for robust pruning."

You replied on-thread that the rationale was "if you had no cross-input edges that were closer than
the edges you started with, you probably didn't need them", that it saved optimize time without a
real recall impact, and that you would **re-check and post numbers**. This experiment is that
re-check.

Arms (`CUVS_FASTENER_CAP_MODE`):

- `default` — cap to `graph_degree`, then `optimize` runs degree-to-degree (current behaviour).
- `headroom` — cap to `min(max(intermediate_graph_degree, graph_degree), candidate_width)`, letting
  `optimize` prune with headroom, exactly as dantegd proposed.

At the harness settings (`graph_degree=64`, `intermediate=128`) the merged candidate width is
`base_degree + scaffold_degree = 64 + 24 = 88`, so `headroom` caps at 88 and `optimize` runs 88→64.

### Results, itopk=160, 3 reps, medians (`results-cap-itopk160/`)

| Dataset | Mode | Merge ms | Δ | Recall@12 | Δ | Spread |
|---|---|---:|---:|---:|---:|---:|
| wiki-1m | default | 512.9 | — | 0.991842 | — | 2.5e-5 |
| | headroom | 597.8 | **+16.56%** | 0.991792 | −0.000050 | 3.4e-5 |
| openai-2m | default | 2201.9 | — | 0.958742 | — | 7.5e-5 |
| | headroom | 2402.6 | **+9.12%** | 0.957100 | **−0.001642** | 8.3e-5 |
| yfcc-10m | default | 2418.7 | — | 0.983567 | — | 2.5e-5 |
| | headroom | 3259.6 | **+34.76%** | 0.982217 | **−0.001350** | 3.4e-5 |

### Finding: the current early cap is correct — keep it

Adding candidate headroom is worse on **both** axes:

- **Merge time +9% to +35%.** `optimize` doing 88→64 instead of 64→64 is materially more work, and
  it scales with the extra width.
- **Recall drops**, and not marginally: −0.0016 on openai and −0.0014 on yfcc, roughly 20x the
  rep spread. Wiki is −0.00005, i.e. neutral.

This directly supports the rationale you gave on-thread — "if you had no cross-input edges that were
closer than the edges you started with, you probably didn't need them" — and confirms the earlier
evaluation you recalled. It also refines it: the effect is not merely "saves time without hurting
recall", it is "saves time **and** helps recall".

Worth noting for the reply, because it is counterintuitive and answers dantegd's "no candidate
headroom for robust pruning" concern head-on: handing `optimize` *more* candidates makes the result
**worse**, not better. The extra candidates admitted by the wider cap are by construction the
*farthest* ones — the prefix is distance-sorted — so they dilute the pruning and reverse-edge budget
with edges the nearest-prefix cap had correctly already discarded.

The concern's premise — "this cap can remove every cross index edge before `optimize()` gets a
chance to consider it" — is factually right; it just turns out to be the desired behaviour. When an
input's own local neighbors are all closer than every scaffold candidate, that row does not need a
cross-input edge, and forcing one in costs both time and quality.

**No change required.** Worktree `c2-candidate-headroom` retained only to hold the experiment patch;
the shipping code is unchanged.

---

## C3 — `cagra_merge.cuh:427`, AUTO fallback on OOM

> "Should AUTO fall back to rebuild if Fastener runs out of memory? [...] changing the existing
> overload to AUTO means some merges that previously succeeded may now fail. I think explicit
> FASTENER should still fail, but AUTO retrying the rebuild path would preserve the existing
> behavior."

Change requested, no experiment needed — this is a behaviour-preservation argument, and dantegd is
right that `merge_rebuild` already has a `std::bad_alloc` catch with a host-memory fallback
(`merge:179`) that the Fastener path bypasses.

Change written in the worktree: AUTO wraps `merge_fastener` in a `try`/`catch (std::bad_alloc)` and
retries `merge_rebuild`; explicit FASTENER is untouched and still propagates. `rmm::bad_alloc` and
`rmm::out_of_memory` both derive from `std::bad_alloc`, and this matches the existing catch in
`merge_rebuild`.

The retry is safe because Fastener never mutates its inputs — it only reads them and allocates its
own output — so a failed attempt unwinds cleanly and leaves the indices usable by the rebuild path.
That property is worth stating explicitly in the reply, since it is what makes the fallback sound.

**Test gap, deliberately left open.** Forcing a deterministic OOM in a unit test is awkward. The
viable approach is an `rmm::mr::limiting_resource_adaptor` sized so Fastener's consolidated dataset
plus GEMM workspace exceeds the cap while `merge_rebuild` still succeeds via its host-memory
fallback. That is doable but sensitive to allocator behaviour and could be flaky in CI, so it is
flagged here rather than written blind.

---

## C4 — `cagra_merge.cuh:330`, missing `kMaxSortDegree` preflight check

> "the graph passed to `sort_knn_graph_device_inplace()` can't exceed `kMaxSortDegree` (1024). [...]
> Could we include `max_input_degree + scaffold_degree <= kMaxSortDegree` in the eligibility check?"

Change requested, no experiment needed. Straightforwardly correct: preflight enumerates 24
conditions but none bounds the summed degree against the sorter's limit.

Verified the mechanism: `kMaxSortDegree = raft::WarpSize * 32 = 1024` (`graph_core.cuh:180`), and
`select_sort_kernel` `RAFT_FAIL`s past it (`graph_core.cuh:196-200`) — that is, *after*
`merge_fastener` has consolidated the dataset and built the scaffold, so the failure comes well
into a mutating operation, and under AUTO the rebuild fallback has already been passed.

Change written in the worktree (uncommitted): a preflight rejection of
`max_input_degree + scaffold_degree > cagra::detail::graph::kMaxSortDegree`, placed next to the
existing capacity check. This is the exact condition dantegd proposed.

Still to do before committing: a case in
`InvalidManywayOptionsFailPreflightWithoutMutation`, which currently enumerates 17 bad configs but
covers only knob ranges, not the degree-sum limit. Note that list asserts `eligible == false`
without checking `reason`, so a new case there could pass for the wrong reason — worth pinning the
reason string, as `LeafGemmLimitsRejectOversizedWorkspaceDimensions` does.

---

## C5 — `test_merge_fastener.cu:568`, recall coverage gaps

> "Could we add recall coverage for at least one non-float dtype and one merge with more than two
> inputs? [...] That feels important if AUTO is going to become the default."

Change requested, no experiment needed. Confirmed gap: the suite has exactly one
`INSTANTIATE_TEST_CASE_P`, varying only `graph_build_algo`, so 2 instances — both float, both
2-input (the fixture always splits into 2 at `splite_ratio = 0.55`). The dtype tests
(`SupportsAllScalarTypes`) assert graph structure and dataset order only, never recall.

**Why not extend the shared fixture.** `AnnCagraIndexMergeTest` hardcodes two parts
(`index0`, `index1` at `ann_cagra.cuh:1450`) and is shared with the other CAGRA merge tests, so
making it n-way would ripple beyond this PR. Instead the worktree adds a self-contained
`run_fastener_merge_recall<DataT>(n_inputs, min_recall)` that builds the parts directly with
`cagra::build`, merges with FASTENER, searches, and checks against `naive_knn` ground truth via
`eval_neighbours` — the same helpers the fixture uses.

It slices the database into consecutive ranges and merges in order, so the merged index reproduces
the original row order and ground-truth indices apply without remapping. Three cases added:

| Test | dtype | Fan-in |
|---|---|---|
| `Uint8TwoWayMergeSearchRecall` | uint8 | 2 |
| `FloatFourWayMergeSearchRecall` | float | 4 |
| `Uint8FourWayMergeSearchRecall` | uint8 | 4 |

That covers both axes dantegd asked for, plus their combination.

---

## C6 — `cagra_merge_scaffold.cuh:606`, derive `tile_rows`

> "Could we derive `tile_rows` from the dimension and padded leader count instead of always using
> 2048? Preflight only checks the leaf-GEMM shape, while assignment needs roughly
> `(tile_rows + padded_leaders) * dim + tile_rows * padded_leaders` float elements."

Change requested. dantegd is right that the sizing is inconsistent: `assign_bucket` computes
`bytes_per_batch` for a single tile of the fixed 2048-row height and then asserts it fits
`gemm_workspace_bytes`, while preflight validates only the *leaf* GEMM shape. A wide dimension or a
wide padded leader count therefore passes preflight and trips the assertion mid-merge.

Change written in the worktree: solve the same budget for the tile height instead of asserting
against it. The workspace holds the gathered leader matrix once plus per-row point, dot-product and
selection storage, so

```
tile_rows = min(assignment_tile_rows, (gemm_workspace_bytes - leader_bytes) / row_bytes)
```

and the merge is rejected only when the leader matrix plus a single point row cannot fit — exactly
the boundary dantegd proposed. **This only ever shrinks the tile**, so every configuration that fits
today keeps the 2048 default and is bit-identical.

Interaction with Q2 worth noting in the reply: shrinking the tile raises the tile count, which
raises the leader re-gather volume measured in Q2. Since the derivation only shrinks when the
alternative is a hard failure, that is the right trade, but the two threads should be answered
together.

---

## C7 — `cagra_merge_scaffold.cuh:391`, aligned consolidation

> "for dimensions that aren't naturally 16-byte aligned, this allocates and copies a second full
> consolidated dataset while all input datasets are still alive. Would it be practical to
> consolidate directly into aligned storage and pass a row stride through the scaffold/sort kernels?"

Change requested. Analysed rather than implemented, for the reason below.

### When this actually triggers

`make_aligned_dataset` (`common.hpp:388`) computes
`required_stride = round_up(dim * sizeof(T), lcm(16, sizeof(T))) / sizeof(T)`, and
`make_strided_dataset` returns a **non-owning view with no copy** when `required_stride` already
equals the source stride. So the second full copy happens only when
`dim * sizeof(T)` is not a multiple of 16:

| dtype | Copy when |
|---|---|
| `float` | `dim % 4 != 0` |
| `half` | `dim % 8 != 0` |
| `int8_t` / `uint8_t` | `dim % 16 != 0` |

**None of the three benchmark datasets trigger it** — Wiki1M `dim=768` (float, 768%4=0), OpenAI
`dim=1536` (float, %4=0), YFCC `dim=192` (uint8, 192%16=0). Nor do the common ANN dimensions
(128, 384, 768, 960, 1024). It bites on awkward dims such as uint8 `dim=100` or float `dim=17`.

It is also confined to `attach_dataset_on_build = true`; the other branch attaches an
`empty_dataset` and never makes the copy.

### Cost when it does trigger

Peak device memory during the merge goes from roughly `2N` to `3N`, where `N` is the dataset size:
the caller's inputs, the dense consolidated copy, and the aligned copy are all live at once, since
`make_strided_dataset` takes its source by const reference and so keeps it alive across the copy.
dantegd's proposal would bring the peak back to about `2N`, a ~33% reduction — but only in the
unaligned case.

### Feasibility blocker

Consolidating directly into aligned storage is straightforward on the merge side —
`copy_input_datasets` already takes an arbitrary destination pitch, so it would just take
`required_stride` instead of `dim`. The obstacle is downstream: the scaffold kernels index
`dataset[point * dim + d]` densely, and more importantly

```cpp
void sort_knn_graph_device_inplace(
  raft::resources const& res,
  cuvs::distance::DistanceType metric,
  raft::device_matrix_view<const DataT, int64_t, raft::row_major> dataset,   // dense, not strided
  raft::device_matrix_view<IdxT, int64_t, raft::row_major> knn_graph)
```

takes a dense `raft::row_major` view (`graph_core.cuh:1001-1005`). Threading a stride through would
mean widening that signature to a padded or strided layout — shared CAGRA code well outside the
merge path, and outside what this PR should be changing.

### Recommendation

Reply with the trigger condition and the ~33% figure, note that it affects no common dimension and
none of the benchmarked datasets, and propose deferring to a follow-up that widens
`sort_knn_graph_device_inplace` to accept a padded layout. Worth doing, but not here.

---

## C8 — `cagra_merge_scaffold.cuh:1249`, redundant sync

> "Is this synchronization needed? The helper has no local temporary buffers consumed by the copy
> kernels, and the caller immediately sorts the returned graph on the same resource stream."

Change requested. dantegd's reasoning about the *helper* is correct — `append_to_input_graphs`
allocates exactly one device buffer, `graph`, which it returns by move, so it owns no temporary
consumed by the copy kernels.

But there is a subtlety worth raising on-thread rather than silently dropping the sync. The
*caller's* `scaffold` is a temporary inside a lambda:

```cpp
auto merged_graph = [&] {
  auto scaffold = merge_scaffold::build<T>(...);
  return merge_scaffold::append_to_input_graphs<T, IdxT>(..., scaffold.view());
}();
```

The copy kernels read `scaffold`, and it is destroyed when the lambda returns. Today the sync inside
the helper is what makes that safe. Removing the sync leaves correctness resting on the allocator's
stream-ordered deallocation rather than on anything visible in the code.

Change written in the worktree (uncommitted): drop the sync **and** hoist `scaffold` out of the
lambda so its lifetime provably outlives the merged graph's use. That gets the intended saving
without trading an explicit sync for an implicit allocator guarantee.

---

## Q1 — `cagra_merge_scaffold.cuh:758`, work spent on padded leaders

> "Curious, have you measured how much work is going to padded leaders here? A parent just over a
> power-of-two boundary could nearly double the gathered leader storage, dot-product GEMM, distance
> materialization, and `select_k` width."

You replied comparing against non-batched GEMM (substantially slower) and per-size batching
(moderately faster, but more LoC than felt worth it). That answers *which batching strategy*, but
not dantegd's actual question — *how much* work the power-of-two padding wastes.

Measured with `CUVS_FASTENER_SPLIT_STATS=1`, which is computable entirely from the host-side
`split_plan`: assignment work per parent scales with the padded leader count, so the waste fraction
is `1 - Σ(rows × leaders) / Σ(rows × padded_leaders)`.

| Dataset | Level | Split parents | Leader-pad waste |
|---|---:|---:|---:|
| wiki-1m | 0 | 1 | **2.34%** |
| | 1 | 863 | **30.53%** |
| openai-2m | 0 | 1 | 2.34% |
| | 1 | 987 | **29.98%** |
| yfcc-10m | 0 | 1 | 2.34% |
| | 1 | 1000 | **20.32%** |

### Finding: dantegd's concern is real, and concentrated at the lower level

- **Level 0 is nearly free.** The root always takes `max_leaders = 1000` leaders, padded to 1024, so
  the waste is a fixed 24/1024 = 2.34% regardless of dataset.
- **Level 1 wastes 20–31%** of the assignment gather / GEMM / distance-materialization / `select_k`
  width. With ~1000 split parents whose leader counts follow `ceil(0.02 × size)`, many land just
  above a power of two — exactly the case dantegd described.

This is the number he asked for, and it is consistent with your recollection that per-size batching
was "moderately faster": ~30% of *one stage* at *one level*, not ~30% of the merge. Whether that
justifies the extra LoC is a judgement call, but the reply should quote the measurement rather than
leave the question unanswered.

---

## Q2 — `cagra_merge_scaffold.cuh:657`, per-tile leader re-gather

> "It looks like every tile of the same parent gathers the same leader matrix again. [...] Could we
> gather each parent's leaders once and reuse them across its tiles, or did benchmarking show that
> the added bookkeeping costs more?"

Not yet answered on-thread. Measured with the same `CUVS_FASTENER_SPLIT_STATS=1` instrumentation:
`regather_factor` is `Σ(tiles × padded × dim) / Σ(padded × dim)`, i.e. how many times the leader
matrix is gathered versus the once it needs to be.

| Dataset | Level | Tiles | Widest parent | Re-gather factor | Redundant bytes |
|---|---:|---:|---:|---:|---:|
| wiki-1m | 0 | 489 | 1,000,000 | **489x** | 1.54 GB |
| | 1 | 1,441 | 28,847 | 3.24x | 0.39 GB |
| openai-2m | 0 | 1,134 | 2,321,096 | **1,134x** | 7.13 GB |
| | 1 | 2,755 | 37,308 | 4.83x | 3.15 GB |
| yfcc-10m | 0 | 4,883 | 10,000,000 | **4,883x** | 3.84 GB |
| | 1 | 10,265 | 178,893 | 14.68x | 5.32 GB |

### Finding: mechanism confirmed, absolute cost below the noise floor

The re-gather is exactly as dantegd describes, and the *factor* is dramatic — the root is a single
parent spanning up to 4,883 tiles, each re-gathering the identical leader matrix.

But the *absolute* volume does not justify a fix. Totals are 1.9 GB (wiki), 10.3 GB (openai),
9.2 GB (yfcc) of redundant gather. Counting both the read and the contiguous write, and taking
~2 TB/s of achievable HBM bandwidth on the H100, that is roughly **2 ms, 10 ms and 9 ms**
respectively — against merge times of 513 ms, 2202 ms and 2419 ms, i.e. **0.4%, 0.5% and 0.4%**.

Every one of those sits under the 1% significance threshold this PR has used throughout, so caching
each parent's leaders would not produce a measurable speedup while adding real bookkeeping. That is
a direct answer to "or did benchmarking show that the added bookkeeping costs more?" — the saving is
below what the harness can resolve.

**Caveat on rigour:** this is an analytical bound derived from *measured* byte volumes and assumed
bandwidth, not a measured speedup from an implemented cache. It is strong enough to justify not
doing the work, but if dantegd pushes back the honest next step is to implement the cache behind a
flag and measure it, rather than defend the estimate.

**No change proposed.** Worth noting that C6 (`tile_rows` derivation) interacts here: raising
`ASSIGNMENT_TILE_ROWS` would reduce the tile count and shrink this redundancy for free, so the two
threads should be answered together.

---

## API notes from the review body

- **C API** — dantegd noted it still calls the C++ overload without `merge_params`. Addressed by
  commit `5e3f436d` ("updating C API"), already on the branch; worth confirming on-thread.
- **Java API** — you flagged on 2026-07-29 that it still lags and asked whether to fold it into this
  PR or defer. Awaiting dantegd. No worktree until that is decided.

---

## Summary

### Experiments run

| # | Comment | Question | Cells | Answer |
|---|---|---|---:|---|
| 1 | C1 `scaffold:818` | Which fix for single-origin leaves? | 27 | `roundrobin` — free, ships |
| 2 | C1 `scaffold:818` | Does taking oversized partitions whole beat it? | 36 | Yes (+0.001..0.002 recall, +3..5% time) — deferred |
| 3 | C2 `merge:374` | Does candidate headroom help? | 18 | **No** — worse on both axes |
| 4 | Q1 `scaffold:758` | How much work goes to padded leaders? | 3 | 2.3% at level 0, 20–31% at level 1 |
| 5 | Q2 `scaffold:657` | How costly is the per-tile leader re-gather? | 3 | Up to 4,883x factor, but ~0.4–0.5% of merge time |

### Three results worth carrying into the thread replies

1. **C1's recall case is weaker than it first appeared.** At the library-default itopk the gains
   looked like +0.001..+0.006; at itopk=160 only yfcc survives. The fix is justified by
   correctness — 316k membership slots with no cross-input edge on Wiki1M, and two disconnected
   components for identical inputs — not by recall.
2. **C2 refutes the suggestion, with the mechanism explained.** More candidates make the result
   *worse* because a distance-sorted prefix admits only the farthest edges. This is the numbers
   reply promised on-thread.
3. **Q2 confirms the mechanism but not the priority.** A 4,883x re-gather factor sounds alarming and
   is real, yet lands under 1% of merge time. Reporting both halves is more useful than either
   alone.

### State of the tree

- `~/cuvs-fastener-pr` — integration checkout, now at `e788608e`. Holds the configured build and
  this document. Used as the validation sandbox; snapshot and restore its working tree when doing
  so, since the author works here too.
- **Landed upstream:** C1, C3, C4, C5.
- **Ready, uncommitted in the integration checkout:** C6 and C8, validated together at 23/23.
- **No change needed:** C2 — refuted by measurement.
- **Deferral recommended:** C7 — blocked on a shared signature, see that section.

All eight change-requesting comments are addressed. C7 is the only one carrying no code change, and
that is a reasoned deferral rather than an omission.
- Raw benchmark logs under `/raid/blandrum/fastener-leaf-ab/` in `results/`,
  `results-itopk160-3arm/`, `results-itopk160/` and `results-cap-itopk160/`, plus the harness
  (`LEAF_BENCH`, `leaf_bench.cu`, `run_sweep.sh`, `run_cap_sweep.sh`). All outside the repository.

### Remaining work

1. Commit C6 and C8 from the integration checkout (or cherry-pick `bb128d06`). Both are green at
   23/23 on `e788608e`. Note C6 was already lost once to a branch reset.
2. Post replies for the threads that are answerable now: C2 (refuted, with the table), Q1 (padding
   waste figures), Q2 (re-gather factor and why it does not matter), C6 (paired with Q2), C7
   (trigger condition and the deferral rationale).
3. C3 — optionally add the `limiting_resource_adaptor` OOM test noted in that section.
4. Decide the Java API question with dantegd.
5. Consider the `big512` follow-up from C1, the only arm that improved recall on all three datasets.
6. Update `FASTENER_PR_PLAN.md`, which still describes consecutive slicing.

## Merging main, and the `check-c-abi` failure

Both had the same root cause: the branch was behind `NVIDIA/cuvs` main.

### Remotes point at the wrong org

`origin` is `landrumb/cuvs` and `upstream` is `rapidsai/cuvs`, both sitting at `ad9e2d2a`, while the
PR targets **`NVIDIA/cuvs`**, which was at `f72199e3`. The project appears to have moved orgs, so
`git fetch` looked clean against a stale base while GitHub reported `CONFLICTING`. Worth fixing:

```
git remote set-url upstream https://github.com/NVIDIA/cuvs.git
```

### The conflict was one file, and positional

Only `cpp/src/neighbors/cagra.cuh`. Both sides append a function immediately after the existing
`merge()` overload — this branch adds `merge()` taking `merge_params` (plus a `merge` entry in
`CUVS_INST_CAGRA_MERGE`), main adds a multi-partition `search()` overload plus
`#include <cuvs/core/bitset.hpp>`. No semantic overlap: main never touches the instantiation macro
and this branch never touches `search`. Resolved by keeping both.

### `check-c-abi` was a false alarm caused by the same staleness

The analyzer reported four **removed** functions:

```
cuvsResourcesSetWorkspacePool      cuvs/core/c_api.h
cuvsRMMAsyncMemoryResourceEnable   cuvs/core/c_api.h
cuvsCagraSearchMultiPartition      cuvs/neighbors/cagra.h
cuvsSelectK                        cuvs/selection/select_k.h
```

None belong to this PR. They were added upstream *after* the branch's base, so comparing the branch
against a `release/26.08` baseline made them look deleted. Confirmed by symbol presence —
`branch=0, main=1, merged=1` for all four — so the merge fixes the check with no code change.

This PR's own C API work is purely additive and does not break ABI: a new `cuvsCagraMergeAlgo` enum,
a new `cuvsCagraMergeParams` struct, and `cuvsCagraMergeParamsCreate` / `Destroy` /
`cuvsCagraMergeWithParams`. The existing `cuvsCagraMerge` signature is unchanged.

Side note: the job's `Comment on PR with results` step also failed, with a 404 posting to
`/repos/NVIDIA/cuvs/issues//comments` — an empty PR number. That is a workflow bug on cross-repo
PRs, which is why no bot comment appeared explaining the failure.

### Outcome

Merge commit `652a7ddc` (parents `a44ca425` + `f72199e3`). Full rebuild of 3268 targets including
main's new multi-partition kernels, then 23/23 tests, then pushed to
`origin/landrumb/cagra-fastener-merge`. The PR moved from `CONFLICTING` to `MERGEABLE`.

## Documentation drift

`FASTENER_PR_PLAN.md` still states that `make_leaves` "slices oversized groups into consecutive
`leaf_size` ranges", which C1 makes untrue. Update before the PR description is finalized.
