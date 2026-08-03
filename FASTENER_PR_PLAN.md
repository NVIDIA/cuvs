# Fastener CAGRA merge: reusable many-way splitting

## Goal and scope

Fastener merges two or more attached, uncompressed CAGRA indices by preserving their input graphs
and adding a cross-input scaffold. The result is a normal owning CAGRA index with datasets
concatenated in input order. AUTO uses Fastener only after a non-mutating preflight succeeds;
FASTENER rejects unsupported input explicitly; REBUILD retains concatenate-and-build behavior.

The production path supports float, half, int8_t, and uint8_t data, uint32_t graph IDs, and
L2Expanded. Filters, compressed datasets, and other metrics remain on the rebuild path in AUTO.

## Public controls and defaults

    struct merge_params {
      merge_algo algo          = merge_algo::AUTO;
      uint32_t levels          = 2;
      uint32_t root_fanout     = 2;
      uint32_t lower_fanout    = 3;
      double leader_fraction   = 0.02;
      uint32_t max_leaders     = 1024;
      uint32_t leaf_size       = 256;
      uint32_t leaf_degree     = 4;
    };

Supported values are positive levels, fanouts 1 through 32, leader fractions in (0, 1], leader caps
through 8192, leaf sizes 64/128/256, and leaf degrees 4/8. The cap must cover both fanouts. Preflight
evaluates spill width without overflow and requires:

    root_fanout * lower_fanout^(levels - 1) * leaf_degree <= 255

The combined row count times spill width must also fit the membership representation. The
deterministic seed, assignment tile size, and split/leaf GEMM workspaces remain internal.

The benchmark parser exposes every public Fastener field. Maintained presets are the default above,
a higher-quality two-level 4x2/.01/1000 point, and the rebuild control.

## Reusable splitting design

A partition_set owns device-resident (point_id, occurrence) memberships plus compact grouped ranges.
The root is one identity partition; partitioning never copies dataset vectors.

Every configured level invokes the same boundary:

    auto partitions = make_root_partition(res, rows);
    uint32_t occurrence_stride = 1;
    for (uint32_t level = 0; level < params.levels; ++level) {
      auto fanout = level == 0 ? params.root_fanout : params.lower_fanout;
      partitions = split_manyway(
        res, dataset, partitions,
        split_params{fanout, fraction, cap, leaf_size, level, occurrence_stride},
        workspace);
      occurrence_stride *= fanout;
    }
    auto leaves = make_leaves(res, partitions, params.leaf_size);
    auto scaffold = build_leaf_neighbors(
      res, dataset, leaves, origins, spill * params.leaf_degree, params);

split_manyway applies this sequence:

1. Parents no larger than leaf_size are copied with memberships and occurrences unchanged.
2. Each active parent selects clamp(ceil(fraction * parent_size), fanout, cap, parent_size)
   deterministic stratified leaders.
3. Parents are bucketed by a power-of-two padded leader count and divided into assignment tiles.
4. Gathered points and leaders use batched FP32 GEMM and deterministic nearest-leader selection.
   Root and lower levels share these kernels and this host path.
5. One membership is emitted for each selected leader, stable-sorted by a compact child key, and
   reduced into the next grouped ranges.

Occurrence numbering uses the previous spill width as a stride. Completed parents retain their
records while active siblings expand without writing the same candidate slot.

After the exact configured depth, make_leaves slices oversized groups into consecutive leaf_size
ranges. It never performs an implicit geometric split.

## Leaf scaffold and merge phases

For each occurrence, the leaf stage selects directed nearest neighbors whose source input differs
from the row's input. Float and half use FP32 batched Gram matrices. Integer data use centered
INT8/INT32 GEMM. Inputs must fit the leaf GEMM workspace, and integer dimensions must remain within
the worst-case-safe INT32 accumulation bound. That integer limit exceeds 131,000 dimensions and is
not realistic for ANN workloads, so the implementation rejects it before mutation instead of
retaining a second direct-distance leaf path. Leaf degree is a runtime loop bound over
fixed-capacity eight-entry top-k arrays; it is not a kernel template.

A matched three-process A/B comparison found no meaningful benefit from specializing leaf degree.
At degree 4, runtime degree changed geometric-mean merge time by +0.155% across all nine dataset and
fan-in cells, with individual changes from -0.326% to +0.580%. At degree 8 and fan-in 8, the
three-dataset geometric-mean change was +0.277% (individual changes +0.159% to +0.446%). Median
recall deltas were within +/-0.00035. These differences are below the 1% significance threshold, so
the single runtime implementation is retained for simplicity.

Scaffold rows begin filled with their self ID, and leaf selection overwrites available occurrence
slots. The later metric-sorted prefix pass is the only place that removes self IDs and duplicates and
cyclically pads short rows. A matched three-process A/B across five completed Wiki1M and OpenAI
2.3M cells found that removing the redundant pre-compaction improved geometric-mean merge time by
1.058%, with per-cell changes from 0.207% to 2.250% faster. Recall deltas ranged from -0.000242 to
+0.000213. The remaining equivalent fan-in cells were not needed to establish the result. Merge
orchestration then:

1. copies the input datasets into one contiguous device matrix;
2. shifts and appends mixed-degree input graphs;
3. distance-sorts candidates and retains the nearest unique prefix;
4. runs the existing CAGRA graph optimizer; and
5. constructs the owning result without modifying any input index.

The input datasets remain unchanged throughout the merge. The consolidated device matrix becomes
the returned index's owning dataset.

## Initial measured default rationale

The two-level 2x3/.02/1000 point with 256/4 leaves was selected over eight independent binary trees
on Wiki1M, OpenAI 2.3M, and YFCC10M at fan-ins 2, 8, and 32:

| Dataset | Merge-time change | Recall change range |
|---|---:|---:|
| Wiki1M | 42.7-44.0% faster | -0.001875 to +0.000141 |
| OpenAI 2.3M | 35.7-36.6% faster | +0.000217 to +0.003392 |
| YFCC10M | 6.0-6.9% faster | -0.000327 to +0.001247 |

It was faster in all nine measured cells and stayed within the accepted maximum recall loss of
0.002. This is the speed-oriented initial default. The final leaf tuple remains subject to the
focused {64,128,256} x {4,8} post-refactor sweep.

## Verification and final selection

Correctness coverage includes direct split tests, three-level splitting, completed-parent carry,
range coverage and valid IDs, pre-mutation validation, all scalar types, mixed graph degrees,
ownership and dataset ordering, dispatch/fallback behavior, graph shape/range, and search recall.

Build the CAGRA library, NEIGHBORS_ANN_CAGRA_MERGE_TEST, and ANN benchmark target in the CUDA 13.2
devcontainer. Run targeted cuvs-bench parser/constraint tests and git diff --check.

After implementation, sweep {64,128,256} x {4,8} on all datasets at fan-ins 2/8/32. Choose the
lowest geometric-mean merge-time ratio whose worst recall loss versus repeat-8 is no worse than
0.002; retain 256/4 if it remains the winner. Re-run the selected default in three fresh processes
per cell and report median/range merge time, Recall@12, and repeat-8 deltas. At fan-in 8 compare
seeds 7, 42, 1234, and 2025 to ensure the fixed seed is not unusually favorable. Raw CSVs and
temporary sweep configurations stay outside the repository.
