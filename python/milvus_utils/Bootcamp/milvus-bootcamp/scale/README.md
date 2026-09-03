# Dataset Scaler

Dataset scaling and controlled noise injection tool for large vector
datasets (**FBIN / FVECs**).

Designed for ANN benchmarking scenarios where synthetic scaling and
deterministic perturbation are required.

------------------------------------------------------------------------

## Overview

This tool:

-   Replicates a base dataset by a configurable **scale factor**
-   Applies controlled **±noise**
-   Supports **multi-threaded execution (MT-only)**
-   Guarantees deterministic output layout (under fixed configuration)
-   Preserves format correctness for FBIN and FVECs
-   Optionally renormalizes vectors
-   Supports FBIN sharding
-   Can emit structured JSON metadata

------------------------------------------------------------------------

## Output Layout (Batch-Major Guarantee)

The scaler produces **batch-major output layout**.

For each input batch:

    batch_0: replica_0, replica_1, ..., replica_(S-1)
    batch_1: replica_0, replica_1, ...
    ...

This ordering is guaranteed by the planner and is:

-   Deterministic
-   Independent of thread scheduling
-   Stable across runs (if configuration is unchanged)

Total output vectors:

    total_n = base_n * scale

------------------------------------------------------------------------

## Supported Formats

### Input

-   `.fbin` (headered)
-   `.fvec` / `.fvecs`

### Output

-   Same format as input
-   FBIN may produce multiple shard files
-   FVECs always produces a single file

------------------------------------------------------------------------

## FBIN Format

    [int32 count][int32 dim][float32 data...]

-   Payload starts at byte offset 8
-   For sharded outputs, each shard is a fully valid standalone FBIN
    file
-   Headers are finalized after threaded writes complete

------------------------------------------------------------------------

## FVECs Format

Repeated records:

    [int32 dim][float32 * dim]

-   Each vector embeds its own dimension prefix
-   No global header
-   Sharding is **not supported**

------------------------------------------------------------------------

## Noise Schemes

Noise is always applied as:

    x + sign * noise_amplitude

Replica 0 is copied as-is (unless normalization is enabled).

### `ALL`

-   All dimensions perturbed
-   Each dimension gets independent ±noise

### `PARTIAL` (default)

For each vector:

-   Exactly `floor(dim / 2)` dimensions selected randomly
-   Only selected dimensions perturbed
-   Signs chosen independently

------------------------------------------------------------------------

## Determinism

Determinism is achieved via:

-   Batch-major deterministic planning
-   Per-work-unit seed derivation based on:
    -   `replica_id`
    -   `batch_id`
    -   base seed

If `--seed` is not provided, randomness is non-deterministic.

------------------------------------------------------------------------

## Determinism Boundary

Bytewise-identical output is guaranteed **only if all configuration
parameters remain unchanged**, including:

-   Input dataset
-   Scale
-   Noise scheme
-   Noise amplitude
-   Seed
-   Batch size
-   Normalization flag
-   Tolerance
-   Shard size

Changing `batch_size` changes batch boundaries and therefore:

-   Changes `batch_id`
-   Changes per-work-unit seeds
-   Produces different bytewise results

Therefore, batch size is part of the determinism contract.

------------------------------------------------------------------------

## Normalization

If `--normalize` is enabled:

-   All replicas (including replica 0) are L2-renormalized
-   Only rows whose norm deviates by more than `--tolerance` are rescaled
-   No vectors are discarded

Default tolerance:

    3.7e-4

If `--normalize` is not set:

-   No renormalization occurs

------------------------------------------------------------------------

## Sharding (FBIN Only)

-   Controlled via `--shard-size`
-   If not specified, shard size defaults to int32 max
-   Headers are backpatched after writes complete

Shard naming:

    output.fbin
    output_part000.fbin
    output_part001.fbin
    ...

Each shard is a valid FBIN file.

------------------------------------------------------------------------

## Multi-Threaded Execution

Controlled via:

    -w / --workers

-   Default: number of available CPU cores
-   Must be >= 1
-   Each worker thread owns its own file descriptors
-   Output write safety is guaranteed by non-overlapping planned ranges

------------------------------------------------------------------------

## Usage

### Basic Example

``` bash
python -m scale.cli   -i base.fbin   -o scaled.fbin   -s 5   -a 0.02   -m PARTIAL
```

### Multi-threaded + normalization + metadata

``` bash
python -m scale.cli   -i base.fbin   -o scaled.fbin   -s 10   -a 0.02   -m PARTIAL   -n   -t 3.7e-4   -w 8   -x 1234   -j summary.json
```

------------------------------------------------------------------------

## Command-line Arguments

  Argument                 Description
  ------------------------ ------------------------------------------
  `-i, --input-dataset`    Input dataset (.fbin or .fvec/.fvecs)
  `-o, --output-dataset`   Output dataset (same format as input)
  `-s, --scale`            Number of replicas (>=1)
  `-b, --batch-size`       Batch size (default 8192)
  `-a, --noise`            Noise amplitude (default 0.02)
  `-m, --noise-scheme`     `ALL` or `PARTIAL` (default PARTIAL)
  `-n, --normalize`        Enable L2 renormalization
  `-t, --tolerance`        Renormalization tolerance (default 3.7e-4)
  `-k, --shard-size`       FBIN only: vectors per shard
  `-x, --seed`             Deterministic base seed
  `-w, --workers`          Number of worker threads (>=1)
  

------------------------------------------------------------------------

## Ground Truth

-   This tool does **not** generate or adjust ground truth
-   Ground truth must be recomputed separately
