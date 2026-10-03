# Ground Truth (GT) Generation

This tool computes **exact ground truth (GT)** for vector search benchmarks using a **streaming, multi‑GPU** approach. It is designed for large datasets stored in FBIN format and supports both ```ibin``` and ```parquet``` format.
---

## Overview

- Exact top‑K nearest‑neighbor ground truth
- Streaming over base vectors (does not load the full base dataset into memory)
- Query‑partitioned, multi‑GPU execution
- Fail‑fast preflight validation to avoid late failures after long runs
- Two output formats:
  - **IBIN** (legacy, compact, int32‑limited)
  - **Parquet** (int64‑safe)

---

## Requirements

### Hardware

- At least **one CUDA‑capable GPU**
- NVIDIA driver correctly installed (`nvidia-smi` must detect GPUs)

> CPU‑only execution is **not supported**.

### Software

- Python **>= 3.10**
- PyTorch built with CUDA support
- NumPy
- tqdm
- For Parquet output only:
  - `pyarrow`

### PyTorch installation example (CUDA 12.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

See https://pytorch.org/get-started/locally/ for other CUDA versions.

---

## Input Formats

### Base vectors

- One or more `.fbin` files
- Multiple FBIN files may be provided as a comma‑separated list
- All FBIN files must have identical dimensionality

### Query vectors

- `.fvecs` or `.fbin`

---

## Running the Script

Basic usage:

```bash
python groundtruth/main.py \
  --base <base_dataset.fbin[,base_shard2.fbin,...]> \
  --query <queries.fvecs | queries.fbin>
```

Show all options:

```bash
python groundtruth/main.py -h
```

Common options:

- `--k` – top‑K neighbors (default: 100)
- `--batch_size` – base vectors per batch (default: 100000)
- `--metric` – distance metric: `l2` or `ip`
- `--out_prefix` – output file prefix (default: `groundtruth`)
- `--out_format` – `ibin` or `parquet` (default: `ibin`)

---

