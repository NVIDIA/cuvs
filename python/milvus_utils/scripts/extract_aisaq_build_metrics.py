#!/usr/bin/env python3
"""Extract AISAQ index-build metrics from Milvus datanode logs into a CSV.

For each segment that the datanode builds, this script captures the key
log markers used by the Kioxia AiSAQ paper methodology:

  * Vamana graph build time         -> aux_utils.cpp:1794 "Training graph cost: <s>"
  * Reported AiSAQ index build time -> scope_metric.cpp:49 "slow function CreateIndex done with duration <s>s"
                                       (the native CGO CreateIndex duration)
  * Gross AiSAQ index build time    -> task_index.go:388 "index building all done" duration=<...>
                                       (Go-level wall time of the whole build job)

Plus useful context:
  * collection / build / segment IDs, num_rows, dim, index_type
  * Training PQ codes cost (aux_utils.cpp:1779)
  * GPU detection + Vamana shard counts (Building with GPU! vs CPU)
  * Output sizes (serializedSize / memSize) from task_index.go:389
  * Vamana on-disk file size (aux_utils.cpp:1270, create_aisaq_layout)
  * AiSAQ build start (diskann_aisaq.cc:313)

Usage:
    python extract_aisaq_build_metrics.py [PATH ...] [-o OUT.csv]

PATH may be a log file or a directory; directories are walked and every
file ending in ``.log`` is parsed. If no path is given, the script reads
from stdin.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z"
)

GO_CREATE_INDEX_RE = re.compile(
    r'task_index\.go:327\].*?"create index".*?'
    r'clusterID=(?P<cluster>[^\]\s]+).*?'
    r'buildID=(?P<build>\d+).*?'
    r'collection=(?P<collection>\d+).*?'
    r'segmentID=(?P<segment>\d+).*?'
    r'currentIndexVersion=(?P<idx_ver>\d+)'
)
GO_BUILD_PARAMS_NUMROWS_RE = re.compile(r'num_rows:(\d+)')
GO_BUILD_PARAMS_DIM_RE = re.compile(r'\bdim:(\d+)')
GO_BUILD_PARAMS_INDEX_TYPE_RE = re.compile(r'key:\\?"index_type\\?"\s+value:\\?"([A-Z0-9_]+)\\?"')

GO_SUCCESS_BUILD_RE = re.compile(
    r'task_index\.go:341\].*?"Successfully build index".*?buildID=(?P<build>\d+)'
)
GO_ALL_DONE_RE = re.compile(
    r'task_index\.go:388\].*?IndexBuildID:\s*(?P<build>\d+).*?'
    r'"index building all done".*?duration=(?P<dur>\S+?)\]'
)
GO_SAVE_FILES_RE = re.compile(
    r'task_index\.go:389\].*?"Successfully save index files".*?'
    r'buildID=(?P<build>\d+).*?'
    r'serializedSize=(?P<ser>\d+).*?memSize=(?P<mem>\d+)'
)

CGO_CREATEINDEX_DONE_RE = re.compile(
    r'scope_metric\.cpp:49\].*?slow function CreateIndex done with duration\s+([\d.]+)s'
)
CGO_TRAIN_GRAPH_RE = re.compile(
    r'aux_utils\.cpp:1794\].*?Training graph cost:\s*([\d.]+)s'
)
CGO_TRAIN_PQ_RE = re.compile(
    r'aux_utils\.cpp:1779\].*?Training PQ codes cost:\s*([\d.]+)s'
)
CGO_GPU_DETECT_RE = re.compile(
    r'aux_utils\.cpp:495\].*?GPU has\s+(\d+)\s*Gib free memory out of\s+(\d+)\s*Gib total'
)
CGO_BUILD_DEVICE_RE = re.compile(
    r'aux_utils\.cpp:608\].*?Building with (GPU|CPU)!\s*R=\s*(\d+)\s+L=\s*(\d+)'
)
CGO_COMPRESSING_RE = re.compile(
    r'aux_utils\.cpp:1727\].*?Compressing\s+(\d+)-dimensional data into\s+(\d+)\s+bytes per vector'
)
CGO_AISAQ_CFG_RE = re.compile(
    r'diskann_aisaq\.cc:313\].*?AiSAQ build Configuration'
)
CGO_VAMANA_FILE_SIZE_RE = re.compile(
    r'aux_utils\.cpp:1270\].*?Vamana index file size=(\d+)'
)


@dataclass
class BuildRecord:
    log_file: str
    datanode: str
    cluster_id: str = ""
    build_id: str = ""
    collection_id: str = ""
    segment_id: str = ""
    num_rows: Optional[int] = None
    dim: Optional[int] = None
    index_type: str = ""

    create_index_ts: Optional[str] = None
    aisaq_cfg_ts: Optional[str] = None
    successful_build_ts: Optional[str] = None
    all_done_ts: Optional[str] = None

    gross_duration_s: Optional[float] = None
    reported_aisaq_duration_s: Optional[float] = None
    training_graph_cost_s: Optional[float] = None
    training_pq_codes_cost_s: Optional[float] = None

    serialized_size_bytes: Optional[int] = None
    mem_size_bytes: Optional[int] = None
    vamana_index_file_bytes: Optional[int] = None
    pq_compress_dim: Optional[int] = None
    pq_bytes_per_vector: Optional[int] = None

    gpu_total_gib: Optional[int] = None
    gpu_free_gib: Optional[int] = None
    gpu_used: str = "no"
    vamana_shards_gpu: int = 0
    vamana_shards_cpu: int = 0

    status: str = "partial"
    attempts: int = 1
    finished: bool = False

    @property
    def key(self) -> tuple:
        """Logical identity of a build, used to merge retries across logs."""
        return (self.collection_id, self.build_id, self.segment_id)

    def completeness(self) -> int:
        """Higher value = more complete. Used to pick the best of N retries."""
        score = 0
        if self.all_done_ts:
            score += 1000
        if self.successful_build_ts:
            score += 100
        if self.reported_aisaq_duration_s is not None:
            score += 10
        if self.training_graph_cost_s is not None:
            score += 5
        if self.serialized_size_bytes is not None:
            score += 1
        return score


def _parse_duration(s: str) -> Optional[float]:
    """Parse Go time.Duration strings like '1h8m16.511s', '54m47.07s', '110.898s'."""
    if not s:
        return None
    s = s.strip().rstrip(",")
    pat = re.compile(r"(\d+(?:\.\d+)?)(h|m|s|ms|us|µs|ns)")
    total = 0.0
    matched = False
    pos = 0
    for m in pat.finditer(s):
        if m.start() != pos:
            return None
        matched = True
        v = float(m.group(1))
        unit = m.group(2)
        if unit == "h":
            total += v * 3600
        elif unit == "m":
            total += v * 60
        elif unit == "s":
            total += v
        elif unit == "ms":
            total += v / 1e3
        elif unit in ("us", "µs"):
            total += v / 1e6
        elif unit == "ns":
            total += v / 1e9
        pos = m.end()
    if not matched or pos != len(s):
        try:
            return float(s)
        except ValueError:
            return None
    return total


def _ts(line: str) -> Optional[str]:
    """Return an Excel-friendly timestamp (YYYY-MM-DD HH:MM:SS.fff) from a log line.

    Excel auto-recognizes ``YYYY-MM-DD HH:MM:SS[.fff]`` as a datetime; the
    docker/k8s logs use ISO-8601 with ``T`` and trailing ``Z`` plus nanosecond
    precision, which Excel does not parse natively. We strip the ``T``/``Z``
    and truncate the fractional seconds to 3 digits (milliseconds).
    """
    m = TS_RE.match(line)
    if not m:
        return None
    date, time, frac = m.group(1), m.group(2), m.group(3)
    if frac:
        ms = (frac + "000")[:3]
        return f"{date} {time}.{ms}"
    return f"{date} {time}"


def _datanode_label(path: Path) -> str:
    """Pick a short identifier for the datanode from the file path."""
    parts = path.resolve().parts
    for p in reversed(parts):
        if "datanode" in p and p not in ("0.log", "datanode"):
            return p
        if p.startswith("datanode-"):
            return p
    return path.parent.name or path.name


class LogParser:
    """Linear log parser.

    Routing rule for native (CGO) log lines that don't carry a buildID:
    they are attributed to the *most recently started, not-yet-finished*
    build seen in this log file. When a new ``create index`` event arrives
    we advance the pointer, so any later CGO output belongs to the new
    attempt - this matches Milvus' single-slot indexnode behavior and
    correctly handles retries (the same buildID can appear twice if a
    previous attempt was interrupted).
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.datanode = _datanode_label(log_path)
        self.records: List[BuildRecord] = []
        self.by_build: Dict[str, BuildRecord] = {}
        self.current: Optional[BuildRecord] = None

    def _route_cgo(self) -> Optional[BuildRecord]:
        """Return the build that should receive the next CGO log line."""
        if self.current is None or self.current.finished:
            return None
        return self.current

    def feed(self, line: str) -> None:
        ts = _ts(line)
        m = GO_CREATE_INDEX_RE.search(line)
        if m:
            rec = BuildRecord(
                log_file=str(self.log_path),
                datanode=self.datanode,
                cluster_id=m.group("cluster").strip('"'),
                build_id=m.group("build"),
                collection_id=m.group("collection"),
                segment_id=m.group("segment"),
                create_index_ts=ts,
            )
            mr = GO_BUILD_PARAMS_NUMROWS_RE.search(line)
            if mr:
                rec.num_rows = int(mr.group(1))
            md = GO_BUILD_PARAMS_DIM_RE.search(line)
            if md:
                rec.dim = int(md.group(1))
            mt = GO_BUILD_PARAMS_INDEX_TYPE_RE.search(line)
            if mt:
                rec.index_type = mt.group(1)
            self.by_build[rec.build_id] = rec
            self.records.append(rec)
            self.current = rec
            return

        m = GO_SUCCESS_BUILD_RE.search(line)
        if m:
            rec = self.by_build.get(m.group("build"))
            if rec:
                rec.successful_build_ts = ts
            return

        m = GO_ALL_DONE_RE.search(line)
        if m:
            rec = self.by_build.get(m.group("build"))
            if rec:
                rec.all_done_ts = ts
                rec.gross_duration_s = _parse_duration(m.group("dur"))
                rec.finished = True
                rec.status = "complete"
            return

        m = GO_SAVE_FILES_RE.search(line)
        if m:
            rec = self.by_build.get(m.group("build"))
            if rec:
                rec.serialized_size_bytes = int(m.group("ser"))
                rec.mem_size_bytes = int(m.group("mem"))
            return

        m = CGO_CREATEINDEX_DONE_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                rec.reported_aisaq_duration_s = float(m.group(1))
            return

        m = CGO_TRAIN_GRAPH_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                rec.training_graph_cost_s = float(m.group(1))
            return

        m = CGO_TRAIN_PQ_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                rec.training_pq_codes_cost_s = float(m.group(1))
            return

        m = CGO_GPU_DETECT_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                rec.gpu_free_gib = int(m.group(1))
                rec.gpu_total_gib = int(m.group(2))
                rec.gpu_used = "yes"
            return

        m = CGO_BUILD_DEVICE_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                device = m.group(1)
                if device == "GPU":
                    rec.vamana_shards_gpu += 1
                    rec.gpu_used = "yes"
                else:
                    rec.vamana_shards_cpu += 1
            return

        m = CGO_COMPRESSING_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                rec.pq_compress_dim = int(m.group(1))
                rec.pq_bytes_per_vector = int(m.group(2))
            return

        m = CGO_AISAQ_CFG_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                rec.aisaq_cfg_ts = ts
                if not rec.index_type:
                    rec.index_type = "AISAQ"
            return

        m = CGO_VAMANA_FILE_SIZE_RE.search(line)
        if m:
            rec = self._route_cgo()
            if rec:
                rec.vamana_index_file_bytes = int(m.group(1))


CSV_COLUMNS = [
    "log_file",
    "datanode",
    "cluster_id",
    "collection_id",
    "build_id",
    "segment_id",
    "index_type",
    "num_rows",
    "dim",
    "pq_compress_dim",
    "pq_bytes_per_vector",
    "status",
    "attempts",
    "create_index_ts",
    "aisaq_cfg_ts",
    "successful_build_ts",
    "all_done_ts",
    "training_pq_codes_cost_s",
    "training_graph_cost_s",
    "reported_aisaq_duration_s",
    "gross_duration_s",
    "serialized_size_bytes",
    "mem_size_bytes",
    "vamana_index_file_bytes",
    "gpu_used",
    "gpu_total_gib",
    "gpu_free_gib",
    "vamana_shards_gpu",
    "vamana_shards_cpu",
]


def dedupe_records(records: List[BuildRecord]) -> List[BuildRecord]:
    """Collapse multiple attempts of the same (collection, build, segment).

    Keeps the most complete attempt (preferring a finished build with
    ``all_done_ts`` over a partial one) and reports the total number of
    attempts seen in the ``attempts`` field.
    """
    groups: Dict[tuple, List[BuildRecord]] = {}
    order: List[tuple] = []
    for r in records:
        if r.key not in groups:
            groups[r.key] = []
            order.append(r.key)
        groups[r.key].append(r)
    out: List[BuildRecord] = []
    for key in order:
        attempts = groups[key]
        best = max(attempts, key=lambda r: (r.completeness(), r.create_index_ts or ""))
        best.attempts = len(attempts)
        out.append(best)
    return out


def discover_log_files(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".log"):
                        out.append(Path(root) / f)
        elif p.is_file():
            out.append(p)
        else:
            print(f"warning: skipping non-existent path: {raw}", file=sys.stderr)
    return sorted(out)


def parse_log(path: Path) -> List[BuildRecord]:
    parser = LogParser(path)
    with path.open("r", errors="replace") as fh:
        for line in fh:
            parser.feed(line)
    return parser.records


def parse_stream(stream, label: str) -> List[BuildRecord]:
    parser = LogParser(Path(label))
    for line in stream:
        parser.feed(line)
    return parser.records


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="Log files or directories to scan. If empty, reads stdin.")
    ap.add_argument("-o", "--output", default="-", help="CSV output path ('-' for stdout, default).")
    ap.add_argument(
        "--all-attempts",
        action="store_true",
        help=(
            "Emit every (collection, build, segment) attempt as its own row, "
            "instead of collapsing retries into a single best-of-N row."
        ),
    )
    ap.add_argument(
        "--include-partial",
        action="store_true",
        help=(
            "When deduplicating, also emit segments that never reached "
            "'index building all done' (no completed retry found). "
            "Off by default; ignored with --all-attempts."
        ),
    )
    args = ap.parse_args(argv)

    records: List[BuildRecord] = []
    if not args.paths:
        records.extend(parse_stream(sys.stdin, label="<stdin>"))
    else:
        files = discover_log_files(args.paths)
        if not files:
            print("error: no log files found", file=sys.stderr)
            return 2
        for f in files:
            print(f"parsing {f} ...", file=sys.stderr)
            records.extend(parse_log(f))

    total_attempts = len(records)
    if args.all_attempts:
        out_records = sorted(records, key=lambda r: (r.create_index_ts or "", r.build_id))
    else:
        deduped = dedupe_records(records)
        if not args.include_partial:
            deduped = [r for r in deduped if r.status == "complete"]
        out_records = sorted(deduped, key=lambda r: (r.create_index_ts or "", r.build_id))

    if args.output == "-":
        out_fh = sys.stdout
        close = False
    else:
        out_fh = open(args.output, "w", newline="")
        close = True
    try:
        writer = csv.DictWriter(out_fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in out_records:
            row = {k: asdict(rec).get(k, "") for k in CSV_COLUMNS}
            for k, v in row.items():
                if v is None:
                    row[k] = ""
            writer.writerow(row)
    finally:
        if close:
            out_fh.close()

    dropped = total_attempts - len(out_records)
    msg = f"wrote {len(out_records)} row(s) from {total_attempts} attempt(s)"
    if dropped:
        msg += f" ({dropped} merged/filtered)"
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
