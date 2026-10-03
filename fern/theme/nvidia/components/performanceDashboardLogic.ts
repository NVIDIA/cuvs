/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { BenchmarkRow } from "./benchmarkData";

export type { BenchmarkRow };

export type FilterKey =
  | "algo"
  | "gpu"
  | "cpu"
  | "bs"
  | "topk"
  | "totalVectors"
  | "dims"
  | "dtype"
  | "mode";

export type SortMode = "default" | "desc" | "asc";

export const BUCKET_ORDER = ["[90-95%)", "[95-99%)", ">=99%"] as const;
export const BUCKET_LABEL: Record<string, string> = {
  "[90-95%)": "90%",
  "[95-99%)": "95%",
  ">=99%": "99%",
};

export const FILTER_KEYS: FilterKey[] = [
  "algo",
  "gpu",
  "cpu",
  "bs",
  "topk",
  "totalVectors",
  "dims",
  "dtype",
  "mode",
];

const NV_GREEN_SHADES = [
  "#76B900",
  "#558700",
  "#3E6600",
  "#A3D82C",
  "#C7E867",
  "#8FC733",
  "#2A4A00",
  "#DFF1A8",
];
const INTEL_BLUE_SHADES = [
  "#0071C5",
  "#004F8F",
  "#00355E",
  "#3393D6",
  "#66AEE0",
  "#99C9EA",
  "#002038",
  "#CCE4F4",
];

function shortGpu(g: string | number | null): string {
  return String(g)
    .replace(" Blackwell", " BSE")
    .replace("-SXM4-80GB", "")
    .replace("-SXM-80GB", "")
    .replace("-80GB", "");
}

function shortCpu(c: string | number | null): string {
  const match = String(c).match(/^(\d+x\s+)?([^,]+)/);
  return match ? match[2].trim() : String(c);
}

function colorFor(key: string, isGpu: boolean): string {
  const palette = isGpu ? NV_GREEN_SHADES : INTEL_BLUE_SHADES;
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) | 0;
  }
  return palette[Math.abs(hash) % palette.length];
}

function uniq<T>(values: T[]): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const value of values) {
    const key = String(value);
    if (!seen.has(key) && key !== "" && key !== "NA") {
      seen.add(key);
      out.push(value);
    }
  }
  return out;
}

export function labelFor(filterKey: FilterKey, value: string | number): string {
  if (filterKey === "gpu") return shortGpu(value);
  if (filterKey === "cpu") return shortCpu(value);
  if (filterKey === "bs") return `bs=${value}`;
  if (filterKey === "topk") return `k=${value}`;
  if (filterKey === "totalVectors") {
    const n = Number(value);
    if (n >= 1e9) return `${n / 1e9}B`;
    if (n >= 1e6) return `${n / 1e6}M`;
    if (n >= 1e3) return `${n / 1e3}K`;
    return String(value);
  }
  if (filterKey === "dims") return `${value}D`;
  return String(value);
}

export function labelPlural(filterKey: FilterKey): string {
  const labels: Record<FilterKey, string> = {
    algo: "algorithms",
    gpu: "GPUs",
    cpu: "CPUs",
    bs: "batch sizes",
    topk: "TopK values",
    totalVectors: "sizes",
    dims: "dim sizes",
    dtype: "dtypes",
    mode: "modes",
  };
  return labels[filterKey];
}

export function buildFilterOptions(
  rows: BenchmarkRow[],
): Record<FilterKey, (string | number)[]> {
  return {
    algo: uniq(rows.map((r) => r["cuVS Algo"])).sort() as string[],
    gpu: uniq(
      rows.filter((r) => r["Hardware Type"] === "GPU").map((r) => r["GPU"]),
    ).sort() as string[],
    cpu: uniq(
      rows.filter((r) => r["Hardware Type"] === "CPU").map((r) => r["CPU"]),
    ).sort() as string[],
    bs: uniq(rows.map((r) => r["Search Batch Size"])).sort(
      (a, b) => Number(a) - Number(b),
    ) as number[],
    topk: uniq(rows.map((r) => r["TopK"])).sort(
      (a, b) => Number(a) - Number(b),
    ) as number[],
    totalVectors: uniq(rows.map((r) => r["Total Vectors"])).sort(
      (a, b) => Number(a) - Number(b),
    ) as number[],
    dims: uniq(rows.map((r) => r["Dimensions"])).sort(
      (a, b) => Number(a) - Number(b),
    ) as number[],
    dtype: uniq(rows.map((r) => r["dtype"])).sort() as string[],
    mode: uniq(rows.map((r) => r["Mode"])).sort() as string[],
  };
}

export function defaultFilters(
  options: Record<FilterKey, (string | number)[]>,
): Record<FilterKey, (string | number)[]> {
  const first = (arr: (string | number)[]) => (arr.length ? [arr[0]] : []);
  return {
    algo: first(options.algo),
    gpu: first(options.gpu),
    cpu: first(options.cpu),
    bs: first(options.bs),
    topk: first(options.topk),
    totalVectors: first(options.totalVectors),
    dims: first(options.dims),
    dtype: first(options.dtype),
    mode: first(options.mode),
  };
}

export function applyFilters(
  rows: BenchmarkRow[],
  filters: Record<FilterKey, (string | number)[]>,
): BenchmarkRow[] {
  return rows.filter((row) => {
    if (!filters.algo.includes(row["cuVS Algo"] as string | number)) return false;
    if (!filters.bs.includes(row["Search Batch Size"] as string | number)) return false;
    if (!filters.topk.includes(row["TopK"] as string | number)) return false;
    if (!filters.totalVectors.includes(row["Total Vectors"] as string | number)) {
      return false;
    }
    if (!filters.dims.includes(row["Dimensions"] as string | number)) return false;
    if (!filters.dtype.includes(row["dtype"] as string | number)) return false;
    if (!filters.mode.includes(row["Mode"] as string | number)) return false;
    if (row["Hardware Type"] === "GPU") {
      return filters.gpu.includes(row["GPU"] as string | number);
    }
    return filters.cpu.includes(row["CPU"] as string | number);
  });
}

export interface SeriesLegendItem {
  label: string;
  color: string;
  isGpu: boolean;
}

export interface ChartDataset {
  label: string;
  backgroundColor: string | string[];
  borderColor: string | string[];
  borderWidth: number;
  data: (number | null)[];
  _seriesMeta?: { label: string; key: string | null }[];
  _sortedSlot?: boolean;
}

export interface ChartDataResult {
  labels: string[];
  datasets: ChartDataset[];
  sorted: boolean;
  allSeries: SeriesLegendItem[];
  anchorBucket?: string;
}

export function buildChartData(
  rows: BenchmarkRow[],
  metricCol: string,
  sortMode: SortMode = "default",
): ChartDataResult {
  const seriesMap: Record<
    string,
    {
      label: string;
      color: string;
      isGpu: boolean;
      data: Record<string, number>;
    }
  > = {};

  for (const row of rows) {
    const isGpu = row["Hardware Type"] === "GPU";
    const hw = isGpu ? shortGpu(row["GPU"]) : shortCpu(row["CPU"]);
    const key =
      `${isGpu ? "A-GPU" : "B-CPU"}|${hw}|${row["cuVS Algo"]}` +
      `|bs${row["Search Batch Size"]}|k${row["TopK"]}`;
    if (!seriesMap[key]) {
      seriesMap[key] = {
        label:
          `${hw} · ${row["cuVS Algo"]} · bs=${row["Search Batch Size"]} · k=${row["TopK"]}`,
        color: colorFor(
          `${hw}${row["cuVS Algo"]}${row["Search Batch Size"]}${row["TopK"]}`,
          isGpu,
        ),
        isGpu,
        data: {},
      };
    }
    seriesMap[key].data[String(row["Recall Range"])] = Number(row[metricCol]);
  }

  const keys = Object.keys(seriesMap).sort();
  const labels = BUCKET_ORDER.map((bucket) => BUCKET_LABEL[bucket]);
  const allSeries = keys.map((key) => {
    const series = seriesMap[key];
    return { label: series.label, color: series.color, isGpu: series.isGpu };
  });

  if (sortMode === "default") {
    return {
      labels,
      sorted: false,
      allSeries,
      datasets: keys.map((key) => {
        const series = seriesMap[key];
        return {
          label: series.label,
          backgroundColor: series.color,
          borderColor: series.color,
          borderWidth: 1,
          data: BUCKET_ORDER.map((bucket) => series.data[bucket] ?? null),
        };
      }),
    };
  }

  const perBucketRank: Record<string, { key: string; value: number }[]> = {};
  for (const bucket of BUCKET_ORDER) {
    const entries: { key: string; value: number }[] = [];
    for (const key of keys) {
      const value = seriesMap[key].data[bucket];
      if (value != null) entries.push({ key, value });
    }
    entries.sort((a, b) =>
      sortMode === "desc" ? b.value - a.value : a.value - b.value,
    );
    perBucketRank[bucket] = entries;
  }

  let maxN = 0;
  for (const bucket of BUCKET_ORDER) {
    maxN = Math.max(maxN, perBucketRank[bucket].length);
  }

  const datasets: ChartDataset[] = [];
  for (let slot = 0; slot < maxN; slot++) {
    const dataArr: (number | null)[] = [];
    const bgArr: string[] = [];
    const metaArr: { label: string; key: string | null }[] = [];
    for (const bucket of BUCKET_ORDER) {
      const entry = perBucketRank[bucket][slot];
      if (entry) {
        dataArr.push(entry.value);
        bgArr.push(seriesMap[entry.key].color);
        metaArr.push({ label: seriesMap[entry.key].label, key: entry.key });
      } else {
        dataArr.push(null);
        bgArr.push("#ccc");
        metaArr.push({ label: "(no data)", key: null });
      }
    }
    datasets.push({
      label: `Rank ${slot + 1}`,
      backgroundColor: bgArr,
      borderColor: bgArr,
      borderWidth: 1,
      data: dataArr,
      _seriesMeta: metaArr,
      _sortedSlot: true,
    });
  }

  let anchorBucket: string | undefined;
  for (const bucket of BUCKET_ORDER) {
    if (perBucketRank[bucket].length > 0) {
      anchorBucket = bucket;
      break;
    }
  }

  let orderedSeries = allSeries;
  if (anchorBucket) {
    const orderByKey: Record<string, number> = {};
    perBucketRank[anchorBucket].forEach((entry, idx) => {
      orderByKey[entry.key] = idx;
    });
    const ranked = keys
      .filter((key) => orderByKey[key] != null)
      .sort((a, b) => orderByKey[a] - orderByKey[b]);
    const stragglers = keys.filter((key) => orderByKey[key] == null);
    orderedSeries = [...ranked, ...stragglers].map((key) => ({
      label: seriesMap[key].label,
      color: seriesMap[key].color,
      isGpu: seriesMap[key].isGpu,
    }));
  }

  return {
    labels,
    datasets,
    sorted: true,
    allSeries: orderedSeries,
    anchorBucket,
  };
}

export function describeFilterSelection(
  filters: Record<FilterKey, (string | number)[]>,
  options: Record<FilterKey, (string | number)[]>,
): string {
  const parts: string[] = [];

  const describe = (
    key: FilterKey,
    emptyLabel: string,
    format: (values: (string | number)[]) => string,
  ) => {
    if (filters[key].length === 0) parts.push(emptyLabel);
    else if (filters[key].length < options[key].length) parts.push(format(filters[key]));
  };

  describe("algo", "no algos", (values) => values.join(", "));
  describe("bs", "no bs", (values) => `bs=${values.join("/")}`);
  describe("topk", "no k", (values) => `k=${values.join("/")}`);
  describe("totalVectors", "no dataset size", (values) =>
    values.map((value) => labelFor("totalVectors", value)).join("/"),
  );
  describe("dims", "no dims", (values) => values.map((value) => `${value}D`).join("/"));
  describe("dtype", "no dtype", (values) => values.join("/"));
  describe("mode", "no mode", (values) => values.join("/"));

  if (filters.gpu.length && filters.gpu.length < options.gpu.length) {
    parts.push(`GPUs: ${filters.gpu.map((value) => labelFor("gpu", value)).join(", ")}`);
  } else if (filters.gpu.length === 0) {
    parts.push("GPUs: none");
  }

  if (filters.cpu.length && filters.cpu.length < options.cpu.length) {
    parts.push(`CPUs: ${filters.cpu.map((value) => labelFor("cpu", value)).join(", ")}`);
  } else if (filters.cpu.length === 0) {
    parts.push("CPUs: none");
  }

  return parts.length ? parts.join(" · ") : "all rows (unfiltered)";
}

export function fmtBar(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)}K`;
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2);
}
