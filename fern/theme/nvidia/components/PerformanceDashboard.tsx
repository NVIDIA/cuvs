/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { BENCHMARK_ROWS } from "./benchmarkData";
import {
  applyFilters,
  BUCKET_LABEL,
  buildChartData,
  buildFilterOptions,
  ChartDataResult,
  defaultFilters,
  describeFilterSelection,
  FILTER_KEYS,
  FilterKey,
  fmtBar,
  labelFor,
  labelPlural,
  SeriesLegendItem,
  SortMode,
  type BenchmarkRow,
} from "./performanceDashboardLogic";

declare global {
  interface Window {
    Chart?: any;
    ChartDataLabels?: { default: unknown };
  }
}

const CHART_JS_URL =
  "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js";
const CHART_DATALABELS_URL =
  "https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js";

const FILTER_LABELS: Record<FilterKey, string> = {
  algo: "Algorithm",
  gpu: "GPU SKU",
  cpu: "CPU SKU",
  bs: "Search Batch Size",
  topk: "TopK",
  totalVectors: "Total Vectors",
  dims: "Dimensions",
  dtype: "dtype",
  mode: "Mode",
};

const FILTER_COLOR_MODE: Partial<Record<FilterKey, "gpu" | "cpu">> = {
  gpu: "gpu",
  cpu: "cpu",
};

const CHARTS: {
  id: string;
  title: string;
  metric: string;
  yLabel: string;
}[] = [
  {
    id: "chart-build",
    title: "Index Build Time (s)",
    metric: "Index Build Time (s)",
    yLabel: "Build time (s)",
  },
  {
    id: "chart-qps",
    title: "Search Throughput (QPS)",
    metric: "Mean Search Throughput (QPS)",
    yLabel: "QPS",
  },
  {
    id: "chart-lat",
    title: "Search Latency (ms)",
    metric: "Mean Search Latency (ms)",
    yLabel: "Latency (ms)",
  },
];

let chartLoaderPromise: Promise<void> | null = null;

function loadChartJs(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if (window.Chart && window.ChartDataLabels) return Promise.resolve();
  if (chartLoaderPromise) return chartLoaderPromise;

  chartLoaderPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = CHART_JS_URL;
    script.async = true;
    script.onload = () => {
      const plugin = document.createElement("script");
      plugin.src = CHART_DATALABELS_URL;
      plugin.async = true;
      plugin.onload = () => {
        if (window.Chart && window.ChartDataLabels) {
          const pluginModule =
            (window.ChartDataLabels as { default?: unknown }).default ??
            window.ChartDataLabels;
          window.Chart.register(pluginModule);
        }
        resolve();
      };
      plugin.onerror = () => reject(new Error("Failed to load Chart.js datalabels plugin"));
      document.head.appendChild(plugin);
    };
    script.onerror = () => reject(new Error("Failed to load Chart.js"));
    document.head.appendChild(script);
  });

  return chartLoaderPromise;
}

function toggleValue<T>(values: T[], value: T): T[] {
  const index = values.indexOf(value);
  if (index === -1) return [...values, value];
  return values.filter((_, i) => i !== index);
}

function MultiSelectFilter({
  filterKey,
  options,
  selected,
  onChange,
}: {
  filterKey: FilterKey;
  options: (string | number)[];
  selected: (string | number)[];
  onChange: (values: (string | number)[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const colorMode = FILTER_COLOR_MODE[filterKey];

  useEffect(() => {
    const onDocumentClick = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("click", onDocumentClick);
    return () => document.removeEventListener("click", onDocumentClick);
  }, []);

  const summary =
    selected.length === 0
      ? `No ${labelPlural(filterKey)}`
      : selected.length === options.length
        ? `All ${labelPlural(filterKey)}`
        : selected.map((value) => labelFor(filterKey, value)).join(", ");

  return (
    <div className="pd-filter-group" ref={rootRef}>
      <label className="pd-title">{FILTER_LABELS[filterKey]}</label>
      <div className="pd-ms-dropdown">
        <button
          type="button"
          className={`pd-ms-btn${open ? " open" : ""}`}
          onClick={() => setOpen((current) => !current)}
        >
          <span className="pd-ms-summary">{summary}</span>
          <span className="pd-ms-count-pill">
            {selected.length}/{options.length}
          </span>
          <span className="caret">▾</span>
        </button>
        {open ? (
          <div className="pd-ms-panel open">
            <div className="pd-ms-actions">
              <button type="button" onClick={() => onChange([...options])}>
                Select all
              </button>
              <button type="button" onClick={() => onChange([])}>
                Clear
              </button>
            </div>
            {options.map((value) => (
              <label className="pd-ms-option" key={String(value)}>
                <input
                  type="checkbox"
                  checked={selected.includes(value)}
                  onChange={() => onChange(toggleValue(selected, value))}
                />
                {colorMode ? (
                  <span className={`pd-swatch ${colorMode}`} />
                ) : null}
                <span>{labelFor(filterKey, value)}</span>
              </label>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function SeriesLegend({ chartData }: { chartData: ChartDataResult | null }) {
  if (!chartData?.allSeries.length) {
    return <span className="item">No series to show — adjust filters.</span>;
  }

  return (
    <>
      {chartData.sorted && chartData.anchorBucket ? (
        <span className="legend-caption">
          Legend ordered by {BUCKET_LABEL[chartData.anchorBucket]} bucket:
        </span>
      ) : null}
      {chartData.allSeries.map((item: SeriesLegendItem) => (
        <span className="item" key={item.label}>
          <span className="swatch-sq" style={{ background: item.color }} />
          <span>{item.label}</span>
        </span>
      ))}
    </>
  );
}

function ChartPanel({
  chartId,
  title,
  metric,
  yLabel,
  subhead,
  rows,
  sortMode,
  onSortModeChange,
  chartsReady,
}: {
  chartId: string;
  title: string;
  metric: string;
  yLabel: string;
  subhead: string;
  rows: BenchmarkRow[];
  sortMode: SortMode;
  onSortModeChange: (mode: SortMode) => void;
  chartsReady: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<any>(null);
  const chartData = useMemo(
    () => (rows.length ? buildChartData(rows, metric, sortMode) : null),
    [rows, metric, sortMode],
  );

  useEffect(() => {
    if (!chartsReady || !canvasRef.current || !chartData || !window.Chart) {
      return;
    }

    const totalBars = chartData.datasets.reduce(
      (count, dataset) =>
        count + dataset.data.filter((value) => value != null).length,
      0,
    );
    const showLabels = totalBars > 0 && totalBars <= 30;

    chartRef.current?.destroy();
    chartRef.current = new window.Chart(canvasRef.current, {
      type: "bar",
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 18 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label(context: any) {
                const value = context.parsed.y;
                const dataset = context.dataset as ChartDataResult["datasets"][number];
                const meta = dataset._seriesMeta?.[context.dataIndex];
                const label = meta?.label ?? dataset.label;
                if (value == null) return `${label}: (no data)`;
                return `${label}: ${value >= 1000 ? value.toLocaleString() : value.toFixed(2)}`;
              },
            },
          },
          datalabels: {
            display: showLabels,
            anchor: "end",
            align: "end",
            clamp: true,
            offset: 2,
            color: "#333",
            font: { size: 9, weight: "600" },
            formatter: (value: number | null) => fmtBar(value),
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Recall" },
            grid: { display: false },
          },
          y: {
            title: { display: true, text: yLabel },
            beginAtZero: true,
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [chartData, chartsReady, yLabel]);

  return (
    <div className="pd-chart-card">
      <div className="pd-card-head">
        <h3>{title}</h3>
        <div className="pd-sort-toggle">
          <span className="pd-sort-label">Sort in-bucket:</span>
          {(["default", "desc", "asc"] as SortMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              className={sortMode === mode ? "active" : ""}
              onClick={() => onSortModeChange(mode)}
            >
              {mode === "default"
                ? "Default"
                : mode === "desc"
                  ? "Highest first"
                  : "Lowest first"}
            </button>
          ))}
        </div>
      </div>
      <div className="pd-subhead">{subhead}</div>
      <div className="pd-chart-wrap">
        <canvas id={chartId} ref={canvasRef} />
      </div>
      <div className="pd-series-legend">
        <SeriesLegend chartData={chartData} />
      </div>
    </div>
  );
}

export default function PerformanceDashboard() {
  const [mounted, setMounted] = useState(false);
  const allRows = BENCHMARK_ROWS;
  const options = useMemo(() => buildFilterOptions(allRows), [allRows]);
  const [filters, setFilters] = useState<Record<FilterKey, (string | number)[]>>(() =>
    defaultFilters(options),
  );
  const [sortModes, setSortModes] = useState<Record<string, SortMode>>({
    "chart-build": "default",
    "chart-qps": "default",
    "chart-lat": "default",
  });
  const [chartsReady, setChartsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    loadChartJs()
      .then(() => setChartsReady(true))
      .catch((err: Error) => setError(err.message));
  }, []);

  const filteredRows = useMemo(
    () => applyFilters(allRows, filters),
    [allRows, filters],
  );

  const filterSummary = useMemo(
    () => `${describeFilterSelection(filters, options)} · ${filteredRows.length} rows`,
    [filters, filteredRows.length, options],
  );

  if (!mounted) {
    return (
      <div className="performance-dashboard">
        <div className="pd-status">Loading dashboard…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="performance-dashboard">
        <div className="pd-status err">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="performance-dashboard">
      <div className="pd-filters">
        {FILTER_KEYS.map((filterKey) => (
          <MultiSelectFilter
            key={filterKey}
            filterKey={filterKey}
            options={options[filterKey]}
            selected={filters[filterKey]}
            onChange={(values) =>
              setFilters((current) => ({ ...current, [filterKey]: values }))
            }
          />
        ))}
      </div>

      {CHARTS.map((chart) => (
        <ChartPanel
          key={chart.id}
          chartId={chart.id}
          title={chart.title}
          metric={chart.metric}
          yLabel={chart.yLabel}
          subhead={filterSummary}
          rows={filteredRows}
          sortMode={sortModes[chart.id]}
          onSortModeChange={(mode) =>
            setSortModes((current) => ({ ...current, [chart.id]: mode }))
          }
          chartsReady={chartsReady}
        />
      ))}
    </div>
  );
}
