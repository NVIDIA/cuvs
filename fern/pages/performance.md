import PerformanceDashboard from "@/theme/nvidia/components/PerformanceDashboard";

# Performance

Compare cuVS-Bench performance across hardware and algorithms. Adjust the filters to see comparative performance for index build time, search throughput (QPS), and search latency.

<PerformanceDashboard />

## About this data

These results come from [cuVS-Bench](cuvs_bench/index.md) runs on cuVS 26.04. The source dataset is [MIRACL](https://huggingface.co/datasets/miracl/miracl-corpus), embedded with [Llama Nemotron Embed 1B](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2). Each row represents a tuned configuration bucketed by recall range (`90%`, `95%`, `99%`). Green bars denote GPU SKUs; blue bars denote CPU SKUs.

To reproduce or extend these benchmarks, see the [cuVS Bench guide](cuvs_bench/index.md).
