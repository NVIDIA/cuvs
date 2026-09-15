import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime

import pandas as pd


def monitor_resources(stop_event, log, hw_type='cpu', interval=1.0):
    """
    Monitor CPU/RAM (and per-GPU GPU/VRAM) usage every `interval` seconds
    until `stop_event` is set. Each GPU is logged as its own set of columns:
    gpu{i}_util and gpu{i}_vram_gb.
    """
    import psutil

    handles = []
    if hw_type == 'gpu':
        import pynvml

        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]

    try:
        while not stop_event.is_set():
            cpu_util = psutil.cpu_percent(interval=interval)
            ram_util = psutil.virtual_memory().used / (1024 ** 3)

            sample = {
                'timestamp': time.perf_counter(),
                'wallclock': datetime.now().isoformat(timespec='seconds'),
                'cpu_util': cpu_util,
                'ram_gb': ram_util,
            }

            if hw_type == 'gpu':
                for i, h in enumerate(handles):
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    sample[f'gpu{i}_util'] = util.gpu
                    sample[f'gpu{i}_mem_util'] = util.memory
                    sample[f'gpu{i}_vram_gb'] = mem.used / (1024 ** 3)

            log.append(sample)
    finally:
        if hw_type == 'gpu':
            import pynvml
            pynvml.nvmlShutdown()


def summarize_telemetry(resource_log, stage_prefix, hw_type='cpu'):
    """
    Summarize telemetry data.

    resource_log: list of dict
      Resource monitoring logs stored as a list of dictionaries.
    stage_prefix: str
      Prefix to add to output dictionary, e.g. 'idx_build', 'vec_search'.
    hw_type: str
      Hardware used. Select 'cpu' or 'gpu'.
    """
    results_df = pd.DataFrame(resource_log)
    sys_results = {
        stage_prefix + '_duration_sec': results_df['timestamp'].max()
        - results_df['timestamp'].min(),
        stage_prefix + '_avg_cpu_util': results_df['cpu_util'].mean(),
        stage_prefix + '_max_cpu_util': results_df['cpu_util'].max(),
        stage_prefix + '_max_ram_gb': results_df['ram_gb'].max(),
    }

    if hw_type == 'gpu':
        util_cols = [c for c in results_df.columns if c.endswith('_util') and c.startswith('gpu')]
        vram_cols = [c for c in results_df.columns if c.endswith('_vram_gb')]
        for c in util_cols:
            sys_results[f'{stage_prefix}_avg_{c}'] = results_df[c].mean()
            sys_results[f'{stage_prefix}_max_{c}'] = results_df[c].max()
        for c in vram_cols:
            sys_results[f'{stage_prefix}_max_{c}'] = results_df[c].max()

    return sys_results


def run_loop(output_path, hw_type='gpu', interval=1.0, flush_every=10, duration=None):
    """
    Run telemetry monitoring in a background thread and periodically flush
    collected samples to a CSV file. Runs until SIGINT/SIGTERM or until
    `duration` seconds have elapsed (if provided).
    """
    log = []
    stop_event = threading.Event()

    thread = threading.Thread(
        target=monitor_resources,
        args=(stop_event, log),
        kwargs={'hw_type': hw_type, 'interval': interval},
        daemon=True,
    )
    thread.start()

    def _handle_signal(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    start = time.time()
    last_written = 0
    header_written = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    print(f'[telemetry] logging {hw_type} every {interval}s -> {output_path}')
    try:
        while not stop_event.is_set():
            time.sleep(flush_every)
            if len(log) > last_written:
                new_rows = log[last_written:]
                last_written = len(log)
                df = pd.DataFrame(new_rows)
                df.to_csv(
                    output_path,
                    mode='a',
                    header=not header_written,
                    index=False,
                )
                header_written = True
                print(
                    f'[telemetry] flushed {len(new_rows)} samples '
                    f'(total={last_written}) at {datetime.now().isoformat(timespec="seconds")}'
                )
            if duration is not None and (time.time() - start) >= duration:
                stop_event.set()
    finally:
        stop_event.set()
        thread.join(timeout=interval * 3 + 2)
        if len(log) > last_written:
            df = pd.DataFrame(log[last_written:])
            df.to_csv(
                output_path,
                mode='a',
                header=not header_written,
                index=False,
            )
            print(f'[telemetry] final flush: wrote {len(log) - last_written} samples')


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Log CPU/RAM and per-GPU telemetry to a CSV file.'
    )
    parser.add_argument(
        '-o', '--output',
        default=f'logs/telemetry_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        help='Output CSV path (default: logs/telemetry_<timestamp>.csv).',
    )
    parser.add_argument(
        '--hw', choices=['cpu', 'gpu'], default='gpu',
        help='Hardware to monitor (default: gpu).',
    )
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='Sampling interval in seconds (default: 1.0).',
    )
    parser.add_argument(
        '--flush-every', type=float, default=10.0,
        help='Flush samples to CSV every N seconds (default: 10).',
    )
    parser.add_argument(
        '--duration', type=float, default=None,
        help='Optional total duration in seconds. Runs forever if omitted.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_loop(
        output_path=args.output,
        hw_type=args.hw,
        interval=args.interval,
        flush_every=args.flush_every,
        duration=args.duration,
    )