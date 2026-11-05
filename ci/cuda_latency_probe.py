import ctypes
import glob
import os
import statistics
import subprocess
import sys
import time


def find_cudart():
    roots = [os.environ.get("CONDA_PREFIX", "")] + sys.path
    for root in roots:
        for pat in ("lib/libcudart.so.1*", "nvidia/*/lib/libcudart.so.1*", "nvidia/cuda_runtime/lib/libcudart.so.1*"):
            hits = sorted(glob.glob(os.path.join(root, pat)))
            if hits:
                return hits[-1]
    raise RuntimeError("libcudart not found")


def run(flags):
    rt = ctypes.CDLL(find_cudart())
    if flags is not None:
        assert rt.cudaSetDeviceFlags(flags) == 0
    assert rt.cudaFree(None) == 0
    s = ctypes.c_void_p()
    assert rt.cudaStreamCreate(ctypes.byref(s)) == 0
    d = ctypes.c_void_p()
    assert rt.cudaMalloc(ctypes.byref(d), 1 << 20) == 0
    h = (ctypes.c_char * 4)()
    ev = ctypes.c_void_p()
    rt.cudaEventCreate(ctypes.byref(ev))

    def t(name, fn, n=2000):
        for _ in range(50):
            fn()
        xs = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            fn()
            xs.append(time.perf_counter_ns() - t0)
        xs.sort()
        print(f"  {name:38s} median {xs[n // 2] / 1e3:9.1f} us  p90 {xs[n * 9 // 10] / 1e3:9.1f} us")

    t("empty stream sync", lambda: rt.cudaStreamSynchronize(s))
    t("memsetAsync 4B + sync", lambda: (rt.cudaMemsetAsync(d, 0, 4, s), rt.cudaStreamSynchronize(s)))
    t("memcpyAsync H2D 4B pageable + sync", lambda: (rt.cudaMemcpyAsync(d, h, 4, 1, s), rt.cudaStreamSynchronize(s)))
    t("memcpyAsync D2H 4B pageable + sync", lambda: (rt.cudaMemcpyAsync(h, d, 4, 2, s), rt.cudaStreamSynchronize(s)))
    t("memcpyAsync H2D 4B pageable (no sync)", lambda: rt.cudaMemcpyAsync(d, h, 4, 1, s))
    rt.cudaStreamSynchronize(s)
    t("memsetAsync 4B (no sync)", lambda: rt.cudaMemsetAsync(d, 0, 4, s))
    rt.cudaStreamSynchronize(s)
    t("eventRecord + eventSynchronize", lambda: (rt.cudaEventRecord(ev, s), rt.cudaEventSynchronize(ev)))
    p = ctypes.c_void_p()
    t("mallocAsync+freeAsync 256B (no sync)", lambda: (rt.cudaMallocAsync(ctypes.byref(p), 256, s), rt.cudaFreeAsync(p, s)))
    rt.cudaStreamSynchronize(s)
    t("memcpy 1 MiB D2H pageable", lambda: rt.cudaMemcpy(ctypes.create_string_buffer(1 << 20), d, 1 << 20, 2), n=200)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(None if sys.argv[1] == "default" else int(sys.argv[1]))
    else:
        print(f"CUDA latency probe ({find_cudart()})")
        for name, flags in (("default", "default"), ("spin", "1"), ("yield", "2"), ("blocking", "4")):
            print(f"device flags: {name}", flush=True)
            subprocess.run([sys.executable, __file__, flags], check=False)
