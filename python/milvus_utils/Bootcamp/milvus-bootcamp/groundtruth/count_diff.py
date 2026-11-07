import numpy as np
import sys
import os

def read_diskann_gt(path):
    with open(path, 'rb') as f:
        nq = np.frombuffer(f.read(4), dtype=np.int32)[0]
        k = np.frombuffer(f.read(4), dtype=np.int32)[0]

        idx = np.frombuffer(f.read(nq * k * 4), dtype=np.int32).reshape(nq, k)
        dists = np.frombuffer(f.read(nq * k * 4), dtype=np.float32).reshape(nq, k)

    return nq, k, idx, dists


# ------------------------------
# Argument parsing
# ------------------------------
if len(sys.argv) != 3:
    print("Usage: python count_diffs.py <cpu_gt_file> <gpu_gt_file>")
    sys.exit(1)

cpu_file = sys.argv[1]
gpu_file = sys.argv[2]

if not os.path.exists(cpu_file):
    print(f"Error: CPU groundtruth file not found: {cpu_file}")
    sys.exit(1)

if not os.path.exists(gpu_file):
    print(f"Error: GPU groundtruth file not found: {gpu_file}")
    sys.exit(1)


# ------------------------------
# Load files
# ------------------------------
print(f"Loading CPU GT: {cpu_file}")
nq, k, ref_idx, ref_dist = read_diskann_gt(cpu_file)

print(f"Loading GPU GT: {gpu_file}")
_, _, gpu_idx, gpu_dist = read_diskann_gt(gpu_file)


# ------------------------------
# Count diffs
# ------------------------------
diff_queries = np.where(~np.all(ref_idx == gpu_idx, axis=1))[0]
print(f"Total queries with differing indices: {len(diff_queries)}")

# Classification counters
tie_like = 0
float_drift = 0
real_error = 0
dist_mismatch = 0

# Detailed print limit
MAX_SHOW = 10
shown = 0

for qi in diff_queries:
    ref_i = ref_idx[qi]
    gpu_i = gpu_idx[qi]

    ref_d = ref_dist[qi]
    gpu_d = gpu_dist[qi]

    mism_pos = np.where(ref_i != gpu_i)[0]

    for pos in mism_pos:
        dr = ref_d[pos]
        dg = gpu_d[pos]
        absdiff = abs(dr - dg)

        # --- classify ---
        if ref_i[pos] != gpu_i[pos] and absdiff == 0:
            tie_like += 1

        elif absdiff < 1e-6:
            float_drift += 1

        elif absdiff > 1e-3:
            real_error += 1

        else:
            dist_mismatch += 1

        # --- print a few details ---
        if shown < MAX_SHOW:
            print("\n==============================")
            print(f"Query {qi}, pos {pos}")
            print(f"CPU idx={ref_i[pos]}, dist={dr}")
            print(f"GPU idx={gpu_i[pos]}, dist={dg}")
            print(f"Abs dist diff = {absdiff:e}")
            shown += 1


# ------------------------------
# Summary
# ------------------------------
print("\n=== DIAGNOSTIC SUMMARY ===")
print(f"Tie-like mismatches (distances identical) : {tie_like}")
print(f"Floating-point drift (<1e-6)             : {float_drift}")
print(f"Large distance mismatch (>1e-3)          : {real_error}")
print(f"Medium mismatches (other)                : {dist_mismatch}")

