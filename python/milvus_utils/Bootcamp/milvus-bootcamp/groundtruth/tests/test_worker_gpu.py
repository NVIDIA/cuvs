import numpy as np
import queue
import torch

from groundtruth.main import worker_thread


def test_worker_l2(skip_if_no_gpu):
    queries = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    base = np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32)

    q = queue.Queue()
    q.put((0, base))
    q.put(None)

    results = [None]

    worker_thread(
        gpu_id=0,
        query_indices=np.array([0, 1]),
        queries_np=queries,
        work_queue=q,
        k=1,
        metric="l2",
        results_container=results,
    )

    vals, idxs = results[0]
    assert idxs.shape == (2, 1)
    assert idxs[0, 0] == 0  # closest to [0,0]
    assert idxs[1, 0] == 0  # closest to [0,0]


def test_worker_ip(skip_if_no_gpu):
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    base = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    q = queue.Queue()
    q.put((0, base))
    q.put(None)

    results = [None]

    worker_thread(
        gpu_id=0,
        query_indices=np.array([0]),
        queries_np=queries,
        work_queue=q,
        k=1,
        metric="ip",
        results_container=results,
    )

    _, idxs = results[0]
    assert idxs[0, 0] == 0
