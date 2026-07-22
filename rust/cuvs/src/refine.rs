/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Refinement of approximate nearest neighbor results

use crate::distance_type::DistanceType;
use crate::dlpack::{AsDlTensor, AsDlTensorMut};
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;

/// Refine nearest neighbor search results.
///
/// Refinement is an operation that follows an approximate nearest neighbors
/// search. The approximate search has already selected `n_candidates` neighbor
/// candidates for each query. This narrows the candidate list down to the `k`
/// nearest neighbors by computing the exact distance between each query and its
/// candidates against the original dataset, then selecting the `k` closest.
///
/// All tensors must reside in the same memory space: either all on the device
/// or all on the host. The dataset and queries may be `f32`, `f16`, `i8`, or
/// `u8` (with matching dtype codes). The candidate and output index tensors
/// must be `i64`, and the output distance tensor must be `f32`.
///
/// Tensors are borrowed through the [`AsDlTensor`] /
/// [`AsDlTensorMut`] traits: `dataset`, `queries`, and `candidates` are read,
/// while `indices` and `distances` are written in place. See the
/// [`dlpack`](crate::dlpack) module for the tensor model and `examples/cagra.rs`
/// for a device-tensor adapter.
///
/// # Arguments
///
/// * `res` - Resources to use
/// * `dataset` - A row-major matrix of the original dataset - shape `(n_rows, dims)`
/// * `queries` - A row-major matrix of the queries - shape `(n_queries, dims)`
/// * `candidates` - A row-major `i64` matrix of candidate indices into `dataset`
///   - shape `(n_queries, n_candidates)`, where `n_candidates >= k`
/// * `metric` - DistanceType used to rank candidates
/// * `indices` - Output `i64` matrix that receives the refined indices - shape
///   `(n_queries, k)`. `k` is inferred from this tensor's shape.
/// * `distances` - Output `f32` matrix that receives the refined distances -
///   shape `(n_queries, k)`
pub fn refine<DS, Q, C, I, D>(
    res: &Resources,
    dataset: &DS,
    queries: &Q,
    candidates: &C,
    metric: DistanceType,
    indices: &mut I,
    distances: &mut D,
) -> Result<()>
where
    DS: AsDlTensor + ?Sized,
    Q: AsDlTensor + ?Sized,
    C: AsDlTensor + ?Sized,
    I: AsDlTensorMut + ?Sized,
    D: AsDlTensorMut + ?Sized,
{
    let dataset = dataset.as_dl_tensor()?;
    let queries = queries.as_dl_tensor()?;
    let candidates = candidates.as_dl_tensor()?;
    let indices = indices.as_dl_tensor_mut()?;
    let distances = distances.as_dl_tensor_mut()?;
    unsafe {
        check_cuvs(ffi::cuvsRefine(
            res.0,
            dataset.to_c().as_mut_ptr(),
            queries.to_c().as_mut_ptr(),
            candidates.to_c().as_mut_ptr(),
            metric,
            indices.to_c().as_mut_ptr(),
            distances.to_c().as_mut_ptr(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::DeviceTensor;

    /// Refinement must repair a candidate list that contains deliberately
    /// wrong entries: after refine, the top-k must equal the exact
    /// brute-force top-k.
    #[test]
    fn test_refine_fixes_wrong_candidates() {
        let res = Resources::new().unwrap();

        // A small, well-separated 2-D dataset. The exact L2 ranking of every
        // query is unambiguous, so we can hard-assert the refined output.
        //
        //   index : point
        //     0   : (0, 0)
        //     1   : (1, 0)
        //     2   : (0, 1)
        //     3   : (2, 2)
        //     4   : (5, 5)
        //     5   : (9, 9)
        let dataset = ndarray::array![
            [0.0f32, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
            [9.0, 9.0],
        ];

        // Two queries near distinct clusters.
        //   q0 sits next to point 0; true top-3 = [0, 1, 2]
        //   q1 sits next to point 4; true top-3 = [4, 3, 5] (4 closest, then 3, then 5)
        let queries = ndarray::array![[0.1f32, 0.1], [4.9, 4.9]];

        // Candidate lists are intentionally *wrong order* and include far-away
        // points. Each list is a superset of the true top-3 but jumbled, plus a
        // planted bad candidate (index 5 for q0, index 0 for q1). Refine must
        // re-rank these exactly and select the correct nearest three.
        let candidates = ndarray::array![
            [5i64, 2, 0, 1], // q0: true nearest 0 is buried, 5 is far noise
            [0i64, 5, 3, 4], // q1: true nearest 4 is last, 0 is far noise
        ];

        let n_queries = 2;
        let k = 3;

        let dataset_d = DeviceTensor::from_host(&res, &dataset).unwrap();
        let queries_d = DeviceTensor::from_host(&res, &queries).unwrap();
        let candidates_d = DeviceTensor::from_host(&res, &candidates).unwrap();

        let mut indices_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut indices_d = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances_d = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        refine(
            &res,
            &dataset_d,
            &queries_d,
            &candidates_d,
            DistanceType::L2Expanded,
            &mut indices_d,
            &mut distances_d,
        )
        .unwrap();

        indices_d.copy_to_host(&res, &mut indices_host).unwrap();
        distances_d.copy_to_host(&res, &mut distances_host).unwrap();
        res.sync_stream().unwrap();

        // Exact brute-force top-3, independent of the candidate ordering.
        // q0: distances to (0.1,0.1): 0 -> ~0.14, 1 -> ~0.91, 2 -> ~0.91, ...
        //     point 0 is strictly nearest; 1 and 2 are tied next.
        // q1: distances to (4.9,4.9): 4 -> ~0.14, 3 -> ~4.1, 5 -> ~5.8.
        assert_eq!(
            indices_host[[0, 0]],
            0,
            "q0 nearest must be repaired to index 0, got {:?}",
            indices_host.row(0)
        );
        assert_eq!(
            indices_host[[1, 0]],
            4,
            "q1 nearest must be repaired to index 4, got {:?}",
            indices_host.row(1)
        );

        // The planted noise candidates (5 for q0, 0 for q1) must be evicted
        // from the refined top-k.
        let q0: Vec<i64> = indices_host.row(0).to_vec();
        let q1: Vec<i64> = indices_host.row(1).to_vec();
        assert!(!q0.contains(&5), "q0 must drop far candidate 5, got {:?}", q0);
        assert!(!q1.contains(&0), "q1 must drop far candidate 0, got {:?}", q1);

        // The refined top-3 sets must match the exact brute-force top-3 sets.
        let mut q0_sorted = q0.clone();
        q0_sorted.sort_unstable();
        assert_eq!(q0_sorted, vec![0, 1, 2], "q0 refined set wrong: {:?}", q0);

        let mut q1_sorted = q1.clone();
        q1_sorted.sort_unstable();
        assert_eq!(q1_sorted, vec![3, 4, 5], "q1 refined set wrong: {:?}", q1);

        // Refined distances must be sorted ascending (nearest first) across
        // the full top-k, and the first entry must be the small in-cluster
        // distance, not noise.
        for q in 0..2 {
            for i in 0..2 {
                assert!(
                    distances_host[[q, i]] <= distances_host[[q, i + 1]],
                    "q{q} distances not ascending at {i}: {:?}",
                    (distances_host[[q, i]], distances_host[[q, i + 1]])
                );
            }
        }
        assert!(
            distances_host[[0, 0]] < 1.0,
            "q0 nearest distance should be small, got {}",
            distances_host[[0, 0]]
        );
    }
}
