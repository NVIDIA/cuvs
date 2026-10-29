/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FlowANN is described in "Disentangling Graph Dependencies for Efficient
 * Billion-Scale GPU Vector Search" (OSDI 2026).
 */

//! Options for experimental CAGRA graph storage across GPU and CPU memory.

use super::CagraError;
use crate::error::check_cuvs;

/// Representation of the native CAGRA graph.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphStorage {
    /// Store the adjacency graph on the device.
    #[default]
    Device,
    /// Store a compressed subset on the GPU and fetch other edges from the CPU.
    Tiered,
}

impl From<GraphStorage> for ffi::cuvsCagraGraphStorage {
    fn from(value: GraphStorage) -> Self {
        match value {
            GraphStorage::Device => Self::CUVS_CAGRA_GRAPH_DEVICE,
            GraphStorage::Tiered => Self::CUVS_CAGRA_GRAPH_TIERED,
        }
    }
}

impl From<ffi::cuvsCagraGraphStorage> for GraphStorage {
    fn from(value: ffi::cuvsCagraGraphStorage) -> Self {
        match value {
            ffi::cuvsCagraGraphStorage::CUVS_CAGRA_GRAPH_DEVICE => Self::Device,
            ffi::cuvsCagraGraphStorage::CUVS_CAGRA_GRAPH_TIERED => Self::Tiered,
        }
    }
}

/// Experimental graph layout options. Unchanged defaults match the C library.
#[derive(Debug, Copy, Clone)]
pub struct TieredGraphParams {
    /// Packed GPU edge budget in bytes; zero stores all edges on the host.
    pub device_graph_budget_bytes: usize,
    /// Nodes per packed row, from 1 to 64 with grouping or 1 to 32 without it.
    pub node_per_cacheline: u16,
    /// Group and reorder nodes to improve locality.
    pub grouping_enabled: bool,
    /// Final group count; zero chooses automatically, maximum 65,536.
    pub n_groups: u32,
    /// Local ID width; zero chooses automatically, otherwise from 1 to 32.
    pub n_bits: u16,
    /// Allowed group imbalance in the open interval (0, 1).
    pub balance_tolerance: f64,
    /// Maximum training rows per grouping node; zero chooses automatically.
    pub training_rows: usize,
    /// Assignment batch size in rows; zero chooses automatically.
    pub assignment_batch_rows: usize,
    /// Positive number of k-means training iterations.
    pub kmeans_n_iters: u32,
    /// Verify encoded edges after building the graph.
    pub validate: bool,
    /// Number of medoid entry points to generate; zero disables generation.
    pub num_seeds: u32,
    /// Seed training sample rows; zero chooses automatically.
    pub seed_training_rows: usize,
    /// Random seed for grouping and seed generation.
    pub seed: u64,
}

impl Default for TieredGraphParams {
    fn default() -> Self {
        Self {
            device_graph_budget_bytes: 0,
            node_per_cacheline: 2,
            grouping_enabled: true,
            n_groups: 0,
            n_bits: 0,
            balance_tolerance: 0.1,
            training_rows: 0,
            assignment_batch_rows: 0,
            kmeans_n_iters: 20,
            validate: false,
            num_seeds: 0,
            seed_training_rows: 0,
            seed: 0x9e3779b97f4a7c15,
        }
    }
}

impl TieredGraphParams {
    pub(super) fn validate(&self) -> Result<(), CagraError> {
        let max_nodes = if self.grouping_enabled { 64 } else { 32 };
        if self.node_per_cacheline == 0 || self.node_per_cacheline > max_nodes {
            return Err(CagraError::Validation(format!(
                "node_per_cacheline must be in [1, {max_nodes}]"
            )));
        }
        if self.n_groups > 65_536 || self.n_bits > 32 || self.kmeans_n_iters == 0 {
            return Err(CagraError::Validation(
                "invalid tiered grouping count, ID width, or iteration count".into(),
            ));
        }
        if !self.balance_tolerance.is_finite()
            || self.balance_tolerance <= 0.0
            || self.balance_tolerance >= 1.0
        {
            return Err(CagraError::Validation("balance_tolerance must be in (0, 1)".into()));
        }
        Ok(())
    }
}

impl From<TieredGraphParams> for ffi::cuvsCagraTieredGraphParams {
    fn from(p: TieredGraphParams) -> Self {
        Self {
            device_graph_budget_bytes: p.device_graph_budget_bytes,
            node_per_cacheline: p.node_per_cacheline,
            grouping_enabled: p.grouping_enabled,
            n_groups: p.n_groups,
            n_bits: p.n_bits,
            balance_tolerance: p.balance_tolerance,
            training_rows: p.training_rows,
            assignment_batch_rows: p.assignment_batch_rows,
            kmeans_n_iters: p.kmeans_n_iters,
            validate: p.validate,
            num_seeds: p.num_seeds,
            seed_training_rows: p.seed_training_rows,
            seed: p.seed,
        }
    }
}

/// Experimental search queue options. Searches complete before returning.
#[derive(Debug, Copy, Clone)]
pub struct TieredSearchParams {
    /// Maximum stored medoid entry points used per query.
    pub num_seeds: u32,
    /// Adaptive synchronization window scale.
    pub sync_window_scale: f32,
    /// Parent position drop that forces synchronization.
    pub sync_drop_threshold: u32,
    /// Positive number of CPU/GPU command queues.
    pub num_queues: u32,
    /// CPU pause instructions after each empty poll.
    pub empty_pause: u32,
    /// Collect native queue statistics.
    pub collect_statistics: bool,
    /// Keep CPU workers active between searches, consuming CPU while idle.
    pub keep_pollers_running: bool,
}

impl Default for TieredSearchParams {
    fn default() -> Self {
        Self {
            num_seeds: 0,
            sync_window_scale: 10.0,
            sync_drop_threshold: 51,
            num_queues: 1,
            empty_pause: 64,
            collect_statistics: false,
            keep_pollers_running: false,
        }
    }
}

impl TieredSearchParams {
    pub(super) fn validate(&self) -> Result<(), CagraError> {
        if self.num_queues == 0
            || !self.sync_window_scale.is_finite()
            || self.sync_window_scale <= 0.0
        {
            return Err(CagraError::Validation(
                "tiered search requires positive num_queues and sync_window_scale".into(),
            ));
        }
        Ok(())
    }
}

impl From<TieredSearchParams> for ffi::cuvsCagraTieredSearchParams {
    fn from(p: TieredSearchParams) -> Self {
        Self {
            num_seeds: p.num_seeds,
            sync_window_scale: p.sync_window_scale,
            sync_drop_threshold: p.sync_drop_threshold,
            num_queues: p.num_queues,
            empty_pause: p.empty_pause,
            collect_statistics: p.collect_statistics,
            keep_pollers_running: p.keep_pollers_running,
        }
    }
}

/// Owns the native compression parameters used by [`super::Index::build_compressed`].
#[derive(Debug)]
pub struct CompressionParams {
    handle: ffi::cuvsCagraCompressionParams_t,
}

#[bon::bon]
impl CompressionParams {
    /// Allocate native defaults and optionally set VPQ dimensions and training settings.
    #[builder]
    pub fn new(
        pq_bits: Option<u32>,
        pq_dim: Option<u32>,
        vq_n_centers: Option<u32>,
        kmeans_n_iters: Option<u32>,
        vq_kmeans_trainset_fraction: Option<f64>,
        pq_kmeans_trainset_fraction: Option<f64>,
    ) -> Result<Self, CagraError> {
        if pq_bits.is_some_and(|bits| bits != 8) {
            return Err(CagraError::Validation("tiered VPQ requires 8-bit codes".into()));
        }
        if kmeans_n_iters == Some(0) {
            return Err(CagraError::Validation("kmeans_n_iters must be positive".into()));
        }
        for fraction in
            [vq_kmeans_trainset_fraction, pq_kmeans_trainset_fraction].into_iter().flatten()
        {
            if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
                return Err(CagraError::Validation("training fractions must be in (0, 1]".into()));
            }
        }
        let mut handle = std::ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraCompressionParamsCreate(&mut handle) })?;
        let result = Self { handle };
        unsafe {
            if let Some(v) = pq_bits {
                (*handle).pq_bits = v;
            }
            if let Some(v) = pq_dim {
                (*handle).pq_dim = v;
            }
            if let Some(v) = vq_n_centers {
                (*handle).vq_n_centers = v;
            }
            if let Some(v) = kmeans_n_iters {
                (*handle).kmeans_n_iters = v;
            }
            if let Some(v) = vq_kmeans_trainset_fraction {
                (*handle).vq_kmeans_trainset_fraction = v;
            }
            if let Some(v) = pq_kmeans_trainset_fraction {
                (*handle).pq_kmeans_trainset_fraction = v;
            }
        }
        Ok(result)
    }

    pub(super) fn handle(&self) -> ffi::cuvsCagraCompressionParams_t {
        self.handle
    }
}

impl Drop for CompressionParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsCagraCompressionParamsDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_graph_options_are_rejected() {
        for p in [
            TieredGraphParams { node_per_cacheline: 0, ..Default::default() },
            TieredGraphParams {
                grouping_enabled: false,
                node_per_cacheline: 33,
                ..Default::default()
            },
            TieredGraphParams { n_groups: 65_537, ..Default::default() },
            TieredGraphParams { n_bits: 33, ..Default::default() },
            TieredGraphParams { balance_tolerance: f64::NAN, ..Default::default() },
            TieredGraphParams { kmeans_n_iters: 0, ..Default::default() },
        ] {
            assert!(matches!(p.validate(), Err(CagraError::Validation(_))));
        }
        assert!(TieredGraphParams::default().validate().is_ok());
    }

    #[test]
    fn invalid_queue_options_are_rejected() {
        assert!(TieredSearchParams { num_queues: 0, ..Default::default() }.validate().is_err());
        assert!(
            TieredSearchParams { sync_window_scale: f32::NAN, ..Default::default() }
                .validate()
                .is_err()
        );
        assert!(TieredSearchParams::default().validate().is_ok());
    }
}
