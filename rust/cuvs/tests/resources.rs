/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::time::Duration;

use cuvs::resources::Resources;

#[test]
fn memory_tracking_writes_csv_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("alloc.csv");
    {
        let _resources = Resources::with_memory_tracking(&csv, Some(Duration::from_millis(2)))
            .expect("with_memory_tracking should succeed");
    }
    let meta = std::fs::metadata(&csv).expect("CSV file should exist after drop");
    assert!(meta.len() > 0, "tracking CSV should be non-empty");
}
