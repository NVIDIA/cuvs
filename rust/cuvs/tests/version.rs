/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuvs::version::version;

#[test]
fn linked_library_version_is_compatible_with_crate() {
    let (library_major, library_minor, library_patch) =
        version().expect("the linked cuVS library should report its version");
    let (crate_major, crate_minor, crate_patch) = (
        env!("CARGO_PKG_VERSION_MAJOR").parse::<u16>().unwrap(),
        env!("CARGO_PKG_VERSION_MINOR").parse::<u16>().unwrap(),
        env!("CARGO_PKG_VERSION_PATCH").parse::<u16>().unwrap(),
    );

    // This mirrors the compatibility policy used by cuvs-sys when locating the
    // native CMake package: major and minor must match so the expected API and
    // ABI are present.
    assert_eq!(
        (library_major, library_minor),
        (crate_major, crate_minor),
        "the linked cuVS library must have the same major and minor version as the crate"
    );

    // cuvs-sys permits a newer patch release because they are backward-compatible.
    assert!(
        library_patch >= crate_patch,
        "the linked cuVS library patch version must not be older than the crate"
    );
}
