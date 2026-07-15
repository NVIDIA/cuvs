/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use cmake_package::{Error as CmakeError, Version, VersionError, find_package};

const CUVS_COMPONENT: &str = "c_api";
const CUVS_C_API_TARGET: &str = "cuvs::c_api";
const CUVS_CMAKE_INSPECTION_FAILED: &str = "CMake failed while inspecting cuVS. Check the build environment for missing tools such as ninja/make, C/C++ compilers, or CUDA dependencies.";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_PRINT_LIBCUVS_PACKAGE_DIR: &str = r#"
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("libcuvs")
if spec is None or spec.submodule_search_locations is None:
    raise ModuleNotFoundError("libcuvs")

print(Path(next(iter(spec.submodule_search_locations))).resolve())
"#;

pub(crate) struct CuvsMetadata {
    pub(crate) include_dir: PathBuf,
    #[cfg(feature = "generate-bindings")]
    pub(crate) bindgen_include_dirs: Vec<PathBuf>,
    pub(crate) lib_dir: PathBuf,
}

fn cmake_unavailable_error() -> anyhow::Error {
    anyhow::anyhow!(
        "CMake is not installed or does not satisfy this build's requirements. Install the required CMake version and try again."
    )
}

fn cuvs_package_not_found_error() -> anyhow::Error {
    anyhow::anyhow!(
        "Could not find a cuVS CMake package compatible with cuvs-sys {PACKAGE_VERSION}.\n\n\
         Install cuVS via one of:\n\
         - conda: conda install -c rapidsai libcuvs\n\
         - pip:   pip install libcuvs-cu<CUDA_VERSION> and set LIBCUVS_USE_PYTHON=1\n\
         Or set CMAKE_PREFIX_PATH to point to your cuVS build/install directory."
    )
}

fn cuvs_incompatible_version_error(
    required_version: &Version,
    candidates: &[cmake_package::Considered],
) -> anyhow::Error {
    let considered = candidates
        .iter()
        .map(|candidate| format!("- {} (version: {})", candidate.config, candidate.version))
        .collect::<Vec<_>>()
        .join("\n");

    anyhow::anyhow!(
        "Found cuVS CMake package candidates, but none are compatible with cuvs-sys {PACKAGE_VERSION}.\n\n\
         Required compatibility: same major/minor as {required_version} and not older than {required_version}.\n\n\
         Considered candidates:\n{considered}"
    )
}

fn cmake_package_error(error: CmakeError, required_version: &Version) -> anyhow::Error {
    match error {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        CmakeError::PackageNotFound => cuvs_package_not_found_error(),
        CmakeError::Version(VersionError::VersionIncompatible(candidates)) => {
            cuvs_incompatible_version_error(required_version, &candidates)
        }
        CmakeError::Version(VersionError::InvalidVersion) => anyhow::anyhow!(
            "{CUVS_CMAKE_INSPECTION_FAILED}\n\nCMake reported an invalid cuVS package version."
        ),
        CmakeError::IO(error) => {
            anyhow::anyhow!("{CUVS_CMAKE_INSPECTION_FAILED}\n\nUnderlying error: {error}")
        }
        CmakeError::Internal => anyhow::anyhow!("{CUVS_CMAKE_INSPECTION_FAILED}"),
    }
}

fn find_target(
    package: &cmake_package::CMakePackage,
    target_name: &str,
) -> Result<cmake_package::CMakeTarget> {
    package.target(target_name).with_context(|| {
        format!("Found CMake package {}, but target {target_name} was not exported.", package.name)
    })
}

fn find_cuvs_package(
    required_version: &Version,
    python_package_dir: Option<&Path>,
) -> Result<cmake_package::CMakePackage> {
    let mut package = find_package("cuvs")
        .version(*required_version)
        .components([CUVS_COMPONENT.to_owned()])
        .define("CMAKE_FIND_PACKAGE_PREFER_CONFIG", "TRUE");

    if let Some(python_package_dir) = python_package_dir {
        package = package.prefix_paths(vec![python_package_dir.to_path_buf()]);
    }

    package.find().map_err(|error| cmake_package_error(error, required_version))
}

#[cfg(feature = "generate-bindings")]
fn find_dlpack_package() -> Result<cmake_package::CMakePackage> {
    find_package("dlpack").find().map_err(|e| match e {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        _ => anyhow::anyhow!(
            "Could not find DLPack CMake package.\n\n\
             Install DLPack so that `find_package(dlpack)` succeeds."
        ),
    })
}

/// Run CMake `find_package(cuvs <version>)` and extract the include and library directories.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
pub(crate) fn try_find_cuvs_package(
    required_version: &Version,
    python_package_dir: Option<&Path>,
) -> Result<CuvsMetadata> {
    let package = find_cuvs_package(required_version, python_package_dir)?;
    let target = find_target(&package, CUVS_C_API_TARGET)?;

    let include_dir = target
        .include_directories
        .first()
        .map(PathBuf::from)
        .context("cuVS CMake target did not export any include directories")?;

    // DLPack include directories are only needed for bindgen.
    #[cfg(feature = "generate-bindings")]
    let bindgen_include_dirs: Vec<_> = {
        let dlpack = find_dlpack_package()?;
        let dlpack_target = find_target(&dlpack, "dlpack::dlpack")?;
        dlpack_target
            .include_directories
            .iter()
            .map(PathBuf::from)
            .filter(|dir| dir.is_dir())
            .filter(|dir| dir != &include_dir)
            .collect()
    };

    let lib_dir = target
        .location
        .as_deref()
        .and_then(|location| Path::new(location).parent())
        .map(Path::to_path_buf)
        .or_else(|| target.link_directories.first().map(PathBuf::from))
        .context("cuVS CMake target did not export a library location or link directory")?;

    target.link();

    Ok(CuvsMetadata {
        include_dir,
        #[cfg(feature = "generate-bindings")]
        bindgen_include_dirs,
        lib_dir,
    })
}

fn find_python_cuvs_package_dir() -> Result<PathBuf> {
    let python =
        Path::new(if std::env::var_os("VIRTUAL_ENV").is_some() { "python" } else { "python3" });
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_PRINT_LIBCUVS_PACKAGE_DIR)
        .output()
        .with_context(|| format!("LIBCUVS_USE_PYTHON is set, but failed to run {:?}.", python))?;

    anyhow::ensure!(
        output.status.success(),
        "LIBCUVS_USE_PYTHON is set, but {:?} could not locate the Python libcuvs package.\n\n\
             Install the libcuvs wheel in that Python environment, or unset LIBCUVS_USE_PYTHON.\n\n\
             {}",
        python,
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let package_dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let cmake_dir = package_dir.join("lib64/cmake/cuvs");
    anyhow::ensure!(
        cmake_dir.is_dir(),
        "LIBCUVS_USE_PYTHON is set, but the Python libcuvs package at {} does not contain a cuVS CMake package under {}.",
        package_dir.display(),
        cmake_dir.display(),
    );

    Ok(package_dir)
}

/// Locate cuVS either from standard CMake search paths or, when explicitly
/// requested, from the active Python environment.
pub(crate) fn locate_cuvs() -> Result<CuvsMetadata> {
    let required_version: Version = PACKAGE_VERSION
        .try_into()
        .expect("workspace package version must be a valid semantic version");

    let python_package_dir = std::env::var_os("LIBCUVS_USE_PYTHON")
        .map(|_| find_python_cuvs_package_dir())
        .transpose()?;

    try_find_cuvs_package(&required_version, python_package_dir.as_deref())
}
