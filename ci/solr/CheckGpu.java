/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuvs.CuVSResources;

/** Fail setup when the nightly Java bindings cannot initialize the native GPU library. */
class CheckGpu {
  public static void main(String[] args) throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      System.out.println("Nightly cuVS initialized CUDA device " + resources.deviceId());
    }
  }
}
