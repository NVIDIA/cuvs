/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/TileAlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>
#include <cuvs/detail/jit_lto/cutile_module.hpp>
#include <cuvs/detail/jit_lto/cutile_smoke_fragments.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>
#include <memory>
#include <vector>

namespace cuvs::detail::jit_lto {

#if !CUVS_CUTILE_ENABLED

TEST(CutileSmoke, DisabledBuild)
{
  GTEST_SKIP() << "cuTile embedded kernels are disabled in this build";
}

#else

namespace {

template <typename ArchTag>
using smoke_fragment = StaticCubinFragmentEntry<fragment_tag_cutile_smoke_add_cubin<ArchTag>>;

std::vector<std::unique_ptr<CubinFragmentEntry>> make_smoke_fragments()
{
  std::vector<std::unique_ptr<CubinFragmentEntry>> fragments;
  fragments.emplace_back(std::make_unique<smoke_fragment<cutile_arch_8_0>>());
  fragments.emplace_back(std::make_unique<smoke_fragment<cutile_arch_8_6>>());
  fragments.emplace_back(std::make_unique<smoke_fragment<cutile_arch_9_0>>());
  fragments.emplace_back(std::make_unique<smoke_fragment<cutile_arch_10_0>>());
  fragments.emplace_back(std::make_unique<smoke_fragment<cutile_arch_12_0>>());
  return fragments;
}

void add_smoke_fragments(TileAlgorithmPlanner& planner)
{
  planner.add_static_fragment<fragment_tag_cutile_smoke_add_cubin<cutile_arch_8_0>>();
  planner.add_static_fragment<fragment_tag_cutile_smoke_add_cubin<cutile_arch_8_6>>();
  planner.add_static_fragment<fragment_tag_cutile_smoke_add_cubin<cutile_arch_9_0>>();
  planner.add_static_fragment<fragment_tag_cutile_smoke_add_cubin<cutile_arch_10_0>>();
  planner.add_static_fragment<fragment_tag_cutile_smoke_add_cubin<cutile_arch_12_0>>();
}

}  // namespace

TEST(CutileSmoke, ResolvesEveryEmbeddedArchitecture)
{
  auto fragments = make_smoke_fragments();

  EXPECT_EQ(find_compatible_cubin_fragment(8, 0, fragments), fragments[0].get());
  EXPECT_EQ(find_compatible_cubin_fragment(8, 9, fragments), fragments[1].get());
  EXPECT_EQ(find_compatible_cubin_fragment(9, 0, fragments), fragments[2].get());
  EXPECT_EQ(find_compatible_cubin_fragment(10, 0, fragments), fragments[3].get());
  EXPECT_EQ(find_compatible_cubin_fragment(12, 1, fragments), fragments[4].get());
  EXPECT_EQ(find_compatible_cubin_fragment(7, 5, fragments), nullptr);
}

TEST(CutileSmoke, LaunchesCompatibleCubin)
{
  CutileRuntimeCapabilities capabilities{};
  if (!query_current_cutile_runtime_capabilities(capabilities)) {
    GTEST_SKIP() << "No CUDA device is available";
  }

  auto fragments = make_smoke_fragments();
  if (find_compatible_cubin_fragment(capabilities.cc_major, capabilities.cc_minor, fragments) ==
      nullptr) {
    GTEST_SKIP() << "No embedded smoke cubin is compatible with this device";
  }

  TileLauncherCache cache;
  TileAlgorithmPlanner planner{"cutile_smoke_add", cache};
  add_smoke_fragments(planner);
  auto launcher = planner.try_get_launcher();
  ASSERT_NE(launcher, nullptr);

  cudaStream_t stream = nullptr;
  constexpr int count = 256;
  std::array<float, count> host_lhs{};
  std::array<float, count> host_rhs{};
  std::array<float, count> host_output{};
  for (int i = 0; i < count; ++i) {
    host_lhs[i] = static_cast<float>(i);
    host_rhs[i] = static_cast<float>(count - i);
  }

  float* lhs    = nullptr;
  float* rhs    = nullptr;
  float* output = nullptr;
  ASSERT_EQ(cudaMalloc(&lhs, sizeof(host_lhs)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&rhs, sizeof(host_rhs)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output, sizeof(host_output)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(lhs, host_lhs.data(), sizeof(host_lhs), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(rhs, host_rhs.data(), sizeof(host_rhs), cudaMemcpyHostToDevice),
            cudaSuccess);

  using smoke_kernel_t = void(void*, int, int, void*, int, int, void*, int, int);
  launcher->template dispatch<smoke_kernel_t>(stream,
                                              dim3{1, 1, 1},
                                              dim3{1, 1, 1},
                                              0,
                                              static_cast<void*>(lhs),
                                              count,
                                              1,
                                              static_cast<void*>(rhs),
                                              count,
                                              1,
                                              static_cast<void*>(output),
                                              count,
                                              1);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(host_output.data(), output, sizeof(host_output), cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaFree(lhs), cudaSuccess);
  ASSERT_EQ(cudaFree(rhs), cudaSuccess);
  ASSERT_EQ(cudaFree(output), cudaSuccess);

  for (const auto value : host_output) {
    EXPECT_FLOAT_EQ(value, static_cast<float>(count));
  }
}

#endif

}  // namespace cuvs::detail::jit_lto
