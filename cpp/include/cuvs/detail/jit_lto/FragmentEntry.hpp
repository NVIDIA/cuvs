/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

#include <nvJitLink.h>

#include "nvjitlink_checker.hpp"

struct FragmentEntry {
  virtual ~FragmentEntry() = default;

  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  virtual const char* get_key() const = 0;
};

struct FatbinFragmentEntry : FragmentEntry {
  virtual const uint8_t* get_data() const = 0;

  virtual size_t get_length() const = 0;

  bool add_to(nvJitLinkHandle& handle) const override final;
};

template <typename FragmentTag>
struct StaticFatbinFragmentEntry final : FatbinFragmentEntry {
  const uint8_t* get_data() const override { return StaticFatbinFragmentEntry<FragmentTag>::data; }

  size_t get_length() const override { return StaticFatbinFragmentEntry<FragmentTag>::length; }

  const char* get_key() const override
  {
    return typeid(StaticFatbinFragmentEntry<FragmentTag>).name();
  }

  static const uint8_t* const data;
  static const size_t length;
};

struct UDFFatbinFragment final : FatbinFragmentEntry {
  UDFFatbinFragment(std::string key, std::vector<uint8_t> bytes)
    : key_(std::move(key)), bytes_(std::move(bytes))
  {
  }

  const uint8_t* get_data() const override { return bytes_.data(); }

  size_t get_length() const override { return bytes_.size(); }

  const char* get_key() const override { return key_.c_str(); }

 private:
  std::string key_;
  std::vector<uint8_t> bytes_;
};

/** cuTile GEMM-style block geometry embedded in generated Static*FragmentEntry specializations. */
struct CutileTileConfig {
  int tile_m;
  int tile_n;
  int tile_k;
};

/** Embedded CUDA binary module (cubin), loaded directly via cudaLibraryLoadData. */
struct CubinFragmentEntry {
  virtual ~CubinFragmentEntry() = default;

  virtual const uint8_t* get_data() const = 0;

  virtual size_t get_length() const = 0;

  virtual const char* get_key() const = 0;

  virtual int get_cc_major() const = 0;

  virtual int get_cc_minor() const = 0;

  virtual int get_tile_m() const { return 0; }

  virtual int get_tile_n() const { return 0; }

  virtual int get_tile_k() const { return 0; }
};

template <typename FragmentTag>
struct StaticCubinFragmentEntry final : CubinFragmentEntry {
  const uint8_t* get_data() const override { return StaticCubinFragmentEntry<FragmentTag>::data; }

  size_t get_length() const override { return StaticCubinFragmentEntry<FragmentTag>::length; }

  const char* get_key() const override
  {
    return typeid(StaticCubinFragmentEntry<FragmentTag>).name();
  }

  int get_cc_major() const override { return FragmentTag::cc_major; }

  int get_cc_minor() const override { return FragmentTag::cc_minor; }

  int get_tile_m() const override { return tile_m; }

  int get_tile_n() const override { return tile_n; }

  int get_tile_k() const override { return tile_k; }

  static const int tile_m;
  static const int tile_n;
  static const int tile_k;

  static const uint8_t* const data;
  static const size_t length;
};

/** Embedded TileIR bytecode, JIT-compiled by the driver when no matching cubin exists. */
struct TileIrBytecodeFragmentEntry {
  virtual ~TileIrBytecodeFragmentEntry() = default;

  virtual const uint8_t* get_data() const = 0;

  virtual size_t get_length() const = 0;

  virtual const char* get_key() const = 0;

  virtual int get_tile_m() const { return 0; }

  virtual int get_tile_n() const { return 0; }

  virtual int get_tile_k() const { return 0; }
};

template <typename FragmentTag>
struct StaticTileIrBytecodeFragmentEntry final : TileIrBytecodeFragmentEntry {
  const uint8_t* get_data() const override
  {
    return StaticTileIrBytecodeFragmentEntry<FragmentTag>::data;
  }

  size_t get_length() const override
  {
    return StaticTileIrBytecodeFragmentEntry<FragmentTag>::length;
  }

  const char* get_key() const override
  {
    return typeid(StaticTileIrBytecodeFragmentEntry<FragmentTag>).name();
  }

  int get_tile_m() const override { return tile_m; }

  int get_tile_n() const override { return tile_n; }

  int get_tile_k() const override { return tile_k; }

  static const int tile_m;
  static const int tile_n;
  static const int tile_k;

  static const uint8_t* const data;
  static const size_t length;
};
