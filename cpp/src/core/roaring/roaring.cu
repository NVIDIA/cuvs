/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring.hpp>

#include <raft/core/bitset.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace cuvs::core {

// ============================================================================
// from_sorted_ids: build GPU Roaring from a sorted host ID array
// ============================================================================

gpu_roaring from_sorted_ids(raft::resources const& res,
                            const uint32_t* sorted_ids,
                            uint32_t n_ids,
                            uint32_t universe_size)
{
  auto stream = raft::resource::get_cuda_stream(res);
  gpu_roaring result(stream);
  result.universe_size = universe_size;

  if (n_ids == 0) return result;

  // Partition IDs into containers by high 16 bits
  struct Container {
    uint16_t key;
    std::vector<uint16_t> values;  // low 16 bits
  };
  std::vector<Container> containers;

  uint16_t cur_key = static_cast<uint16_t>(sorted_ids[0] >> 16);
  containers.push_back({cur_key, {}});

  for (uint32_t i = 0; i < n_ids; ++i) {
    uint16_t key = static_cast<uint16_t>(sorted_ids[i] >> 16);
    uint16_t val = static_cast<uint16_t>(sorted_ids[i] & 0xFFFF);
    if (key != cur_key) {
      cur_key = key;
      containers.push_back({key, {}});
    }
    containers.back().values.push_back(val);
  }

  uint32_t n          = static_cast<uint32_t>(containers.size());
  result.n_containers = n;

  // Build SoA on host
  std::vector<uint16_t> h_keys(n);
  std::vector<roaring_container_type> h_types(n);
  std::vector<uint32_t> h_offsets(n);
  std::vector<uint16_t> h_cards(n);

  std::vector<uint64_t> h_bitmap_pool;
  std::vector<uint16_t> h_array_pool;
  uint32_t n_bitmap = 0, n_array = 0;

  for (uint32_t i = 0; i < n; ++i) {
    h_keys[i]     = containers[i].key;
    uint32_t card = static_cast<uint32_t>(containers[i].values.size());
    h_cards[i]    = static_cast<uint16_t>(card > 65535 ? 0 : card);

    if (card > 4096) {
      // Bitmap container
      h_types[i]   = roaring_container_type::BITMAP;
      h_offsets[i] = static_cast<uint32_t>(h_bitmap_pool.size() * sizeof(uint64_t));
      h_bitmap_pool.resize(h_bitmap_pool.size() + 1024, 0);
      uint64_t* words = h_bitmap_pool.data() + h_bitmap_pool.size() - 1024;
      for (uint16_t v : containers[i].values) {
        words[v / 64] |= 1ULL << (v % 64);
      }
      ++n_bitmap;
    } else {
      // Array container
      h_types[i]   = roaring_container_type::ARRAY;
      h_offsets[i] = static_cast<uint32_t>(h_array_pool.size() * sizeof(uint16_t));
      h_array_pool.insert(
        h_array_pool.end(), containers[i].values.begin(), containers[i].values.end());
      ++n_array;
    }
  }

  result.n_bitmap_containers = n_bitmap;
  result.n_array_containers  = n_array;
  result.n_run_containers    = 0;

  // Upload to device
  result.keys.resize(n, stream);
  result.types.resize(n, stream);
  result.offsets.resize(n, stream);
  result.cardinalities.resize(n, stream);

  raft::update_device(result.keys.data(), h_keys.data(), n, stream);
  raft::update_device(result.types.data(),
                      reinterpret_cast<const roaring_container_type*>(h_types.data()),
                      n,
                      stream);
  raft::update_device(result.offsets.data(), h_offsets.data(), n, stream);
  raft::update_device(result.cardinalities.data(), h_cards.data(), n, stream);

  if (!h_bitmap_pool.empty()) {
    result.bitmap_data.resize(h_bitmap_pool.size(), stream);
    raft::update_device(
      result.bitmap_data.data(), h_bitmap_pool.data(), h_bitmap_pool.size(), stream);
  }
  if (!h_array_pool.empty()) {
    result.array_data.resize(h_array_pool.size(), stream);
    raft::update_device(result.array_data.data(), h_array_pool.data(), h_array_pool.size(), stream);
  }

  // Total cardinality
  result.total_cardinality = static_cast<uint64_t>(n_ids);

  // Host mirrors (count-free filtered search: indptr and dispatch decisions
  // come from these, never from count kernels)
  result.h_keys    = h_keys;
  result.h_types   = h_types;
  result.h_offsets = h_offsets;
  result.h_element_counts.resize(n);
  for (uint32_t i = 0; i < n; ++i)
    result.h_element_counts[i] = static_cast<uint32_t>(containers[i].values.size());

  // Build direct-map key index: key_index[key] = container_idx, 0xFFFF = absent
  if (n > 0) {
    result.max_key   = h_keys[n - 1];
    uint32_t ki_size = result.max_key + 1;
    std::vector<uint16_t> h_key_index(ki_size, 0xFFFF);
    for (uint32_t i = 0; i < n; ++i) {
      h_key_index[h_keys[i]] = static_cast<uint16_t>(i);
    }
    result.key_index.resize(ki_size, stream);
    raft::update_device(result.key_index.data(), h_key_index.data(), ki_size, stream);
  }

  raft::resource::sync_stream(res);
  return result;
}

// ============================================================================
// Decompress kernel
// ============================================================================

__global__ void decompress_kernel(const uint16_t* keys,
                                  const roaring_container_type* types,
                                  const uint32_t* offsets,
                                  const uint16_t* cardinalities,
                                  uint32_t n_containers,
                                  const uint64_t* bitmap_data,
                                  const uint16_t* array_data,
                                  const uint16_t* run_data,
                                  uint32_t* output,
                                  uint32_t output_size_words)
{
  uint32_t cid = blockIdx.x;
  if (cid >= n_containers) return;

  uint32_t key       = keys[cid];
  uint32_t base_word = key * 2048u;
  auto ctype         = types[cid];
  uint32_t offset    = offsets[cid];

  if (ctype == roaring_container_type::BITMAP) {
    const uint32_t* src =
      reinterpret_cast<const uint32_t*>(bitmap_data) + (offset / sizeof(uint32_t));
    for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
      uint32_t dst_idx = base_word + i;
      if (dst_idx < output_size_words) { output[dst_idx] = src[i]; }
    }
  } else if (ctype == roaring_container_type::ARRAY) {
    const uint16_t* arr = array_data + (offset / sizeof(uint16_t));
    uint16_t card       = cardinalities[cid];
    for (uint32_t i = threadIdx.x; i < card; i += blockDim.x) {
      uint16_t val      = arr[i];
      uint32_t abs_bit  = (static_cast<uint32_t>(key) << 16) | val;
      uint32_t word_idx = abs_bit / 32u;
      uint32_t bit_pos  = abs_bit % 32u;
      if (word_idx < output_size_words) { atomicOr(&output[word_idx], 1u << bit_pos); }
    }
  } else if (ctype == roaring_container_type::RUN) {
    const uint16_t* runs = run_data + (offset / sizeof(uint16_t));
    uint16_t n_runs      = cardinalities[cid];
    for (uint32_t r = threadIdx.x; r < n_runs; r += blockDim.x) {
      uint16_t start  = runs[r * 2];
      uint16_t length = runs[r * 2 + 1];
      for (uint32_t v = start; v <= static_cast<uint32_t>(start) + length; ++v) {
        uint32_t abs_bit  = (static_cast<uint32_t>(key) << 16) | v;
        uint32_t word_idx = abs_bit / 32u;
        uint32_t bit_pos  = abs_bit % 32u;
        if (word_idx < output_size_words) { atomicOr(&output[word_idx], 1u << bit_pos); }
      }
    }
  }
}

void decompress_to_bitset(raft::resources const& res,
                          const gpu_roaring& bitmap,
                          uint32_t* output,
                          uint32_t output_size_words)
{
  if (bitmap.n_containers == 0) return;
  auto stream = raft::resource::get_cuda_stream(res);

  RAFT_CUDA_TRY(cudaMemsetAsync(output, 0, output_size_words * sizeof(uint32_t), stream));

  decompress_kernel<<<bitmap.n_containers, 256, 0, stream>>>(bitmap.keys.data(),
                                                             bitmap.types.data(),
                                                             bitmap.offsets.data(),
                                                             bitmap.cardinalities.data(),
                                                             bitmap.n_containers,
                                                             bitmap.bitmap_data.data(),
                                                             bitmap.array_data.data(),
                                                             bitmap.run_data.data(),
                                                             output,
                                                             output_size_words);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

cuvs::core::bitset<uint32_t, int64_t> to_bitset(raft::resources const& res,
                                                const gpu_roaring& bitmap)
{
  auto stream    = raft::resource::get_cuda_stream(res);
  int64_t n_bits = static_cast<int64_t>(bitmap.universe_size);

  // Create a bitset initialized to zero
  cuvs::core::bitset<uint32_t, int64_t> result(res, n_bits, false);

  uint32_t n_words = static_cast<uint32_t>((n_bits + 31) / 32);
  if (n_words > 0 && bitmap.n_containers > 0) {
    decompress_to_bitset(res, bitmap, result.data(), n_words);
  }
  return result;
}

// ============================================================================
// Set operation kernels (bitmap x bitmap)
// ============================================================================

struct BitmapBitmapWork {
  uint16_t key;
  uint32_t a_offset;  // in uint64_t words
  uint32_t b_offset;
};

template <roaring_set_op OP>
__device__ __forceinline__ uint64_t apply_op(uint64_t a, uint64_t b);

template <>
__device__ __forceinline__ uint64_t apply_op<roaring_set_op::AND>(uint64_t a, uint64_t b)
{
  return a & b;
}
template <>
__device__ __forceinline__ uint64_t apply_op<roaring_set_op::OR>(uint64_t a, uint64_t b)
{
  return a | b;
}
template <>
__device__ __forceinline__ uint64_t apply_op<roaring_set_op::ANDNOT>(uint64_t a, uint64_t b)
{
  return a & ~b;
}
template <>
__device__ __forceinline__ uint64_t apply_op<roaring_set_op::XOR>(uint64_t a, uint64_t b)
{
  return a ^ b;
}

template <roaring_set_op OP>
__global__ void bitmap_bitmap_kernel(const BitmapBitmapWork* work,
                                     uint32_t n_pairs,
                                     const uint64_t* a_data,
                                     const uint64_t* b_data,
                                     uint64_t* out_data,
                                     uint16_t* out_cards)
{
  uint32_t pair_idx = blockIdx.x;
  if (pair_idx >= n_pairs) return;

  const auto& w     = work[pair_idx];
  const uint64_t* a = a_data + w.a_offset;
  const uint64_t* b = b_data + w.b_offset;
  uint64_t* out     = out_data + pair_idx * 1024;

  uint32_t local_pop = 0;
  for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x) {
    uint64_t val = apply_op<OP>(a[i], b[i]);
    out[i]       = val;
    local_pop += __popcll(val);
  }

  // Warp reduction
  for (int off = 16; off > 0; off >>= 1)
    local_pop += __shfl_down_sync(0xFFFFFFFF, local_pop, off);

  __shared__ uint32_t warp_counts[8];
  uint32_t warp_id = threadIdx.x / 32;
  uint32_t lane_id = threadIdx.x % 32;
  if (lane_id == 0) warp_counts[warp_id] = local_pop;
  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t total = 0;
    for (uint32_t w = 0; w < blockDim.x / 32; ++w)
      total += warp_counts[w];
    out_cards[pair_idx] = static_cast<uint16_t>(total > 65535 ? 0 : total);
  }
}

// Expand array/run to bitmap
struct ExpandWork {
  uint32_t src_offset;
  uint16_t cardinality;
  roaring_container_type type;
};

__global__ void expand_to_bitmap_kernel(const ExpandWork* work,
                                        uint32_t n_items,
                                        const uint16_t* array_pool,
                                        const uint16_t* run_pool,
                                        uint64_t* out_bitmaps)
{
  uint32_t idx = blockIdx.x;
  if (idx >= n_items) return;

  const auto& w = work[idx];
  uint64_t* out = out_bitmaps + idx * 1024;

  for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x)
    out[i] = 0;
  __syncthreads();

  if (w.type == roaring_container_type::ARRAY) {
    const uint16_t* arr = array_pool + w.src_offset;
    for (uint32_t i = threadIdx.x; i < w.cardinality; i += blockDim.x) {
      uint16_t val      = arr[i];
      uint32_t word_idx = val / 64u;
      uint64_t bit_mask = 1ULL << (val % 64u);
      atomicOr(reinterpret_cast<unsigned long long*>(&out[word_idx]),
               static_cast<unsigned long long>(bit_mask));
    }
  } else if (w.type == roaring_container_type::RUN) {
    const uint16_t* runs = run_pool + w.src_offset;
    for (uint32_t r = threadIdx.x; r < w.cardinality; r += blockDim.x) {
      uint16_t start  = runs[r * 2];
      uint16_t length = runs[r * 2 + 1];
      for (uint32_t v = start; v <= static_cast<uint32_t>(start) + length; ++v) {
        uint32_t word_idx = v / 64u;
        uint64_t bit_mask = 1ULL << (v % 64u);
        atomicOr(reinterpret_cast<unsigned long long*>(&out[word_idx]),
                 static_cast<unsigned long long>(bit_mask));
      }
    }
  }
}

// Copy bitmap container
__global__ void copy_bitmap_kernel(const uint64_t* src_pool,
                                   const uint32_t* src_offsets,
                                   const bool* from_a,
                                   uint32_t n_items,
                                   const uint64_t* a_pool,
                                   const uint64_t* b_pool,
                                   uint64_t* dst_pool,
                                   uint32_t dst_start)
{
  uint32_t item = blockIdx.x;
  if (item >= n_items) return;

  const uint64_t* src = from_a[item] ? (a_pool + src_offsets[item]) : (b_pool + src_offsets[item]);
  uint64_t* dst       = dst_pool + (dst_start + item) * 1024;
  for (uint32_t i = threadIdx.x; i < 1024u; i += blockDim.x)
    dst[i] = src[i];
}

// ============================================================================
// Host-side helpers
// ============================================================================

struct HostIndex {
  std::vector<uint16_t> keys;
  std::vector<roaring_container_type> types;
  std::vector<uint32_t> offsets;
  std::vector<uint16_t> cardinalities;
};

static HostIndex download_index(const gpu_roaring& g, cudaStream_t stream)
{
  HostIndex h;
  uint32_t n = g.n_containers;
  h.keys.resize(n);
  h.types.resize(n);
  h.offsets.resize(n);
  h.cardinalities.resize(n);
  if (n == 0) return h;

  RAFT_CUDA_TRY(cudaMemcpyAsync(
    h.keys.data(), g.keys.data(), n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(h.types.data(),
                                g.types.data(),
                                n * sizeof(roaring_container_type),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    h.offsets.data(), g.offsets.data(), n * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(h.cardinalities.data(),
                                g.cardinalities.data(),
                                n * sizeof(uint16_t),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return h;
}

static void launch_bb(roaring_set_op op,
                      uint32_t n,
                      const BitmapBitmapWork* d_work,
                      const uint64_t* a,
                      const uint64_t* b,
                      uint64_t* out,
                      uint16_t* cards,
                      cudaStream_t stream)
{
  switch (op) {
    case roaring_set_op::AND:
      bitmap_bitmap_kernel<roaring_set_op::AND><<<n, 256, 0, stream>>>(d_work, n, a, b, out, cards);
      break;
    case roaring_set_op::OR:
      bitmap_bitmap_kernel<roaring_set_op::OR><<<n, 256, 0, stream>>>(d_work, n, a, b, out, cards);
      break;
    case roaring_set_op::ANDNOT:
      bitmap_bitmap_kernel<roaring_set_op::ANDNOT>
        <<<n, 256, 0, stream>>>(d_work, n, a, b, out, cards);
      break;
    case roaring_set_op::XOR:
      bitmap_bitmap_kernel<roaring_set_op::XOR><<<n, 256, 0, stream>>>(d_work, n, a, b, out, cards);
      break;
  }
}

// ============================================================================
// set_op: the main set operation implementation
// ============================================================================

gpu_roaring set_op(raft::resources const& res,
                   const gpu_roaring& a,
                   const gpu_roaring& b,
                   roaring_set_op op)
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto ha     = download_index(a, stream);
  auto hb     = download_index(b, stream);

  // Container matching (merge of sorted key arrays)
  struct WorkItem {
    uint16_t key;
    int32_t a_idx, b_idx;
    roaring_container_type a_type, b_type;
  };
  std::vector<WorkItem> work;
  work.reserve(a.n_containers + b.n_containers);

  uint32_t ia = 0, ib = 0;
  while (ia < a.n_containers && ib < b.n_containers) {
    if (ha.keys[ia] < hb.keys[ib]) {
      if (op == roaring_set_op::OR || op == roaring_set_op::XOR || op == roaring_set_op::ANDNOT)
        work.push_back({ha.keys[ia], (int32_t)ia, -1, ha.types[ia], roaring_container_type::ARRAY});
      ++ia;
    } else if (ha.keys[ia] > hb.keys[ib]) {
      if (op == roaring_set_op::OR || op == roaring_set_op::XOR)
        work.push_back({hb.keys[ib], -1, (int32_t)ib, roaring_container_type::ARRAY, hb.types[ib]});
      ++ib;
    } else {
      work.push_back({ha.keys[ia], (int32_t)ia, (int32_t)ib, ha.types[ia], hb.types[ib]});
      ++ia;
      ++ib;
    }
  }
  while (ia < a.n_containers) {
    if (op == roaring_set_op::OR || op == roaring_set_op::XOR || op == roaring_set_op::ANDNOT)
      work.push_back({ha.keys[ia], (int32_t)ia, -1, ha.types[ia], roaring_container_type::ARRAY});
    ++ia;
  }
  while (ib < b.n_containers) {
    if (op == roaring_set_op::OR || op == roaring_set_op::XOR)
      work.push_back({hb.keys[ib], -1, (int32_t)ib, roaring_container_type::ARRAY, hb.types[ib]});
    ++ib;
  }

  if (work.empty()) return gpu_roaring(stream);

  // Classify by type pair
  struct BBW {
    uint16_t key;
    uint32_t a_off, b_off;
  };
  struct CopyW {
    uint16_t key;
    uint32_t offset;
    uint16_t card;
    roaring_container_type type;
    bool from_a;
  };
  struct MixedW {
    uint16_t key;
    roaring_container_type a_type, b_type;
    uint32_t a_off, b_off;
    uint16_t a_card, b_card;
    bool a_is_bmp, b_is_bmp;
  };

  std::vector<BBW> bb_work;
  std::vector<CopyW> copy_bmp, copy_arr, copy_run;
  std::vector<MixedW> mixed;

  auto elem_off = [](roaring_container_type t, uint32_t byte_off) -> uint32_t {
    if (t == roaring_container_type::BITMAP) return byte_off / sizeof(uint64_t);
    return byte_off / sizeof(uint16_t);
  };

  for (auto& wi : work) {
    if (wi.a_idx < 0) {
      auto& ho  = hb;
      int32_t i = wi.b_idx;
      CopyW cw  = {wi.key, ho.offsets[i], ho.cardinalities[i], ho.types[i], false};
      if (ho.types[i] == roaring_container_type::BITMAP)
        copy_bmp.push_back(cw);
      else if (ho.types[i] == roaring_container_type::ARRAY)
        copy_arr.push_back(cw);
      else
        copy_run.push_back(cw);
    } else if (wi.b_idx < 0) {
      auto& ho  = ha;
      int32_t i = wi.a_idx;
      CopyW cw  = {wi.key, ho.offsets[i], ho.cardinalities[i], ho.types[i], true};
      if (ho.types[i] == roaring_container_type::BITMAP)
        copy_bmp.push_back(cw);
      else if (ho.types[i] == roaring_container_type::ARRAY)
        copy_arr.push_back(cw);
      else
        copy_run.push_back(cw);
    } else {
      auto ta = ha.types[wi.a_idx];
      auto tb = hb.types[wi.b_idx];
      if (ta == roaring_container_type::BITMAP && tb == roaring_container_type::BITMAP) {
        bb_work.push_back({wi.key,
                           ha.offsets[wi.a_idx] / (uint32_t)sizeof(uint64_t),
                           hb.offsets[wi.b_idx] / (uint32_t)sizeof(uint64_t)});
      } else {
        // All non-bitmap×bitmap: expand to bitmap
        mixed.push_back({wi.key,
                         ta,
                         tb,
                         elem_off(ta, ha.offsets[wi.a_idx]),
                         elem_off(tb, hb.offsets[wi.b_idx]),
                         ha.cardinalities[wi.a_idx],
                         hb.cardinalities[wi.b_idx],
                         ta == roaring_container_type::BITMAP,
                         tb == roaring_container_type::BITMAP});
      }
    }
  }

  uint32_t out_n_bitmap = static_cast<uint32_t>(bb_work.size() + mixed.size() + copy_bmp.size());

  // Allocate output bitmap pool
  rmm::device_uvector<uint64_t> d_out_bmp(static_cast<size_t>(out_n_bitmap) * 1024, stream);

  // Execute bitmap×bitmap
  std::vector<uint16_t> bb_cards(bb_work.size());
  if (!bb_work.empty()) {
    uint32_t n_bb = static_cast<uint32_t>(bb_work.size());
    rmm::device_uvector<BitmapBitmapWork> d_bb(n_bb, stream);
    rmm::device_uvector<uint16_t> d_bb_cards(n_bb, stream);
    raft::update_device(
      d_bb.data(), reinterpret_cast<const BitmapBitmapWork*>(bb_work.data()), n_bb, stream);

    launch_bb(op,
              n_bb,
              d_bb.data(),
              a.bitmap_data.data(),
              b.bitmap_data.data(),
              d_out_bmp.data(),
              d_bb_cards.data(),
              stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    raft::update_host(bb_cards.data(), d_bb_cards.data(), n_bb, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  // Execute mixed pairs (expand + bitmap op)
  std::vector<uint16_t> mixed_cards(mixed.size());
  rmm::device_uvector<uint64_t> d_tmp_a(0, stream);
  rmm::device_uvector<uint64_t> d_tmp_b(0, stream);
  if (!mixed.empty()) {
    uint32_t n_mx = static_cast<uint32_t>(mixed.size());
    d_tmp_a.resize(static_cast<size_t>(n_mx) * 1024, stream);
    d_tmp_b.resize(static_cast<size_t>(n_mx) * 1024, stream);

    // Build expand work
    std::vector<ExpandWork> exp_a(n_mx), exp_b(n_mx);
    for (uint32_t i = 0; i < n_mx; ++i) {
      exp_a[i] = {mixed[i].a_off, mixed[i].a_card, mixed[i].a_type};
      exp_b[i] = {mixed[i].b_off, mixed[i].b_card, mixed[i].b_type};
    }

    rmm::device_uvector<ExpandWork> d_exp_a(n_mx, stream);
    rmm::device_uvector<ExpandWork> d_exp_b(n_mx, stream);
    raft::update_device(d_exp_a.data(), exp_a.data(), n_mx, stream);
    raft::update_device(d_exp_b.data(), exp_b.data(), n_mx, stream);

    RAFT_CUDA_TRY(cudaMemsetAsync(d_tmp_a.data(), 0, d_tmp_a.size() * sizeof(uint64_t), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_tmp_b.data(), 0, d_tmp_b.size() * sizeof(uint64_t), stream));

    expand_to_bitmap_kernel<<<n_mx, 256, 0, stream>>>(
      d_exp_a.data(), n_mx, a.array_data.data(), a.run_data.data(), d_tmp_a.data());
    expand_to_bitmap_kernel<<<n_mx, 256, 0, stream>>>(
      d_exp_b.data(), n_mx, b.array_data.data(), b.run_data.data(), d_tmp_b.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Copy bitmap containers from source pools into temp buffers
    for (uint32_t i = 0; i < n_mx; ++i) {
      if (mixed[i].a_is_bmp) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(d_tmp_a.data() + i * 1024,
                                      a.bitmap_data.data() + mixed[i].a_off,
                                      1024 * sizeof(uint64_t),
                                      cudaMemcpyDeviceToDevice,
                                      stream));
      }
      if (mixed[i].b_is_bmp) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(d_tmp_b.data() + i * 1024,
                                      b.bitmap_data.data() + mixed[i].b_off,
                                      1024 * sizeof(uint64_t),
                                      cudaMemcpyDeviceToDevice,
                                      stream));
      }
    }

    // Build BitmapBitmapWork for temp pools
    std::vector<BitmapBitmapWork> mx_bb(n_mx);
    for (uint32_t i = 0; i < n_mx; ++i)
      mx_bb[i] = {mixed[i].key, i * 1024, i * 1024};

    rmm::device_uvector<BitmapBitmapWork> d_mx_bb(n_mx, stream);
    rmm::device_uvector<uint16_t> d_mx_cards(n_mx, stream);
    raft::update_device(d_mx_bb.data(), mx_bb.data(), n_mx, stream);

    uint32_t bb_sz = static_cast<uint32_t>(bb_work.size());
    launch_bb(op,
              n_mx,
              d_mx_bb.data(),
              d_tmp_a.data(),
              d_tmp_b.data(),
              d_out_bmp.data() + static_cast<size_t>(bb_sz) * 1024,
              d_mx_cards.data(),
              stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    raft::update_host(mixed_cards.data(), d_mx_cards.data(), n_mx, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  // Copy-through bitmap containers
  if (!copy_bmp.empty()) {
    uint32_t n_cp  = static_cast<uint32_t>(copy_bmp.size());
    uint32_t start = static_cast<uint32_t>(bb_work.size() + mixed.size());
    for (uint32_t i = 0; i < n_cp; ++i) {
      const uint64_t* src = copy_bmp[i].from_a
                              ? a.bitmap_data.data() + copy_bmp[i].offset / sizeof(uint64_t)
                              : b.bitmap_data.data() + copy_bmp[i].offset / sizeof(uint64_t);
      RAFT_CUDA_TRY(cudaMemcpyAsync(d_out_bmp.data() + (start + i) * 1024,
                                    src,
                                    1024 * sizeof(uint64_t),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    }
  }

  // Copy-through array containers
  uint32_t total_arr = 0;
  std::vector<uint32_t> ca_off(copy_arr.size());
  for (size_t i = 0; i < copy_arr.size(); ++i) {
    ca_off[i] = total_arr;
    total_arr += copy_arr[i].card;
  }
  rmm::device_uvector<uint16_t> d_out_arr(total_arr, stream);
  for (size_t i = 0; i < copy_arr.size(); ++i) {
    const uint16_t* src = copy_arr[i].from_a
                            ? a.array_data.data() + copy_arr[i].offset / sizeof(uint16_t)
                            : b.array_data.data() + copy_arr[i].offset / sizeof(uint16_t);
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_out_arr.data() + ca_off[i],
                                  src,
                                  copy_arr[i].card * sizeof(uint16_t),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  }

  // Copy-through run containers
  uint32_t total_run_vals = 0;
  std::vector<uint32_t> cr_off(copy_run.size());
  for (size_t i = 0; i < copy_run.size(); ++i) {
    cr_off[i] = total_run_vals;
    total_run_vals += copy_run[i].card * 2;
  }
  rmm::device_uvector<uint16_t> d_out_run(total_run_vals, stream);
  for (size_t i = 0; i < copy_run.size(); ++i) {
    const uint16_t* src = copy_run[i].from_a
                            ? a.run_data.data() + copy_run[i].offset / sizeof(uint16_t)
                            : b.run_data.data() + copy_run[i].offset / sizeof(uint16_t);
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_out_run.data() + cr_off[i],
                                  src,
                                  copy_run[i].card * 2 * sizeof(uint16_t),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  }

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // Assemble output
  std::vector<uint16_t> out_keys;
  std::vector<roaring_container_type> out_types;
  std::vector<uint32_t> out_offsets;
  std::vector<uint16_t> out_cards;

  uint32_t bb_idx = 0, mx_idx = 0, cb_idx = 0, ca_idx = 0, cr_idx = 0;

  for (auto& wi : work) {
    if (wi.a_idx < 0 || wi.b_idx < 0) {
      // Copy-through
      auto& ho        = (wi.a_idx < 0) ? hb : ha;
      int32_t src_idx = (wi.a_idx < 0) ? wi.b_idx : wi.a_idx;
      auto ct         = ho.types[src_idx];
      if (ct == roaring_container_type::BITMAP) {
        uint32_t slot = static_cast<uint32_t>(bb_work.size() + mixed.size()) + cb_idx;
        out_keys.push_back(wi.key);
        out_types.push_back(roaring_container_type::BITMAP);
        out_offsets.push_back(slot * 1024 * sizeof(uint64_t));
        out_cards.push_back(ho.cardinalities[src_idx]);
        ++cb_idx;
      } else if (ct == roaring_container_type::ARRAY) {
        out_keys.push_back(wi.key);
        out_types.push_back(roaring_container_type::ARRAY);
        out_offsets.push_back(ca_off[ca_idx] * sizeof(uint16_t));
        out_cards.push_back(copy_arr[ca_idx].card);
        ++ca_idx;
      } else {
        out_keys.push_back(wi.key);
        out_types.push_back(roaring_container_type::RUN);
        out_offsets.push_back(cr_off[cr_idx] * sizeof(uint16_t));
        out_cards.push_back(copy_run[cr_idx].card);
        ++cr_idx;
      }
    } else {
      auto ta = ha.types[wi.a_idx];
      auto tb = hb.types[wi.b_idx];
      if (ta == roaring_container_type::BITMAP && tb == roaring_container_type::BITMAP) {
        uint16_t card = bb_cards[bb_idx];
        if (card == 0 && op != roaring_set_op::OR) {
          ++bb_idx;
          continue;
        }
        out_keys.push_back(wi.key);
        out_types.push_back(roaring_container_type::BITMAP);
        out_offsets.push_back(bb_idx * 1024 * sizeof(uint64_t));
        out_cards.push_back(card);
        ++bb_idx;
      } else {
        uint16_t card = mixed_cards[mx_idx];
        if (card == 0 && op != roaring_set_op::OR) {
          ++mx_idx;
          continue;
        }
        uint32_t slot = static_cast<uint32_t>(bb_work.size()) + mx_idx;
        out_keys.push_back(wi.key);
        out_types.push_back(roaring_container_type::BITMAP);
        out_offsets.push_back(slot * 1024 * sizeof(uint64_t));
        out_cards.push_back(card);
        ++mx_idx;
      }
    }
  }

  // Build result
  gpu_roaring result(stream);
  result.n_containers        = static_cast<uint32_t>(out_keys.size());
  result.universe_size       = std::max(a.universe_size, b.universe_size);
  result.bitmap_data         = std::move(d_out_bmp);
  result.n_bitmap_containers = out_n_bitmap;
  result.array_data          = std::move(d_out_arr);
  result.n_array_containers  = static_cast<uint32_t>(copy_arr.size());
  result.run_data            = std::move(d_out_run);
  result.n_run_containers    = static_cast<uint32_t>(copy_run.size());

  if (result.n_containers > 0) {
    result.keys.resize(result.n_containers, stream);
    result.types.resize(result.n_containers, stream);
    result.offsets.resize(result.n_containers, stream);
    result.cardinalities.resize(result.n_containers, stream);
    raft::update_device(result.keys.data(), out_keys.data(), result.n_containers, stream);
    raft::update_device(result.types.data(), out_types.data(), result.n_containers, stream);
    raft::update_device(result.offsets.data(), out_offsets.data(), result.n_containers, stream);
    raft::update_device(result.cardinalities.data(), out_cards.data(), result.n_containers, stream);
    raft::resource::sync_stream(res);
  }

  return result;
}

// ============================================================================
// Multi-AND / Multi-OR
// ============================================================================

gpu_roaring multi_and(raft::resources const& res, const gpu_roaring* bitmaps, uint32_t count)
{
  if (count == 0) return gpu_roaring(raft::resource::get_cuda_stream(res));
  if (count == 1) return set_op(res, bitmaps[0], bitmaps[0], roaring_set_op::AND);

  gpu_roaring result = set_op(res, bitmaps[0], bitmaps[1], roaring_set_op::AND);
  for (uint32_t i = 2; i < count; ++i) {
    gpu_roaring next = set_op(res, result, bitmaps[i], roaring_set_op::AND);
    result           = std::move(next);
  }
  return result;
}

gpu_roaring multi_or(raft::resources const& res, const gpu_roaring* bitmaps, uint32_t count)
{
  if (count == 0) return gpu_roaring(raft::resource::get_cuda_stream(res));
  if (count == 1) return set_op(res, bitmaps[0], bitmaps[0], roaring_set_op::AND);

  gpu_roaring result = set_op(res, bitmaps[0], bitmaps[1], roaring_set_op::OR);
  for (uint32_t i = 2; i < count; ++i) {
    gpu_roaring next = set_op(res, result, bitmaps[i], roaring_set_op::OR);
    result           = std::move(next);
  }
  return result;
}

// ============================================================================
// from_sorted_ids (device input)
// ============================================================================

gpu_roaring from_sorted_ids(raft::resources const& res,
                            raft::device_vector_view<const uint32_t, int64_t> sorted_ids,
                            uint32_t universe_size)
{
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<uint32_t> h_ids(sorted_ids.extent(0));
  raft::update_host(h_ids.data(), sorted_ids.data_handle(), h_ids.size(), stream);
  raft::resource::sync_stream(res);
  return from_sorted_ids(res, h_ids.data(), static_cast<uint32_t>(h_ids.size()), universe_size);
}

// ============================================================================
// Batched CSR emission: every container of every bitmap in one launch.
// One CTA per container; each container writes its sorted member ids into
// its precomputed segment of the global indices array (offsets derive from
// host-side element counts — no count kernels, no syncs).
// ============================================================================

namespace {

constexpr uint32_t kEmitBlock = 256;

__device__ inline void emit_block_scan(uint32_t* smem, uint32_t tid)
{
  for (uint32_t stride = 1; stride < kEmitBlock; stride *= 2) {
    uint32_t v = (tid >= stride) ? smem[tid - stride] : 0;
    __syncthreads();
    smem[tid] += v;
    __syncthreads();
  }
}

__global__ void emit_csr_indices_kernel(const uint32_t* keys,
                                        const uint8_t* types,
                                        const uint32_t* elem_counts,
                                        const void* const* data,
                                        const int64_t* out_offsets,
                                        uint32_t n_containers,
                                        int64_t* indices)
{
  uint32_t cid = blockIdx.x;
  if (cid >= n_containers) return;
  int64_t base_id   = static_cast<int64_t>(keys[cid]) << 16;
  int64_t out_start = out_offsets[cid];
  uint32_t tid      = threadIdx.x;
  auto ctype        = static_cast<roaring_container_type>(types[cid]);

  if (ctype == roaring_container_type::ARRAY) {
    const uint16_t* arr = static_cast<const uint16_t*>(data[cid]);
    for (uint32_t i = tid; i < elem_counts[cid]; i += kEmitBlock)
      indices[out_start + i] = base_id | arr[i];
  } else if (ctype == roaring_container_type::BITMAP) {
    const uint64_t* bmp = static_cast<const uint64_t*>(data[cid]);
    __shared__ uint32_t counts[kEmitBlock];
    uint32_t mine = 0;
    for (uint32_t w = tid * 4; w < tid * 4 + 4; ++w)
      mine += static_cast<uint32_t>(__popcll(bmp[w]));
    counts[tid] = mine;
    __syncthreads();
    emit_block_scan(counts, tid);
    uint32_t pos = (tid > 0) ? counts[tid - 1] : 0;
    for (uint32_t w = tid * 4; w < tid * 4 + 4; ++w) {
      uint64_t word = bmp[w];
      while (word != 0) {
        uint32_t bit = static_cast<uint32_t>(__ffsll(static_cast<long long>(word))) - 1;
        indices[out_start + pos++] = base_id | (w * 64 + bit);
        word &= word - 1;
      }
    }
  }
  // RUN containers are rejected host-side (not produced by v1 construction).
}

}  // namespace

void to_csr_indices(raft::resources const& res,
                    const gpu_roaring* const* bitmaps,
                    uint32_t n_bitmaps,
                    int64_t* indices)
{
  auto stream = raft::resource::get_cuda_stream(res);

  // flatten per-container descriptors from the host mirrors
  std::vector<uint32_t> keys, counts;
  std::vector<uint8_t> types;
  std::vector<const void*> data;
  std::vector<int64_t> out_offsets;
  int64_t cursor = 0;
  for (uint32_t b = 0; b < n_bitmaps; ++b) {
    const gpu_roaring& r = *bitmaps[b];
    RAFT_EXPECTS(!r.negated, "to_csr_indices: negated bitmaps are not supported");
    RAFT_EXPECTS(r.h_keys.size() == r.n_containers,
                 "to_csr_indices: bitmap lacks host container metadata "
                 "(construct via from_sorted_ids)");
    for (uint32_t c = 0; c < r.n_containers; ++c) {
      keys.push_back(r.h_keys[c]);
      types.push_back(static_cast<uint8_t>(r.h_types[c]));
      out_offsets.push_back(cursor);
      switch (r.h_types[c]) {
        case roaring_container_type::ARRAY:
          data.push_back(r.array_data.data() + r.h_offsets[c] / sizeof(uint16_t));
          counts.push_back(r.h_element_counts[c]);
          cursor += r.h_element_counts[c];
          break;
        case roaring_container_type::BITMAP:
          data.push_back(r.bitmap_data.data() + r.h_offsets[c] / sizeof(uint64_t));
          counts.push_back(r.h_element_counts[c]);
          cursor += r.h_element_counts[c];
          break;
        default: RAFT_FAIL("to_csr_indices: RUN containers are not produced by v1 construction");
      }
    }
  }
  uint32_t n_c = static_cast<uint32_t>(keys.size());
  if (n_c == 0) return;

  rmm::device_uvector<uint32_t> d_keys(n_c, stream), d_counts(n_c, stream);
  rmm::device_uvector<uint8_t> d_types(n_c, stream);
  rmm::device_uvector<const void*> d_data(n_c, stream);
  rmm::device_uvector<int64_t> d_off(n_c, stream);
  raft::update_device(d_keys.data(), keys.data(), n_c, stream);
  raft::update_device(d_counts.data(), counts.data(), n_c, stream);
  raft::update_device(d_types.data(), types.data(), n_c, stream);
  raft::update_device(d_data.data(), data.data(), n_c, stream);
  raft::update_device(d_off.data(), out_offsets.data(), n_c, stream);

  emit_csr_indices_kernel<<<n_c, kEmitBlock, 0, stream>>>(
    d_keys.data(), d_types.data(), d_counts.data(), d_data.data(), d_off.data(), n_c, indices);
  RAFT_CUDA_TRY(cudaGetLastError());
}

// ============================================================================
// Batched decompression into a dense [n_bitmaps, n_rows] bit matrix
// ============================================================================

namespace {

__global__ void decompress_bitmap_row_kernel(const uint32_t* keys,
                                             const uint8_t* types,
                                             const uint32_t* elem_counts,
                                             const void* const* data,
                                             const int64_t* row_of,
                                             uint32_t n_containers,
                                             int64_t n_rows,
                                             uint32_t* output)
{
  uint32_t cid = blockIdx.x;
  if (cid >= n_containers) return;
  int64_t row_bit_base = row_of[cid] * n_rows;
  uint32_t key         = keys[cid];
  uint32_t tid         = threadIdx.x;
  auto ctype           = static_cast<roaring_container_type>(types[cid]);

  auto set_bit = [&](uint32_t low) {
    int64_t bit = row_bit_base + ((static_cast<int64_t>(key) << 16) | low);
    atomicOr(&output[bit >> 5], 1u << (bit & 31));
  };

  if (ctype == roaring_container_type::ARRAY) {
    const uint16_t* arr = static_cast<const uint16_t*>(data[cid]);
    for (uint32_t i = tid; i < elem_counts[cid]; i += blockDim.x)
      set_bit(arr[i]);
  } else if (ctype == roaring_container_type::BITMAP) {
    const uint64_t* bmp = static_cast<const uint64_t*>(data[cid]);
    for (uint32_t w = tid; w < 1024; w += blockDim.x) {
      uint64_t word = bmp[w];
      while (word != 0) {
        uint32_t bit = static_cast<uint32_t>(__ffsll(static_cast<long long>(word))) - 1;
        set_bit(w * 64 + bit);
        word &= word - 1;
      }
    }
  }
  // RUN containers are rejected host-side (not produced by v1 construction).
}

}  // namespace

void decompress_to_bitmap(raft::resources const& res,
                          const gpu_roaring* const* bitmaps,
                          uint32_t n_bitmaps,
                          int64_t n_rows,
                          uint32_t* output)
{
  auto stream = raft::resource::get_cuda_stream(res);

  std::vector<uint32_t> keys, counts;
  std::vector<uint8_t> types;
  std::vector<const void*> data;
  std::vector<int64_t> row_of;
  for (uint32_t b = 0; b < n_bitmaps; ++b) {
    const gpu_roaring& r = *bitmaps[b];
    RAFT_EXPECTS(!r.negated, "decompress_to_bitmap: negated bitmaps are not supported");
    RAFT_EXPECTS(r.h_keys.size() == r.n_containers,
                 "decompress_to_bitmap: bitmap lacks host container metadata "
                 "(construct via from_sorted_ids)");
    for (uint32_t c = 0; c < r.n_containers; ++c) {
      keys.push_back(r.h_keys[c]);
      types.push_back(static_cast<uint8_t>(r.h_types[c]));
      row_of.push_back(static_cast<int64_t>(b));
      counts.push_back(r.h_element_counts[c]);
      switch (r.h_types[c]) {
        case roaring_container_type::ARRAY:
          data.push_back(r.array_data.data() + r.h_offsets[c] / sizeof(uint16_t));
          break;
        case roaring_container_type::BITMAP:
          data.push_back(r.bitmap_data.data() + r.h_offsets[c] / sizeof(uint64_t));
          break;
        default:
          RAFT_FAIL("decompress_to_bitmap: RUN containers are not produced by v1 construction");
      }
    }
  }
  uint32_t n_c = static_cast<uint32_t>(keys.size());
  if (n_c == 0) return;

  rmm::device_uvector<uint32_t> d_keys(n_c, stream), d_counts(n_c, stream);
  rmm::device_uvector<uint8_t> d_types(n_c, stream);
  rmm::device_uvector<const void*> d_data(n_c, stream);
  rmm::device_uvector<int64_t> d_row(n_c, stream);
  raft::update_device(d_keys.data(), keys.data(), n_c, stream);
  raft::update_device(d_counts.data(), counts.data(), n_c, stream);
  raft::update_device(d_types.data(), types.data(), n_c, stream);
  raft::update_device(d_data.data(), data.data(), n_c, stream);
  raft::update_device(d_row.data(), row_of.data(), n_c, stream);

  decompress_bitmap_row_kernel<<<n_c, 256, 0, stream>>>(d_keys.data(),
                                                        d_types.data(),
                                                        d_counts.data(),
                                                        d_data.data(),
                                                        d_row.data(),
                                                        n_c,
                                                        n_rows,
                                                        output);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace cuvs::core
