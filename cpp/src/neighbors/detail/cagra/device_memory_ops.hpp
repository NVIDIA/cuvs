/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

namespace cuvs::neighbors::cagra::detail::device {

RAFT_DEVICE_INLINE_FUNCTION void lds(float& x, uint32_t addr)
{
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half& x, uint32_t addr)
{
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(reinterpret_cast<uint16_t&>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half2& x, uint32_t addr)
{
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(reinterpret_cast<uint32_t&>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[1], uint32_t addr)
{
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*reinterpret_cast<uint16_t*>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[2], uint32_t addr)
{
  asm volatile("ld.shared.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[4], uint32_t addr)
{
  asm volatile("ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint8_t& x, uint32_t addr)
{
  uint32_t res;
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(res) : "r"(addr));
  x = static_cast<uint32_t>(res);
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint32_t& x, uint32_t addr)
{
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint64_t& x, uint32_t addr)
{
  asm volatile("ld.shared.u64 {%0}, [%1];" : "=l"(x) : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint32_t& x, const uint32_t* addr)
{
  lds(x, uint32_t(__cvta_generic_to_shared(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint4& x, uint32_t addr)
{
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint4& x, const uint4* addr)
{
  lds(x, uint32_t(__cvta_generic_to_shared(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void sts(uint32_t addr, const uint32_t& x)
{
  asm volatile("st.shared.u32 [%0], %1;" : : "r"(addr), "r"(reinterpret_cast<const uint32_t&>(x)));
}

RAFT_DEVICE_INLINE_FUNCTION void sts(uint32_t addr, const uint64_t& x)
{
  asm volatile("st.shared.u64 [%0], %1;" : : "r"(addr), "l"(reinterpret_cast<const uint64_t&>(x)));
}

RAFT_DEVICE_INLINE_FUNCTION void sts(uint32_t addr, const half2& x)
{
  asm volatile("st.shared.v2.u16 [%0], {%1, %2};"
               :
               : "r"(addr),
                 "h"(reinterpret_cast<const uint16_t&>(x.x)),
                 "h"(reinterpret_cast<const uint16_t&>(x.y)));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_cg(uint4& x, const uint4* addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(uint4& x, const uint4* addr)
{
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(uint32_t& x, const uint32_t* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_cg(uint32_t& x, const uint32_t* addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half& x, const half* addr)
{
  asm volatile("ld.global.ca.u16 {%0}, [%1];"
               : "=h"(reinterpret_cast<uint16_t&>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[1], const half* addr)
{
  asm volatile("ld.global.ca.u16 {%0}, [%1];"
               : "=h"(*reinterpret_cast<uint16_t*>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[2], const half* addr)
{
  asm volatile("ld.global.ca.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[4], const half* addr)
{
  asm volatile("ld.global.ca.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2& x, const half* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(reinterpret_cast<uint32_t&>(x))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2 (&x)[1], const half* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(*reinterpret_cast<uint32_t*>(x))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2 (&x)[2], const half* addr)
{
  asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];"
               : "=r"(*reinterpret_cast<uint32_t*>(x)), "=r"(*reinterpret_cast<uint32_t*>(x + 1))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}

}  // namespace cuvs::neighbors::cagra::detail::device
