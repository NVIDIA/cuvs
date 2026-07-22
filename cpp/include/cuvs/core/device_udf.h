/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#ifndef CUVS_CORE_DEVICE_UDF_H
#define CUVS_CORE_DEVICE_UDF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Device UDF payload representation. */
typedef enum cuvsDeviceUDFPayloadKind {
  CUVS_DEVICE_UDF_PAYLOAD_LTOIR = 1,
  CUVS_DEVICE_UDF_PAYLOAD_CUDA_SOURCE = 2,
} cuvsDeviceUDFPayloadKind;

/** Capture flags for cuvsUDFCapture. */
enum { CUVS_UDF_CAPTURE_READONLY = 1u };

/** Borrowed capture descriptor for a device UDF. */
typedef struct cuvsUDFCapture {
  /** Capture name as used by the frontend, e.g. "weights". */
  const char* name;
  /** Logical dtype string, e.g. "float32". */
  const char* dtype;
  /** Optional shape array of length ndim. */
  const int64_t* shape;
  /** Optional strides array of length ndim, in bytes. Null means contiguous/default. */
  const int64_t* strides;
  /** Number of dimensions in shape/strides. */
  int32_t ndim;
  /** CUDA device ordinal for the capture allocation. */
  int32_t device_id;
  /** CUDA device pointer for the capture allocation. */
  uintptr_t pointer;
  /** Bitmask of CUVS_UDF_CAPTURE_* flags. */
  uint32_t flags;
} cuvsUDFCapture;

/** Borrowed device UDF descriptor. Payload and captures are copied by consumers. */
typedef struct cuvsDeviceUDF {
  /** ABI identifier, e.g. "rapids.cuvs.ivf_flat.metric.v1". */
  const char* abi;
  /** Payload representation. */
  cuvsDeviceUDFPayloadKind payload_kind;
  /** Borrowed payload bytes. */
  const void* payload;
  /** Number of bytes in payload. */
  size_t payload_size;
  /** Optional externally visible device symbol in payload. */
  const char* symbol_name;
  /** Borrowed capture descriptors. */
  const cuvsUDFCapture* captures;
  /** Number of capture descriptors. */
  size_t n_captures;
  /** Cache key used to identify this UDF artifact. */
  const char* cache_key;
  /** Reserved flags; must be zero for now. */
  uint32_t flags;
} cuvsDeviceUDF;

typedef const cuvsDeviceUDF* cuvsDeviceUDF_t;

#ifdef __cplusplus
}
#endif

#endif  // CUVS_CORE_DEVICE_UDF_H
