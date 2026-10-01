/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char** argv)
{
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <libcuvs_c.so>\n", argv[0]);
    return 2;
  }

  void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    fprintf(stderr, "Failed to load %s: %s\n", argv[1], dlerror());
    return 1;
  }

  dlerror();
  void* symbol      = dlsym(handle, "cuvsResourcesCreate");
  const char* error = dlerror();
  if (error != NULL || symbol == NULL) {
    fprintf(stderr,
            "Failed to resolve cuvsResourcesCreate: %s\n",
            error != NULL ? error : "unknown error");
    dlclose(handle);
    return 1;
  }

  printf("Successfully loaded %s and resolved cuvsResourcesCreate\n", argv[1]);
  dlclose(handle);
  return 0;
}
