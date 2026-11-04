import ctypes
import sys

cu = ctypes.CDLL("libcuda.so.1")


def chk(name, *args):
    r = getattr(cu, name)(*args)
    if r != 0:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(r, ctypes.byref(s))
        print(f"  {name} -> {r} {s.value}")
    return r


class Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class Prop(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", Loc),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", AllocFlags),
    ]


class AccessDesc(ctypes.Structure):
    _fields_ = [("location", Loc), ("flags", ctypes.c_int)]


chk("cuInit", 0)
dev = ctypes.c_int()
chk("cuDeviceGet", ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
chk("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
chk("cuCtxSetCurrent", ctx)
ver = ctypes.c_int()
cu.cuDriverGetVersion(ctypes.byref(ver))
print("driver", ver.value)
for a, n in [
    (102, "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED"),
    (116, "GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED"),
    (103, "HANDLE_TYPE_POSIX_FD_SUPPORTED"),
    (129, "HOST_NUMA_ID"),
]:
    v = ctypes.c_int(-99)
    chk("cuDeviceGetAttribute", ctypes.byref(v), a, dev)
    print(f"attr {n} = {v.value}")


def trial(handle_type, rdma):
    p = Prop()
    p.type = 1  # PINNED
    p.location.type = 1  # DEVICE
    p.location.id = 0
    p.requestedHandleTypes = handle_type
    p.allocFlags.gpuDirectRDMACapable = rdma
    g = ctypes.c_size_t()
    if chk("cuMemGetAllocationGranularity", ctypes.byref(g), ctypes.byref(p), 0):
        return "granularity failed"
    size = g.value
    h = ctypes.c_ulonglong()
    if chk("cuMemCreate", ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(p), ctypes.c_ulonglong(0)):
        return "create failed"
    ptr = ctypes.c_ulonglong()
    if chk("cuMemAddressReserve", ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_size_t(g.value), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)):
        return "reserve failed"
    if chk("cuMemMap", ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), h, ctypes.c_ulonglong(0)):
        return "map failed"
    d = AccessDesc()
    d.location.type = 1
    d.location.id = 0
    d.flags = 3
    if chk("cuMemSetAccess", ptr, ctypes.c_size_t(size), ctypes.byref(d), ctypes.c_size_t(1)):
        return "setaccess failed"
    chk("cuMemUnmap", ptr, ctypes.c_size_t(size))
    chk("cuMemAddressFree", ptr, ctypes.c_size_t(size))
    chk("cuMemRelease", h)
    return f"ok (granularity {size})"


# CU_MEM_HANDLE_TYPE_NONE=0, POSIX_FD=1; NCCL uses POSIX_FD by default.
for ht in (0, 1):
    for rdma in (0, 1):
        print(f"handle_type={ht} gpuDirectRDMACapable={rdma}:", trial(ht, rdma))
        sys.stdout.flush()
