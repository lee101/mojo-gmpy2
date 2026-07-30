"""ctypes bridge to the single Mojo arbitrary-precision compilation unit."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "gmpy2.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-gmpy2.so")

P = ctypes.c_void_p
N = ctypes.c_ssize_t

_SIGNATURES = {
    "mg_add": ([P, N, P, N, P], N),
    "mg_sub": ([P, N, P, N, P], N),
    "mg_mul": ([P, N, P, N, P], N),
    "mg_gcd": ([P, N, P, N, P, P, P], N),
    "mg_powmod": ([P, P, N, P, N, P, P, P, P, P], N),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    pixi = shutil.which("pixi")
    if shutil.which("mojo"):
        cmd = ["bash", os.path.join(ROOT, "build", "build.sh")]
    elif pixi:
        cmd = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        raise BuildError("neither mojo nor pixi is available")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(a: np.ndarray) -> ctypes.c_void_p:
    if a.dtype != np.dtype(np.uint32) or a.ndim != 1 or not a.flags.c_contiguous:
        raise TypeError("limb buffers must be contiguous one-dimensional uint32 arrays")
    if a.size == 0:
        raise ValueError("limb buffers must not be empty")
    return ctypes.c_void_p(a.ctypes.data)


def limbs(value: int, size: int = 0) -> np.ndarray:
    value = abs(int(value))
    count = max(size, max(1, (value.bit_length() + 31) // 32))
    raw = value.to_bytes(count * 4, "little")
    return np.frombuffer(raw, dtype="<u4")


def from_limbs(words: np.ndarray, n: int | None = None) -> int:
    if words.dtype != np.dtype(np.uint32) or words.ndim != 1 or not words.flags.c_contiguous:
        raise TypeError("limb buffers must be contiguous one-dimensional uint32 arrays")
    if n is not None:
        if n < 0 or n > words.size:
            raise RuntimeError(f"Mojo kernel returned invalid limb count {n}")
        words = words[:n]
    return int.from_bytes(words.astype("<u4", copy=False).tobytes(), "little")


def add_abs(a: int, b: int) -> int:
    aa, bb = limbs(a), limbs(b)
    dst = np.zeros(max(aa.size, bb.size) + 1, dtype=np.uint32)
    n = lib().mg_add(addr(aa), aa.size, addr(bb), bb.size, addr(dst))
    return from_limbs(dst, n)


def sub_abs(a: int, b: int) -> int:
    aa, bb = limbs(a), limbs(b)
    dst = np.zeros(aa.size, dtype=np.uint32)
    n = lib().mg_sub(addr(aa), aa.size, addr(bb), bb.size, addr(dst))
    return from_limbs(dst, n)


def mul_abs(a: int, b: int) -> int:
    if not a or not b:
        return 0
    aa, bb = limbs(a), limbs(b)
    dst = np.zeros(aa.size + bb.size, dtype=np.uint32)
    n = lib().mg_mul(addr(aa), aa.size, addr(bb), bb.size, addr(dst))
    return from_limbs(dst, n)


def gcd_abs(a: int, b: int) -> int:
    aa, bb = limbs(a), limbs(b)
    capacity = max(aa.size, bb.size) + 1
    scratch = np.zeros((3, capacity), dtype=np.uint32)
    u, v, dst = scratch
    n = lib().mg_gcd(
        addr(aa), aa.size, addr(bb), bb.size, addr(u), addr(v), addr(dst)
    )
    return from_limbs(dst, n)


def powmod_abs(base: int, exponent: int, modulus: int) -> int:
    n = max(1, (modulus.bit_length() + 31) // 32)
    aa = limbs(base % modulus, n)
    ee = limbs(exponent)
    mm = limbs(modulus, n)
    buffers = np.zeros((5, n), dtype=np.uint32)
    result, power, product, addend, tmp = buffers
    used = lib().mg_powmod(
        addr(aa),
        addr(ee),
        ee.size,
        addr(mm),
        n,
        addr(result),
        addr(power),
        addr(product),
        addr(addend),
        addr(tmp),
    )
    return from_limbs(result, used)
