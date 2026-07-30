"""Honest same-process benchmarks against gmpy2/GMP."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import gmpy2 as reference  # noqa: E402
import mojo_gmpy2 as mojo  # noqa: E402


def best_time(fn, repeat=5):
    best = math.inf
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return best, result


def benchmark(name, ours, theirs, operations):
    ours()
    theirs()
    mojo_time, mojo_result = best_time(ours)
    ref_time, ref_result = best_time(theirs)
    if mojo_result != ref_result:
        raise AssertionError(f"{name}: benchmark implementations disagree")
    return name, mojo_time, ref_time, operations


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    a = (1 << 4096) - 0xD6E8FEB86659FD93
    b = (1 << 4087) + 0xA5A3564E27F8862D
    ma, mb = mojo.mpz(a), mojo.mpz(b)
    ra, rb = reference.mpz(a), reference.mpz(b)

    common = (1 << 600) - 93
    ga = ((1 << 2048) - 159) * common
    gb = ((1 << 2039) + 27) * common
    mga, mgb = mojo.mpz(ga), mojo.mpz(gb)
    rga, rgb = reference.mpz(ga), reference.mpz(gb)

    modulus = (1 << 255) - 19
    exponent = (1 << 127) + 0x123456789ABCDEF
    base = (1 << 253) + 0xDEADBEEF

    q_terms = 250

    rows = [
        benchmark(
            "mpz multiply, 4096-bit",
            lambda: [ma * mb for _ in range(200)][-1],
            lambda: [ra * rb for _ in range(200)][-1],
            200,
        ),
        benchmark(
            "gcd, ~2600-bit",
            lambda: [mojo.gcd(mga, mgb) for _ in range(30)][-1],
            lambda: [reference.gcd(rga, rgb) for _ in range(30)][-1],
            30,
        ),
        benchmark(
            "powmod, 255-bit modulus",
            lambda: mojo.powmod(base, exponent, modulus),
            lambda: reference.powmod(base, exponent, modulus),
            1,
        ),
        benchmark(
            "mpq harmonic sum, 250 terms",
            lambda: sum((mojo.mpq(1, i) for i in range(1, q_terms + 1)), mojo.mpq()),
            lambda: sum(
                (reference.mpq(1, i) for i in range(1, q_terms + 1)), reference.mpq()
            ),
            q_terms,
        ),
    ]

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print(
        f"| operation | mojo-gmpy2 | gmpy2 {reference.version()} / GMP | relative |"
    )
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs, operations in rows:
        ours_us = ours * 1e6 / operations
        theirs_us = theirs * 1e6 / operations
        ratio = theirs / ours
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.1f}x slower"
        print(
            f"| {name} | {ours_us:.2f} us/op | {theirs_us:.2f} us/op | {label} |"
        )


if __name__ == "__main__":
    main()
