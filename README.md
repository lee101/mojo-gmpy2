# mojo-gmpy2

Multiprecision integer and rational arithmetic with a compiled Mojo core and a
Python API shaped like [`gmpy2`](https://gmpy2.readthedocs.io/).

The package is named `mojo_gmpy2` so it can be installed beside the real
`gmpy2` used by the parity suite. For the covered subset, use the familiar name:

```python
import mojo_gmpy2 as gmpy2

a = gmpy2.mpz("ffffffffffffffffffffffffffffffff", 16)
b = gmpy2.next_prime(10**30)
product = a * b
power = gmpy2.powmod(a, 65537, b)
ratio = gmpy2.mpq(product, b)

assert ratio == a
assert 0 <= power < b
print(product)
```

This is a real arbitrary-precision implementation: values are not restricted to
64 or 128 bits. It is currently a correctness and interoperability port, not a
replacement for GMP's decades of assembly optimization. As the measured results
below show, upstream `gmpy2` is much faster.

## Coverage

The covered integer surface includes:

- `mpz` construction, representation, comparison, arithmetic, floor division,
  remainder, `divmod`, powers, shifts, and bitwise operators
- `gcd`, `lcm`, `gcdext`, `invert`, `powmod`, `divexact`, and `remove`
- ceiling, floor, and truncating division through `c_div`/`c_mod`,
  `f_div`/`f_mod`, and `t_div`/`t_mod`
- divisibility and congruence checks; bit length, count, masks, scans, tests,
  setting, clearing, flipping, and Hamming distance
- `isqrt`, `isqrt_rem`, `iroot`, `iroot_rem`, and `is_square`
- `fac`, `double_fac`, `multi_fac`, `bincoef`, `fib`, `fib2`,
  `lucas`, `lucas2`, and `primorial`
- probable-prime testing and `next_prime`/`prev_prime`
- base-2 through base-62 `digits` and `num_digits`

The rational surface includes canonical `mpq` construction from integers,
strings, and rational values; arithmetic with integer and rational operands;
comparison; integer powers; numerator and denominator properties; `qdiv`,
`numer`, and `denom`. Results normalize their sign and cancel common factors
like upstream.

The Mojo compilation unit implements unsigned limb addition, subtraction,
schoolbook multiplication, binary GCD, and modular exponentiation. The public
Python layer dispatches addition and subtraction through those kernels and uses
CPython's exact integer operations for multiplication, GCD, and modular power.
Exact division, roots, combinatorial helpers, probable-prime testing, and
rational normalization also use Python integer algorithms or
`fractions.Fraction`.

Not covered are `mpfr`, `mpc`, true division of `mpz`, floating-point or complex
operands, contexts, GMP random-state APIs, formatted conversion beyond
`digits`, serialization APIs, generalized number-theory functions such as
Jacobi/Kronecker symbols, and the rest of the upstream module. Negative moduli
are outside the covered `powmod` contract. Primality uses Miller-Rabin rather
than GMP's exact probable-prime implementation, so adversarial pseudoprimes
above the unsigned 64-bit range can classify differently. Importing this module
as `gmpy2` is drop-in only for the subset listed above.

## Install and run

The supported installation is a source checkout managed by pixi:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` produces `dist/libmojo-gmpy2.so`. The Python loader also
rebuilds a missing or stale library on first use when `mojo` or `pixi` is
available. `gmpy2` itself is a pixi dependency only so tests and benchmarks can
compare against the real upstream package. A standalone binary wheel is not
provided.

To run the example without installing a wheel:

```bash
pixi run python -c \
  'import mojo_gmpy2 as g; print(g.gcd(g.mpz(2)**521-1, g.mpz(2)**127-1))'
```

## Benchmarks

Measured on 2026-07-30 with an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64.
Each row is the best of five same-process runs from `pixi run bench`; the
multiplication and GCD rows execute batches and report per-operation time.

| operation | mojo-gmpy2 | gmpy2 2.3.0 / GMP | relative |
| --- | ---: | ---: | ---: |
| `mpz` multiply, 4096-bit | 15.00 us/op | 2.68 us/op | 5.6x slower |
| GCD, approximately 2600-bit | 7.47 us/op | 1.26 us/op | 5.9x slower |
| `powmod`, 255-bit modulus | 86.46 us/op | 7.24 us/op | 11.9x slower |
| `mpq` harmonic sum, 250 terms | 2.06 us/term | 0.53 us/term | 3.9x slower |

Upstream still wins because it calls GMP's tuned low-level algorithms. The Mojo
limb copy and clear helpers use host-width SIMD with scalar remainder loops.
Read-only input limbs are zero-copy NumPy views. Rational addition constructs
already-normalized `mpq` results directly instead of wrapping and re-reading an
intermediate `Fraction`.

No multithreaded CPU or GPU path is implemented.

## How it works

Operations dispatched to Mojo cross the boundary as contiguous little-endian
`uint32` limb arrays. A negative value is represented by a separate sign in the
Python layer; the Mojo kernels only see magnitudes. Result arrays are allocated
by the caller, so the shared library owns no Python memory and performs no
cross-runtime allocation.

ctypes keeps the NumPy arrays alive for each synchronous call and passes each
array as a pointer plus its limb count. The bridge accepts only contiguous,
one-dimensional native `uint32` arrays and checks kernel result lengths.
`src/gmpy2.mojo` reconstructs
`UnsafePointer[UInt32, AnyOrigin[mut=True]]` inside non-parametric C ABI exports.
The exports reject null pointers and nonpositive lengths before constructing a
Mojo pointer. The whole kernel set is one compilation unit, producing one
shared library.
The input arrays are views over the immutable bytes produced by Python integer
conversion and are passed to ctypes without a second copy. Multiplication
accumulates a limb product in `UInt64`; binary GCD works in caller-provided
scratch buffers; modular exponentiation uses repeated overflow-safe modular
addition, so the retained low-level kernels remain correct without a
wider-than-64-bit scalar type.

The test suite compares values, result types where part of the contract,
normalization, and exception behavior directly with the installed upstream
`gmpy2 2.3.0`. It contains randomized inputs through 4096 bits as well as
boundary sizes around the 32-bit limb representation.

MIT licensed.
