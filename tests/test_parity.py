"""Numerical and behavioral parity with the installed gmpy2 package."""

import math
import random

import gmpy2 as reference
import pytest

import mojo_gmpy2 as mojo
from mojo_gmpy2 import _lib


@pytest.mark.parametrize("bits", [1, 31, 32, 33, 64, 127, 256, 521, 1024, 4096])
def test_mpz_arithmetic(bits):
    rng = random.Random(bits)
    for _ in range(8):
        a = rng.getrandbits(bits)
        b = rng.getrandbits(bits)
        if rng.randrange(2):
            a = -a
        if rng.randrange(2):
            b = -b
        ours_a, ours_b = mojo.mpz(a), mojo.mpz(b)
        ref_a, ref_b = reference.mpz(a), reference.mpz(b)
        assert ours_a + ours_b == ref_a + ref_b
        assert ours_a - ours_b == ref_a - ref_b
        assert ours_a * ours_b == ref_a * ref_b
        if b:
            assert ours_a // ours_b == ref_a // ref_b
            assert ours_a % ours_b == ref_a % ref_b
            assert divmod(ours_a, ours_b) == divmod(ref_a, ref_b)


def test_mpz_construction_repr_and_bits():
    assert mojo.mpz("ff", 16) == reference.mpz("ff", 16)
    assert mojo.mpz("-0b101", 0) == reference.mpz("-0b101", 0)
    assert repr(mojo.mpz(123)) == repr(reference.mpz(123))
    value = -(1 << 300) + 12345
    assert mojo.mpz(value) << 17 == reference.mpz(value) << 17
    assert mojo.mpz(value) >> 17 == reference.mpz(value) >> 17
    assert ~mojo.mpz(value) == ~reference.mpz(value)
    assert (mojo.mpz(value) & 0xFFFF) == (reference.mpz(value) & 0xFFFF)
    assert mojo.mpz(7) ** 33 == reference.mpz(7) ** 33
    assert 3 ** mojo.mpz(17) == 3 ** reference.mpz(17)
    assert mojo.mpz(3) | 8 == reference.mpz(3) | 8
    assert mojo.mpz(3) ^ 10 == reference.mpz(3) ^ 10


@pytest.mark.parametrize("bits", [95, 96, 97, 127, 128, 129])
def test_limb_simd_tail_paths(bits):
    a = (1 << bits) - 0x12345
    b = (1 << (bits - 2)) + 0xABCDE
    assert mojo.mpz(a) + mojo.mpz(b) == reference.mpz(a) + reference.mpz(b)
    assert mojo.mpz(a) - mojo.mpz(b) == reference.mpz(a) - reference.mpz(b)
    assert _lib.mul_abs(a, b) == a * b
    assert _lib.gcd_abs(a, b) == math.gcd(a, b)
    modulus = (1 << bits) - 159
    assert _lib.powmod_abs(a % modulus, 37, modulus) == pow(a, 37, modulus)


@pytest.mark.parametrize(
    "a,b",
    [
        (0, 0),
        (0, 42),
        (-24, 18),
        ((1 << 521) - 1, (1 << 127) - 1),
        (math.factorial(300), math.factorial(220)),
    ],
)
def test_gcd_lcm_and_gcdext(a, b):
    assert mojo.gcd(a, b) == reference.gcd(a, b)
    assert mojo.lcm(a, b) == reference.lcm(a, b)
    g, s, t = mojo.gcdext(a, b)
    rg, _, _ = reference.gcdext(a, b)
    assert g == rg
    assert a * s + b * t == g


@pytest.mark.parametrize("bits", [7, 31, 64, 127, 256])
def test_powmod(bits):
    rng = random.Random(1000 + bits)
    for _ in range(5):
        modulus = rng.getrandbits(bits) | 1
        base = rng.randrange(-modulus * 3, modulus * 3)
        exponent = rng.randrange(0, 300)
        assert mojo.powmod(base, exponent, modulus) == reference.powmod(
            base, exponent, modulus
        )
    assert mojo.powmod(2, 0, 1) == reference.powmod(2, 0, 1)


def test_inverse_and_negative_powmod():
    for x, modulus in [(3, 11), (-7, 19), (65537, (1 << 127) - 1)]:
        assert mojo.invert(x, modulus) == reference.invert(x, modulus)
        assert mojo.powmod(x, -3, modulus) == reference.powmod(x, -3, modulus)
    with pytest.raises(ZeroDivisionError):
        mojo.invert(2, 4)
    assert mojo.divexact(3**40, 3**10) == reference.divexact(3**40, 3**10)


@pytest.mark.parametrize(
    "left,right",
    [
        ((1, 2), (2, 3)),
        ((-7, 15), (11, -9)),
        (((1 << 300) - 1, (1 << 127) + 1), (17, 19)),
    ],
)
def test_mpq_arithmetic(left, right):
    a, b = mojo.mpq(*left), mojo.mpq(*right)
    ra, rb = reference.mpq(*left), reference.mpq(*right)
    assert str(a) == str(ra)
    assert repr(a) == repr(ra)
    assert a.numerator == ra.numerator
    assert a.denominator == ra.denominator
    assert a + b == ra + rb
    assert a - b == ra - rb
    assert a * b == ra * rb
    assert a / b == ra / rb
    assert a**7 == ra**7
    assert (a < b) == (ra < rb)


def test_mpq_string_construction():
    assert mojo.mpq("123/456") == reference.mpq("123/456")
    assert mojo.mpq("ff/10", base=16) == reference.mpq("ff/10", base=16)


def test_rational_helpers():
    q = mojo.qdiv((1 << 200) + 1, 15)
    rq = reference.qdiv((1 << 200) + 1, 15)
    assert q == rq
    assert mojo.numer(q) == reference.numer(rq)
    assert mojo.denom(q) == reference.denom(rq)
    assert math.floor(q) == math.floor(rq)
    assert math.ceil(q) == math.ceil(rq)


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4, 15, 16, 17, (1 << 255) - 19])
def test_roots_and_square(value):
    assert mojo.is_square(value) == reference.is_square(value)
    assert mojo.isqrt(value) == reference.isqrt(value)
    assert mojo.isqrt_rem(value) == reference.isqrt_rem(value)
    for degree in (1, 2, 3, 5, 17):
        assert mojo.iroot(value, degree) == reference.iroot(value, degree)
        assert mojo.iroot_rem(value, degree) == reference.iroot_rem(value, degree)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 20, 100, 1000])
def test_sequences(n):
    assert mojo.fib(n) == reference.fib(n)
    assert mojo.fib2(n) == reference.fib2(n)
    assert mojo.lucas(n) == reference.lucas(n)
    assert mojo.lucas2(n) == reference.lucas2(n)


def test_combinatorics():
    for n in (0, 1, 2, 10, 100, 500):
        assert mojo.fac(n) == reference.fac(n)
        assert mojo.double_fac(n) == reference.double_fac(n)
    for n, k in [(10, 3), (100, 50), (-20, 7), (5, 8)]:
        assert mojo.bincoef(n, k) == reference.bincoef(n, k)
    assert mojo.multi_fac(100, 7) == reference.multi_fac(100, 7)
    assert mojo.primorial(100) == reference.primorial(100)


def test_divisibility_and_bits():
    values = [0, 1, -1, 2, -8, (1 << 200) + 37]
    for value in values:
        assert mojo.is_even(value) == reference.is_even(value)
        assert mojo.is_odd(value) == reference.is_odd(value)
        assert mojo.bit_length(value) == reference.bit_length(value)
        assert mojo.bit_count(value) == reference.bit_count(value)
        for bit in (0, 1, 17, 201):
            assert mojo.bit_test(value, bit) == reference.bit_test(value, bit)
            assert mojo.bit_set(value, bit) == reference.bit_set(value, bit)
            assert mojo.bit_clear(value, bit) == reference.bit_clear(value, bit)
            assert mojo.bit_flip(value, bit) == reference.bit_flip(value, bit)
    for x, y in [(0, 0), (1, 0), (12, 3), (-15, 5), (17, 4)]:
        assert mojo.is_divisible(x, y) == reference.is_divisible(x, y)
    for x, y, modulus in [(1, 6, 5), (1, 2, 0), (-10, 5, 3)]:
        assert mojo.is_congruent(x, y, modulus) == reference.is_congruent(
            x, y, modulus
        )
    for bits in (0, 1, 17, 200):
        assert mojo.bit_mask(bits) == reference.bit_mask(bits)


def test_digits_division_modes_and_remove():
    value = -((1 << 250) + 123456789)
    for base in (2, 10, 16, 36, 37, 62, -2, -10, -16, -36):
        assert mojo.digits(value, base) == reference.digits(value, base)
        assert mojo.mpz(value).digits(base) == reference.mpz(value).digits(base)
        if base > 0:
            assert mojo.num_digits(value, base) == reference.num_digits(value, base)
    for x, y in [(-7, 3), (7, -3), (-7, -3), (100, 9)]:
        for name in ("c_div", "c_mod", "f_div", "f_mod", "t_div", "t_mod"):
            assert getattr(mojo, name)(x, y) == getattr(reference, name)(x, y)
    for x, factor in [(72, 3), (2**100 * 17, 2), (-5**20, 5), (7, 3)]:
        assert mojo.remove(x, factor) == reference.remove(x, factor)
    with pytest.raises(ValueError):
        mojo.remove(72, -3)


def test_bit_scans_and_hamming_distance():
    for value in [0, 1, 2, 0x10100, (1 << 300) + 3, -1, -256]:
        for start in [0, 1, 8, 301]:
            assert mojo.bit_scan0(value, start) == reference.bit_scan0(value, start)
            assert mojo.bit_scan1(value, start) == reference.bit_scan1(value, start)
    for a, b in [(0, 0), (1, 2), (0xFFFF, 0xAAAA), (1 << 500, 3)]:
        assert mojo.hamdist(a, b) == reference.hamdist(a, b)


@pytest.mark.parametrize(
    "function,args",
    [
        ("bit_test", (1, -1)),
        ("bit_set", (1, -1)),
        ("bit_clear", (1, -1)),
        ("bit_flip", (1, -1)),
        ("bit_scan0", (1, -1)),
        ("bit_scan1", (1, -1)),
        ("bit_mask", (-1,)),
    ],
)
def test_negative_bit_indices_match_exception_type(function, args):
    with pytest.raises(OverflowError):
        getattr(reference, function)(*args)
    with pytest.raises(OverflowError):
        getattr(mojo, function)(*args)


def test_ffi_rejects_invalid_buffers_and_addresses():
    import numpy as np

    with pytest.raises(TypeError):
        _lib.addr(np.zeros(4, dtype=np.uint64))
    with pytest.raises(TypeError):
        _lib.addr(np.zeros((2, 2), dtype=np.uint32))
    with pytest.raises(ValueError):
        _lib.addr(np.zeros(0, dtype=np.uint32))
    assert _lib.lib().mg_add(None, 1, None, 1, None) == -1
    assert _lib.lib().mg_add(None, -1, None, 1, None) == -1


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        97,
        561,
        1105,
        2_147_483_647,
        18_446_744_073_709_551_557,
    ],
)
def test_primality(value):
    assert mojo.is_prime(value) == reference.is_prime(value)


def test_primality_repetitions_and_aliases():
    assert mojo.is_prime((1 << 127) - 1, 40) == reference.is_prime(
        (1 << 127) - 1, 40
    )
    with pytest.raises(OverflowError):
        mojo.is_prime(17, -1)
    assert mojo.fac(30) == reference.fac(30)
    assert mojo.multi_fac(30, 4) == reference.multi_fac(30, 4)


def test_neighboring_primes():
    for value in [2, 3, 100, 10_000, 1 << 64]:
        assert mojo.next_prime(value) == reference.next_prime(value)
    for value in [3, 4, 100, 10_000, (1 << 64) + 100]:
        assert mojo.prev_prime(value) == reference.prev_prime(value)
