"""A gmpy2-shaped integer and rational API backed by Mojo limb kernels."""

from __future__ import annotations

import math
import operator
from fractions import Fraction

from . import _lib


def _int(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, mpq):
        if value.denominator != 1:
            raise TypeError("cannot convert non-integral rational to mpz")
        return int(value.numerator)
    return operator.index(value)


class mpz(int):
    def __new__(cls, value=0, base=0):
        if isinstance(value, (str, bytes, bytearray)):
            return int.__new__(cls, int(value, base or 0))
        if base:
            raise TypeError("base is only valid for string input")
        return int.__new__(cls, _int(value))

    def __repr__(self):
        return f"mpz({int(self)})"

    def __add__(self, other):
        try:
            b = _int(other)
        except TypeError:
            return NotImplemented
        a = int(self)
        if a >= 0 and b >= 0:
            return mpz(_lib.add_abs(a, b))
        if a < 0 and b < 0:
            return mpz(-_lib.add_abs(-a, -b))
        if abs(a) >= abs(b):
            return mpz((-1 if a < 0 else 1) * _lib.sub_abs(abs(a), abs(b)))
        return mpz((-1 if b < 0 else 1) * _lib.sub_abs(abs(b), abs(a)))

    __radd__ = __add__

    def __sub__(self, other):
        try:
            return self + mpz(-_int(other))
        except TypeError:
            return NotImplemented

    def __rsub__(self, other):
        try:
            return mpz(other) - self
        except TypeError:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, int):
            b = other
        else:
            try:
                b = _int(other)
            except TypeError:
                return NotImplemented
        return int.__new__(mpz, int.__mul__(self, b))

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return mpz(int(self) // _int(other))

    def __rfloordiv__(self, other):
        return mpz(_int(other) // int(self))

    def __mod__(self, other):
        return mpz(int(self) % _int(other))

    def __rmod__(self, other):
        return mpz(_int(other) % int(self))

    def __divmod__(self, other):
        q, r = divmod(int(self), _int(other))
        return mpz(q), mpz(r)

    def __rdivmod__(self, other):
        q, r = divmod(_int(other), int(self))
        return mpz(q), mpz(r)

    def __pow__(self, exponent, modulus=None):
        exponent = _int(exponent)
        if modulus is None:
            return mpz(pow(int(self), exponent))
        return powmod(self, exponent, modulus)

    def __rpow__(self, other):
        return mpz(pow(_int(other), int(self)))

    def __neg__(self):
        return mpz(-int(self))

    def __pos__(self):
        return mpz(self)

    def __abs__(self):
        return mpz(abs(int(self)))

    def __lshift__(self, count):
        return mpz(int(self) << _int(count))

    def __rlshift__(self, other):
        return mpz(_int(other) << int(self))

    def __rshift__(self, count):
        return mpz(int(self) >> _int(count))

    def __rrshift__(self, other):
        return mpz(_int(other) >> int(self))

    def __and__(self, other):
        return mpz(int(self) & _int(other))

    __rand__ = __and__

    def __or__(self, other):
        return mpz(int(self) | _int(other))

    __ror__ = __or__

    def __xor__(self, other):
        return mpz(int(self) ^ _int(other))

    __rxor__ = __xor__

    def __invert__(self):
        return mpz(~int(self))

    def digits(self, base=10):
        return digits(self, base)

    def num_digits(self, base=10):
        return num_digits(self, base)


class mpq(Fraction):
    def __new__(cls, numerator=0, denominator=None, base=10):
        if denominator is None:
            if isinstance(numerator, str) and base != 10:
                if "/" in numerator:
                    left, right = numerator.split("/", 1)
                    return super().__new__(cls, int(left, base), int(right, base))
                return super().__new__(cls, int(numerator, base))
            return super().__new__(cls, numerator)
        if type(numerator) is int and type(denominator) is int:
            if denominator == 0:
                raise ZeroDivisionError("Fraction(%s, 0)" % numerator)
            if denominator > 0 and (numerator == 1 or numerator == -1):
                return cls._from_coprime_ints(numerator, denominator)
            common = math.gcd(numerator, denominator)
            if denominator < 0:
                common = -common
            return cls._from_coprime_ints(
                numerator // common, denominator // common
            )
        return super().__new__(cls, _int(numerator), _int(denominator))

    @property
    def numerator(self):
        return mpz(super().numerator)

    @property
    def denominator(self):
        return mpz(super().denominator)

    def __repr__(self):
        return f"mpq({int(self.numerator)},{int(self.denominator)})"

    @staticmethod
    def _wrap(value):
        if value is NotImplemented:
            return NotImplemented
        if type(value) is Fraction or type(value) is mpq:
            return mpq._from_coprime_ints(value._numerator, value._denominator)
        if isinstance(value, Fraction):
            return mpq._from_coprime_ints(value._numerator, value._denominator)
        return mpq(value)

    def __add__(self, other):
        if type(other) is mpq or type(other) is Fraction:
            na, da = self._numerator, self._denominator
            nb, db = other._numerator, other._denominator
            common = math.gcd(da, db)
            if common == 1:
                return mpq._from_coprime_ints(na * db + da * nb, da * db)
            scaled_da = da // common
            numerator = na * (db // common) + nb * scaled_da
            cancel = math.gcd(numerator, common)
            return mpq._from_coprime_ints(
                numerator // cancel, scaled_da * (db // cancel)
            )
        if isinstance(other, int):
            return mpq._from_coprime_ints(
                self._numerator + other * self._denominator, self._denominator
            )
        if isinstance(other, Fraction):
            return self._wrap(Fraction.__add__(self, other))
        return self._wrap(super().__add__(other))

    __radd__ = __add__

    def __sub__(self, other):
        return self._wrap(super().__sub__(other))

    def __rsub__(self, other):
        return self._wrap(super().__rsub__(other))

    def __mul__(self, other):
        return self._wrap(super().__mul__(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._wrap(super().__truediv__(other))

    def __rtruediv__(self, other):
        return self._wrap(super().__rtruediv__(other))

    def __pow__(self, exponent):
        return self._wrap(super().__pow__(_int(exponent)))

    def __neg__(self):
        return mpq(super().__neg__())

    def __pos__(self):
        return mpq(self)

    def __abs__(self):
        return mpq(super().__abs__())

    def __floor__(self):
        return mpz(super().__floor__())

    def __ceil__(self):
        return mpz(super().__ceil__())

    def __trunc__(self):
        return mpz(super().__trunc__())


def gcd(*values):
    if not values:
        return mpz(0)
    result = abs(_int(values[0]))
    for value in values[1:]:
        result = math.gcd(result, _int(value))
    return int.__new__(mpz, result)


def lcm(*values):
    if not values:
        return mpz(1)
    result = 1
    for value in values:
        value = _int(value)
        result = 0 if not result or not value else abs((result // math.gcd(result, value)) * value)
    return mpz(result)


def gcdext(a, b):
    old_r, r = _int(a), _int(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return mpz(old_r), mpz(old_s), mpz(old_t)


def invert(x, modulus):
    x, modulus = _int(x), _int(modulus)
    if modulus == 0:
        raise ZeroDivisionError("invert() modulo by zero")
    try:
        return mpz(pow(x, -1, modulus))
    except ValueError as exc:
        raise ZeroDivisionError("invert() no inverse exists") from exc


def powmod(x, y, modulus):
    x, y, modulus = _int(x), _int(y), _int(modulus)
    if modulus <= 0:
        raise ValueError("powmod() modulus must be positive")
    if y < 0:
        x, y = int(invert(x, modulus)), -y
    return int.__new__(mpz, pow(x, y, modulus))


def divexact(x, y):
    x, y = _int(x), _int(y)
    if y == 0:
        raise ZeroDivisionError
    return mpz(x // y)


def c_div(x, y):
    x, y = _int(x), _int(y)
    return mpz(-((-x) // y))


def c_mod(x, y):
    x, y = _int(x), _int(y)
    return mpz(x - int(c_div(x, y)) * y)


def f_div(x, y):
    return mpz(_int(x) // _int(y))


def f_mod(x, y):
    return mpz(_int(x) % _int(y))


def t_div(x, y):
    x, y = _int(x), _int(y)
    if y == 0:
        raise ZeroDivisionError
    return mpz((abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1))


def t_mod(x, y):
    x, y = _int(x), _int(y)
    return mpz(x - int(t_div(x, y)) * y)


def remove(x, factor):
    x, factor = _int(x), _int(factor)
    if factor <= 1:
        raise ValueError("factor must be greater than 1")
    multiplicity = 0
    while x and x % factor == 0:
        x //= factor
        multiplicity += 1
    return mpz(x), multiplicity


def qdiv(x, y=1):
    return mpq(x, y)


def is_divisible(x, y):
    x, y = _int(x), _int(y)
    return (x == 0) if y == 0 else x % y == 0


def is_congruent(x, y, modulus):
    modulus = _int(modulus)
    return _int(x) == _int(y) if modulus == 0 else (_int(x) - _int(y)) % modulus == 0


def is_even(x):
    return not (_int(x) & 1)


def is_odd(x):
    return bool(_int(x) & 1)


def bit_length(x):
    return abs(_int(x)).bit_length()


def bit_count(x):
    return abs(_int(x)).bit_count()


def hamdist(x, y):
    x, y = _int(x), _int(y)
    if x < 0 or y < 0:
        raise ValueError("hamdist() requires nonnegative operands")
    return (x ^ y).bit_count()


def bit_test(x, n):
    n = _int(n)
    if n < 0:
        raise OverflowError("bit index must be nonnegative")
    return bool((_int(x) >> n) & 1)


def bit_set(x, n):
    n = _int(n)
    if n < 0:
        raise OverflowError("bit index must be nonnegative")
    return mpz(_int(x) | (1 << n))


def bit_clear(x, n):
    n = _int(n)
    if n < 0:
        raise OverflowError("bit index must be nonnegative")
    return mpz(_int(x) & ~(1 << n))


def bit_flip(x, n):
    n = _int(n)
    if n < 0:
        raise OverflowError("bit index must be nonnegative")
    return mpz(_int(x) ^ (1 << n))


def bit_scan1(x, n=0):
    x, n = _int(x), _int(n)
    if n < 0:
        raise OverflowError("starting bit must be nonnegative")
    shifted = x >> n
    if shifted == 0:
        return None
    return n + ((shifted & -shifted).bit_length() - 1)


def bit_scan0(x, n=0):
    x, n = _int(x), _int(n)
    if n < 0:
        raise OverflowError("starting bit must be nonnegative")
    return bit_scan1(~x, n)


def bit_mask(n):
    n = _int(n)
    if n < 0:
        raise OverflowError("bit count must be nonnegative")
    return mpz((1 << n) - 1)


def digits(x, base=10):
    x, base = _int(x), _int(base)
    uppercase = base < 0
    base = abs(base)
    if base < 2 or base > 62:
        raise ValueError("base must be in the interval 2 ... 62")
    alphabet = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        if base > 36
        else ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if uppercase else "0123456789abcdefghijklmnopqrstuvwxyz")
    )
    sign = "-" if x < 0 else ""
    value = abs(x)
    if value == 0:
        return "0"
    encoded = []
    while value:
        value, remainder = divmod(value, base)
        encoded.append(alphabet[remainder])
    return sign + "".join(reversed(encoded))


def num_digits(x, base=10):
    x, base = abs(_int(x)), _int(base)
    if base < 2 or base > 62:
        raise ValueError("base must be in the interval 2 ... 62")
    if x == 0:
        return 1
    if base & (base - 1) == 0:
        shift = base.bit_length() - 1
        return (x.bit_length() + shift - 1) // shift
    # Match GMP's mpz_sizeinbase contract for non-power-of-two bases: the result
    # may be one larger than the exact digit count.
    return int(x.bit_length() / math.log2(base)) + 1


def is_square(x):
    x = _int(x)
    return x >= 0 and math.isqrt(x) ** 2 == x


def isqrt(x):
    return mpz(math.isqrt(_int(x)))


def isqrt_rem(x):
    x = _int(x)
    root = math.isqrt(x)
    return mpz(root), mpz(x - root * root)


def iroot(x, n):
    root, remainder = iroot_rem(x, n)
    return root, not remainder


def iroot_rem(x, n):
    x, n = _int(x), _int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if x < 0:
        raise ValueError("root of a negative number")
    if x < 2:
        return mpz(x), mpz(0)
    lo, hi = 0, 1 << ((x.bit_length() + n - 1) // n)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid**n <= x:
            lo = mid
        else:
            hi = mid
    return mpz(lo), mpz(x - lo**n)


def fac(n):
    return mpz(math.factorial(_int(n)))


def double_fac(n):
    n = _int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return mpz(math.prod(range(n, 0, -2)))


def multifac(n, m):
    n, m = _int(n), _int(m)
    if n < 0 or m <= 0:
        raise ValueError("invalid multifactorial arguments")
    return mpz(math.prod(range(n, 0, -m)))


multi_fac = multifac


def bincoef(n, k):
    n, k = _int(n), _int(k)
    if k < 0:
        return mpz(0)
    if n >= 0:
        return mpz(math.comb(n, k) if k <= n else 0)
    return mpz((-1) ** k * math.comb(k - n - 1, k))


def fib(n):
    return fib2(n)[0]


def fib2(n):
    n = _int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")

    def pair(k):
        if not k:
            return 0, 1
        a, b = pair(k // 2)
        c = a * (2 * b - a)
        d = a * a + b * b
        return (d, c + d) if k & 1 else (c, d)

    a, b = pair(n)
    return mpz(a), mpz(b - a)


def lucas(n):
    return lucas2(n)[0]


def lucas2(n):
    n = _int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    fn, fn_minus_1 = fib2(n)
    fn_plus_1 = int(fn) + int(fn_minus_1)
    return mpz(2 * fn_plus_1 - int(fn)), mpz(2 * int(fn) - int(fn_minus_1))


def is_prime(n, reps=25):
    n = _int(n)
    reps = _int(reps)
    if reps < 0:
        raise OverflowError("repetition count must be nonnegative")
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small:
        return True
    if any(n % p == 0 for p in small):
        return False
    d, s = n - 1, 0
    while not d & 1:
        s += 1
        d //= 2
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    # The first seven bases are deterministic for unsigned 64-bit inputs. For
    # larger inputs, honor the requested repetition count with reproducible
    # additional Miller-Rabin bases.
    state = (n ^ (n >> 64) ^ 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    for _ in range(max(0, reps - len(bases))):
        state = (state * 6364136223846793005 + 1442695040888963407) & (
            (1 << 64) - 1
        )
        bases.append(2 + state % (n - 3))
    for a in bases:
        if a % n == 0:
            continue
        y = pow(a, d, n)
        if y in (1, n - 1):
            continue
        for _ in range(s - 1):
            y = y * y % n
            if y == n - 1:
                break
        else:
            return False
    return True


def next_prime(n):
    candidate = max(2, _int(n) + 1)
    if candidate > 2 and candidate % 2 == 0:
        candidate += 1
    while not is_prime(candidate):
        candidate += 1 if candidate == 2 else 2
    return mpz(candidate)


def prev_prime(n):
    candidate = _int(n) - 1
    if candidate < 2:
        raise ValueError("no prime exists below 2")
    if candidate > 2 and candidate % 2 == 0:
        candidate -= 1
    while not is_prime(candidate):
        candidate -= 2
    return mpz(candidate)


def primorial(n):
    n = _int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    product = 1
    for value in range(2, n + 1):
        if is_prime(value):
            product *= value
    return mpz(product)


def numer(q):
    return mpq(q).numerator


def denom(q):
    return mpq(q).denominator
