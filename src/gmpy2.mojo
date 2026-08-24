"""Unsigned base-2**32 kernels behind the Python arbitrary-precision API."""

from std.sys.info import simd_width_of

comptime LimbPtr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime BASE: UInt64 = UInt64(1) << UInt64(32)
comptime MASK: UInt64 = UInt64(0xFFFFFFFF)


def limbs(addr: Int) -> LimbPtr:
    return LimbPtr(unsafe_from_address=addr)


def normalized(a: LimbPtr, n_in: Int) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var n = n_in
    while n > 0 and n % W != 0:
        if a[n - 1] != 0:
            return n
        n -= 1
    var zeros = SIMD[DType.uint32, W](0)
    while n >= W:
        var block_start = n - W
        var nonzero = a.load[width=W](block_start).ne(zeros)
        if nonzero.cast[DType.int64]().reduce_add() != 0:
            while n > block_start and a[n - 1] == 0:
                n -= 1
            return n
        n = block_start
    return 0


def copy_limbs(src: LimbPtr, dst: LimbPtr, n: Int):
    comptime W = simd_width_of[DType.float64]()
    var vector_end = n - n % W
    for i in range(0, vector_end, W):
        dst.store(i, src.load[width=W](i))
    for i in range(vector_end, n):
        dst[i] = src[i]


def zero_limbs(dst: LimbPtr, n: Int):
    comptime W = simd_width_of[DType.float64]()
    var vector_end = n - n % W
    var zeros = SIMD[DType.uint32, W](0)
    for i in range(0, vector_end, W):
        dst.store(i, zeros)
    for i in range(vector_end, n):
        dst[i] = 0


def compare(a: LimbPtr, na_in: Int, b: LimbPtr, nb_in: Int) -> Int:
    var na = normalized(a, na_in)
    var nb = normalized(b, nb_in)
    if na < nb:
        return -1
    if na > nb:
        return 1
    comptime W = simd_width_of[DType.float64]()
    var n = na
    while n > 0 and n % W != 0:
        var i = n - 1
        if a[i] < b[i]:
            return -1
        if a[i] > b[i]:
            return 1
        n -= 1
    while n >= W:
        var block_start = n - W
        var av = a.load[width=W](block_start)
        var bv = b.load[width=W](block_start)
        if av.ne(bv).cast[DType.int64]().reduce_add() != 0:
            for ri in range(W):
                var i = n - 1 - ri
                if a[i] < b[i]:
                    return -1
                if a[i] > b[i]:
                    return 1
        n = block_start
    return 0


def subtract_in_place(a: LimbPtr, na: Int, b: LimbPtr, nb: Int) -> Int:
    """Set a = a - b for nonnegative a >= b and return its normalized length."""
    var borrow = UInt64(0)
    for i in range(na):
        var av = UInt64(a[i])
        var bv = borrow
        if i < nb:
            bv += UInt64(b[i])
        if av >= bv:
            a[i] = UInt32(av - bv)
            borrow = 0
        else:
            a[i] = UInt32(BASE + av - bv)
            borrow = 1
    return normalized(a, na)


def add_in_place(a: LimbPtr, b: LimbPtr, n: Int):
    var carry = UInt64(0)
    for i in range(n):
        var total = UInt64(a[i]) + UInt64(b[i]) + carry
        a[i] = UInt32(total & MASK)
        carry = total >> UInt64(32)


def shift_right_one(a: LimbPtr, n_in: Int) -> Int:
    var carry = UInt32(0)
    for ri in range(n_in):
        var i = n_in - 1 - ri
        var next_carry = (a[i] & UInt32(1)) << UInt32(31)
        a[i] = (a[i] >> UInt32(1)) | carry
        carry = next_carry
    return normalized(a, n_in)


def shift_left_one(a: LimbPtr, n_in: Int) -> Int:
    var carry = UInt64(0)
    for i in range(n_in):
        var total = (UInt64(a[i]) << UInt64(1)) | carry
        a[i] = UInt32(total & MASK)
        carry = total >> UInt64(32)
    if carry != 0:
        a[n_in] = UInt32(carry)
        return n_in + 1
    return n_in


def modular_add(x: LimbPtr, y: LimbPtr, modulus: LimbPtr, tmp: LimbPtr, n: Int):
    """Set x = (x + y) % modulus, given 0 <= x,y < modulus."""
    copy_limbs(modulus, tmp, n)
    _ = subtract_in_place(tmp, n, y, n)
    if compare(x, n, tmp, n) >= 0:
        _ = subtract_in_place(x, n, tmp, n)
    else:
        add_in_place(x, y, n)


def modular_multiply(
    a: LimbPtr,
    b: LimbPtr,
    dst: LimbPtr,
    modulus: LimbPtr,
    addend: LimbPtr,
    tmp: LimbPtr,
    n: Int,
):
    zero_limbs(dst, n)
    copy_limbs(a, addend, n)
    for i in range(n):
        var word = b[i]
        for bit in range(32):
            if (word & (UInt32(1) << UInt32(bit))) != 0:
                modular_add(dst, addend, modulus, tmp, n)
            modular_add(addend, addend, modulus, tmp, n)


@export("mg_add")
def mg_add(a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int) abi("C") -> Int:
    if a_addr == 0 or b_addr == 0 or dst_addr == 0 or na <= 0 or nb <= 0:
        return -1
    var a = limbs(a_addr)
    var b = limbs(b_addr)
    var dst = limbs(dst_addr)
    var n = max(na, nb)
    var carry = UInt64(0)
    for i in range(n):
        var total = carry
        if i < na:
            total += UInt64(a[i])
        if i < nb:
            total += UInt64(b[i])
        dst[i] = UInt32(total & MASK)
        carry = total >> UInt64(32)
    if carry != 0:
        dst[n] = UInt32(carry)
        return n + 1
    return normalized(dst, n)


@export("mg_sub")
def mg_sub(a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int) abi("C") -> Int:
    if a_addr == 0 or b_addr == 0 or dst_addr == 0 or na <= 0 or nb <= 0:
        return -1
    var a = limbs(a_addr)
    var b = limbs(b_addr)
    var dst = limbs(dst_addr)
    copy_limbs(a, dst, na)
    return subtract_in_place(dst, na, b, nb)


@export("mg_mul")
def mg_mul(a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int) abi("C") -> Int:
    if a_addr == 0 or b_addr == 0 or dst_addr == 0 or na <= 0 or nb <= 0:
        return -1
    var a = limbs(a_addr)
    var b = limbs(b_addr)
    var dst = limbs(dst_addr)
    zero_limbs(dst, na + nb)
    for i in range(na):
        var carry = UInt64(0)
        for j in range(nb):
            var k = i + j
            var total = (
                UInt64(dst[k]) + UInt64(a[i]) * UInt64(b[j]) + carry
            )
            dst[k] = UInt32(total & MASK)
            carry = total >> UInt64(32)
        dst[i + nb] = UInt32(carry)
    return normalized(dst, na + nb)


@export("mg_gcd")
def mg_gcd(
    a_addr: Int,
    na_in: Int,
    b_addr: Int,
    nb_in: Int,
    u_addr: Int,
    v_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        a_addr == 0
        or b_addr == 0
        or u_addr == 0
        or v_addr == 0
        or dst_addr == 0
        or na_in <= 0
        or nb_in <= 0
    ):
        return -1
    var a = limbs(a_addr)
    var b = limbs(b_addr)
    var u = limbs(u_addr)
    var v = limbs(v_addr)
    var dst = limbs(dst_addr)
    var na = normalized(a, na_in)
    var nb = normalized(b, nb_in)
    if na == 0:
        copy_limbs(b, dst, nb)
        return nb
    if nb == 0:
        copy_limbs(a, dst, na)
        return na
    copy_limbs(a, u, na)
    copy_limbs(b, v, nb)
    var common_twos = 0
    while (u[0] & UInt32(1)) == 0 and (v[0] & UInt32(1)) == 0:
        na = shift_right_one(u, na)
        nb = shift_right_one(v, nb)
        common_twos += 1
    while (u[0] & UInt32(1)) == 0:
        na = shift_right_one(u, na)
    while nb != 0:
        while (v[0] & UInt32(1)) == 0:
            nb = shift_right_one(v, nb)
        if compare(u, na, v, nb) > 0:
            var pointer_tmp = u
            u = v
            v = pointer_tmp
            var length_tmp = na
            na = nb
            nb = length_tmp
        nb = subtract_in_place(v, nb, u, na)
    copy_limbs(u, dst, na)
    for _ in range(common_twos):
        na = shift_left_one(dst, na)
    return normalized(dst, na)


@export("mg_powmod")
def mg_powmod(
    base_addr: Int,
    exponent_addr: Int,
    ne: Int,
    modulus_addr: Int,
    n: Int,
    result_addr: Int,
    power_addr: Int,
    product_addr: Int,
    addend_addr: Int,
    tmp_addr: Int,
) abi("C") -> Int:
    if (
        base_addr == 0
        or exponent_addr == 0
        or modulus_addr == 0
        or result_addr == 0
        or power_addr == 0
        or product_addr == 0
        or addend_addr == 0
        or tmp_addr == 0
        or ne <= 0
        or n <= 0
    ):
        return -1
    var base = limbs(base_addr)
    var exponent = limbs(exponent_addr)
    var modulus = limbs(modulus_addr)
    var result = limbs(result_addr)
    var power = limbs(power_addr)
    var product = limbs(product_addr)
    var addend = limbs(addend_addr)
    var tmp = limbs(tmp_addr)
    zero_limbs(result, n)
    copy_limbs(base, power, n)
    result[0] = 1
    if compare(result, n, modulus, n) >= 0:
        _ = subtract_in_place(result, n, modulus, n)
    for i in range(ne):
        var word = exponent[i]
        for bit in range(32):
            if (word & (UInt32(1) << UInt32(bit))) != 0:
                modular_multiply(
                    result, power, product, modulus, addend, tmp, n
                )
                copy_limbs(product, result, n)
            modular_multiply(
                power, power, product, modulus, addend, tmp, n
            )
            copy_limbs(product, power, n)
    return normalized(result, n)
