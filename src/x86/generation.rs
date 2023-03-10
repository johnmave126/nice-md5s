use std::mem::transmute;

use asm_block::asm_block;

use super::use_intrinsic;

/// Generate 16 [0-9a-z] ASCII byte in SIMD. The distribution is not uniform.
///
/// # Algorithm
/// To produce 16 bytes where each byte is either a digit or a lowercase letter
/// in ASCII, this function does the follows:
///
/// * Produce 2 64-bit random numbers.
/// * Load the numbers into a 128-bit vector.
/// * For each byte, dissect it into 5 random bits, 2 random bits, and 1 random
///   bits respectively.
/// * Add bits together, (2 ^ 5 - 1) + (2 ^ 2 - 1) + 1 = 35. So we get a byte in
///   range 0-35.
/// * Generate a mask where a byte is >0x09. These bytes will become letters.
/// * Add 0x30 to all bytes, and add 0x27 to bytes >0x09.
/// * Dump the bytes.
#[target_feature(enable = "avx,avx2,sse2")]
#[inline]
unsafe fn generate_strs_simd<R: rand::Rng, const N: usize>(rng: &mut R) -> [[u8; 32]; N] {
    debug_assert!(
        is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("sse2")
    );

    use_intrinsic! {
        _mm256_add_epi8, _mm256_and_si256, _mm256_cmpgt_epi8, _mm256_loadu_si256, _mm256_set1_epi8,
        _mm256_storeu_si256, _mm256_sub_epi8,
    }

    [0; N].map(|_| {
        // Load 128 random bits
        let v = _mm256_loadu_si256(
            [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ]
            .as_ptr()
            .cast(),
        );
        // Keep 6 bits (0-63)
        let v = _mm256_and_si256(v, _mm256_set1_epi8(0x3Fu8 as i8));
        // Mask bytes in range 36-63
        let gt_35 = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(35));
        // Subtract 36 for those bytes
        let v = _mm256_sub_epi8(v, _mm256_and_si256(_mm256_set1_epi8(36), gt_35));
        // Set each byte to 0xFF if it should be a letter (10-35), otherwise 0x00
        let alpha_mask = _mm256_cmpgt_epi8(v, _mm256_set1_epi8(0x09u8 as i8));
        // Shift each byte so that range starts at ASCII `0`
        let to_numbers = _mm256_add_epi8(v, _mm256_set1_epi8(0x30u8 as i8));
        // Shift bytes that should be a letter by additional 0x27, so that the range
        // starts at ASCII `a`
        let to_alphas = _mm256_and_si256(_mm256_set1_epi8(0x27u8 as i8), alpha_mask);
        // Add shifting together to get correct bytes
        let v = _mm256_add_epi8(to_numbers, to_alphas);

        let mut r = [0; 32];
        _mm256_storeu_si256(r.as_mut_ptr().cast(), v);
        r
    })
}

/// Generate `N` 32-byte array, where each byte is an ASCII character [0-9a-z].
/// For maximum performance, the implementation does not guarantee uniformness
/// of the generated strings.
///
/// # Safety
/// The algorithms implemented here are SIMD based, requiring CPU support to
/// run. User of this fcuntion must detect CPU features before calling
/// by testing `is_x86_feature_detected!("avx") &&
/// is_x86_feature_detected!("sse2")`.
pub fn generate_strs<R: rand::Rng, const N: usize>(rng: &mut R) -> [[u8; 32]; N] {
    unsafe { generate_strs_simd(rng) }
}

/// Compute `N` MD5 digests according to the input, using optimized
/// inline-assembly.
///
/// # Safety
/// `super` is guarded by `#[cfg(any(target_arch = "x86", target_arch =
/// "x86_64"))]`, so we are able to use the assemblies
///
/// # Attribution
/// The assembly is derived from the [implementation by Project Nayuki](https://www.nayuki.io/page/fast-md5-hash-implementation-in-x86-assembly).
/// The copyright notice is kept as follows:
///
/// ```text
/// MD5 hash in x86-64 assembly
///
/// Copyright (c) 2021 Project Nayuki. (MIT License)
/// https://www.nayuki.io/page/fast-md5-hash-implementation-in-x86-assembly
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of
/// this software and associated documentation files (the "Software"), to deal in
/// the Software without restriction, including without limitation the rights to
/// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
/// the Software, and to permit persons to whom the Software is furnished to do so,
/// subject to the following conditions:
/// - The above copyright notice and this permission notice shall be included in
///   all copies or substantial portions of the Software.
/// - The Software is provided "as is", without warranty of any kind, express or
///   implied, including but not limited to the warranties of merchantability,
///   fitness for a particular purpose and noninfringement. In no event shall the
///   authors or copyright holders be liable for any claim, damages or other
///   liability, whether in an action of contract, tort or otherwise, arising from,
///   out of or in connection with the Software or the use or other dealings in the
///   Software.
/// ```
pub fn digest_md5s<const N: usize>(srcs: [[u8; 32]; N]) -> [[u8; 16]; N] {
    srcs.map(|v| unsafe { digest_md5_asm(v) })
}

/// MD5 operators
macro_rules! asm_md5_op {
    ($p: ident, $a: tt, $b: tt, $c: tt, $d: tt, 0, $s: literal, $t: literal, $tmp1: tt, $tmp2: tt) => {
        asm_md5_op!($p, $a, $b, $c, $d, $s, $t, $tmp1, $tmp2)
    };
    ($p: ident, $a: tt, $b: tt, $c: tt, $d: tt, $k: tt, $s: literal, $t: literal, $tmp1: tt, $tmp2: tt) => {
        asm_md5_op!($p, $a, $b, $c, $d, @$k, $s, $t, $tmp1, $tmp2)
    };
    (F, $a: tt, $b: tt, $c: tt, $d: tt, $(@$k: tt,)? $s: literal, $t: literal, $tmp1: tt, $tmp2: tt) => {
        concat!(
            asm_block! {
                mov $tmp1, $c;
                $(add $a, $k;)?
                xor $tmp1, $d;
                and $tmp1, $b;
                xor $tmp1, $d;
            },
            asm_md5_op!(END, $a, $b, $s, $t, $tmp1)
        )
    };
    (G, $a: tt, $b: tt, $c: tt, $d: tt, $(@$k: tt,)? $s: literal, $t: literal, $tmp1: tt, $tmp2: tt) => {
        concat!(
            asm_block! {
                mov $tmp1, $d;
                mov $tmp2, $d;
                $(add $a, $k;)?
                not $tmp1;
                and $tmp2, $b;
                and $tmp1, $c;
                or $tmp1, $tmp2;
            },
            asm_md5_op!(END, $a, $b, $s, $t, $tmp1)
        )
    };
    (H, $a: tt, $b: tt, $c: tt, $d: tt, $(@$k: tt,)? $s: literal, $t: literal, $tmp1: tt, $tmp2: tt) => {
        concat!(
            asm_block! {
                mov $tmp1, $c;
                $(add $a, $k;)?
                xor $tmp1, $d;
                xor $tmp1, $b;
            },
            asm_md5_op!(END, $a, $b, $s, $t, $tmp1)
        )
    };
    (I, $a: tt, $b: tt, $c: tt, $d: tt, $(@$k: tt,)? $s: literal, $t: literal, $tmp1: tt, $tmp2: tt) => {
        concat!(
            asm_block! {
                mov $tmp1, $d;
                not $tmp1;
                $(add $a, $k;)?
                or $tmp1, $b;
                xor $tmp1, $c;
            },
            asm_md5_op!(END, $a, $b, $s, $t, $tmp1)
        )
    };
    (END, $a: tt, $b: tt, $s: literal, $t: literal, $tmp: tt) => {
        asm_block! {
            lea $a, [$a + $tmp + $t];
            rol $a, $s;
            add $a, $b;
        }
    }
}

#[rustfmt::skip]
macro_rules! asm_md5 {
    (
        // states
        $a: tt, $b: tt, $c: tt, $d: tt,
        // inputs
        $x0: tt, $x1: tt, $x2: tt, $x3: tt,
        $x4: tt, $x5: tt, $x6: tt, $x7: tt,
        $x8: tt, $x9: tt, $xa: tt, $xb: tt,
        $xc: tt, $xd: tt, $xe: tt, $xf: tt,
        // clobbers
        $t1: tt, $t2: tt
     ) => {
        concat!(
            // round 1
            asm_md5_op!(F, $a, $b, $c, $d, $x0,  7, 0xd76aa478, $t1, $t2),
            asm_md5_op!(F, $d, $a, $b, $c, $x1, 12, 0xe8c7b756, $t1, $t2),
            asm_md5_op!(F, $c, $d, $a, $b, $x2, 17, 0x242070db, $t1, $t2),
            asm_md5_op!(F, $b, $c, $d, $a, $x3, 22, 0xc1bdceee, $t1, $t2),
   
            asm_md5_op!(F, $a, $b, $c, $d, $x4,  7, 0xf57c0faf, $t1, $t2),
            asm_md5_op!(F, $d, $a, $b, $c, $x5, 12, 0x4787c62a, $t1, $t2),
            asm_md5_op!(F, $c, $d, $a, $b, $x6, 17, 0xa8304613, $t1, $t2),
            asm_md5_op!(F, $b, $c, $d, $a, $x7, 22, 0xfd469501, $t1, $t2),
 
            asm_md5_op!(F, $a, $b, $c, $d, $x8,  7, 0x698098d8, $t1, $t2),
            asm_md5_op!(F, $d, $a, $b, $c, $x9, 12, 0x8b44f7af, $t1, $t2),
            asm_md5_op!(F, $c, $d, $a, $b, $xa, 17, 0xffff5bb1, $t1, $t2),
            asm_md5_op!(F, $b, $c, $d, $a, $xb, 22, 0x895cd7be, $t1, $t2),
 
            asm_md5_op!(F, $a, $b, $c, $d, $xc,  7, 0x6b901122, $t1, $t2),
            asm_md5_op!(F, $d, $a, $b, $c, $xd, 12, 0xfd987193, $t1, $t2),
            asm_md5_op!(F, $c, $d, $a, $b, $xe, 17, 0xa679438e, $t1, $t2),
            asm_md5_op!(F, $b, $c, $d, $a, $xf, 22, 0x49b40821, $t1, $t2),

            // round 2
            asm_md5_op!(G, $a, $b, $c, $d, $x1,  5, 0xf61e2562, $t1, $t2),
            asm_md5_op!(G, $d, $a, $b, $c, $x6,  9, 0xc040b340, $t1, $t2),
            asm_md5_op!(G, $c, $d, $a, $b, $xb, 14, 0x265e5a51, $t1, $t2),
            asm_md5_op!(G, $b, $c, $d, $a, $x0, 20, 0xe9b6c7aa, $t1, $t2),

            asm_md5_op!(G, $a, $b, $c, $d, $x5,  5, 0xd62f105d, $t1, $t2),
            asm_md5_op!(G, $d, $a, $b, $c, $xa,  9, 0x02441453, $t1, $t2),
            asm_md5_op!(G, $c, $d, $a, $b, $xf, 14, 0xd8a1e681, $t1, $t2),
            asm_md5_op!(G, $b, $c, $d, $a, $x4, 20, 0xe7d3fbc8, $t1, $t2),

            asm_md5_op!(G, $a, $b, $c, $d, $x9,  5, 0x21e1cde6, $t1, $t2),
            asm_md5_op!(G, $d, $a, $b, $c, $xe,  9, 0xc33707d6, $t1, $t2),
            asm_md5_op!(G, $c, $d, $a, $b, $x3, 14, 0xf4d50d87, $t1, $t2),
            asm_md5_op!(G, $b, $c, $d, $a, $x8, 20, 0x455a14ed, $t1, $t2),

            asm_md5_op!(G, $a, $b, $c, $d, $xd,  5, 0xa9e3e905, $t1, $t2),
            asm_md5_op!(G, $d, $a, $b, $c, $x2,  9, 0xfcefa3f8, $t1, $t2),
            asm_md5_op!(G, $c, $d, $a, $b, $x7, 14, 0x676f02d9, $t1, $t2),
            asm_md5_op!(G, $b, $c, $d, $a, $xc, 20, 0x8d2a4c8a, $t1, $t2),

            // round 3
            asm_md5_op!(H, $a, $b, $c, $d, $x5,  4, 0xfffa3942, $t1, $t2),
            asm_md5_op!(H, $d, $a, $b, $c, $x8, 11, 0x8771f681, $t1, $t2),
            asm_md5_op!(H, $c, $d, $a, $b, $xb, 16, 0x6d9d6122, $t1, $t2),
            asm_md5_op!(H, $b, $c, $d, $a, $xe, 23, 0xfde5380c, $t1, $t2),

            asm_md5_op!(H, $a, $b, $c, $d, $x1,  4, 0xa4beea44, $t1, $t2),
            asm_md5_op!(H, $d, $a, $b, $c, $x4, 11, 0x4bdecfa9, $t1, $t2),
            asm_md5_op!(H, $c, $d, $a, $b, $x7, 16, 0xf6bb4b60, $t1, $t2),
            asm_md5_op!(H, $b, $c, $d, $a, $xa, 23, 0xbebfbc70, $t1, $t2),

            asm_md5_op!(H, $a, $b, $c, $d, $xd,  4, 0x289b7ec6, $t1, $t2),
            asm_md5_op!(H, $d, $a, $b, $c, $x0, 11, 0xeaa127fa, $t1, $t2),
            asm_md5_op!(H, $c, $d, $a, $b, $x3, 16, 0xd4ef3085, $t1, $t2),
            asm_md5_op!(H, $b, $c, $d, $a, $x6, 23, 0x04881d05, $t1, $t2),

            asm_md5_op!(H, $a, $b, $c, $d, $x9,  4, 0xd9d4d039, $t1, $t2),
            asm_md5_op!(H, $d, $a, $b, $c, $xc, 11, 0xe6db99e5, $t1, $t2),
            asm_md5_op!(H, $c, $d, $a, $b, $xf, 16, 0x1fa27cf8, $t1, $t2),
            asm_md5_op!(H, $b, $c, $d, $a, $x2, 23, 0xc4ac5665, $t1, $t2),

            // round 4
            asm_md5_op!(I, $a, $b, $c, $d, $x0,  6, 0xf4292244, $t1, $t2),
            asm_md5_op!(I, $d, $a, $b, $c, $x7, 10, 0x432aff97, $t1, $t2),
            asm_md5_op!(I, $c, $d, $a, $b, $xe, 15, 0xab9423a7, $t1, $t2),
            asm_md5_op!(I, $b, $c, $d, $a, $x5, 21, 0xfc93a039, $t1, $t2),

            asm_md5_op!(I, $a, $b, $c, $d, $xc,  6, 0x655b59c3, $t1, $t2),
            asm_md5_op!(I, $d, $a, $b, $c, $x3, 10, 0x8f0ccc92, $t1, $t2),
            asm_md5_op!(I, $c, $d, $a, $b, $xa, 15, 0xffeff47d, $t1, $t2),
            asm_md5_op!(I, $b, $c, $d, $a, $x1, 21, 0x85845dd1, $t1, $t2),

            asm_md5_op!(I, $a, $b, $c, $d, $x8,  6, 0x6fa87e4f, $t1, $t2),
            asm_md5_op!(I, $d, $a, $b, $c, $xf, 10, 0xfe2ce6e0, $t1, $t2),
            asm_md5_op!(I, $c, $d, $a, $b, $x6, 15, 0xa3014314, $t1, $t2),
            asm_md5_op!(I, $b, $c, $d, $a, $xd, 21, 0x4e0811a1, $t1, $t2),

            asm_md5_op!(I, $a, $b, $c, $d, $x4,  6, 0xf7537e82, $t1, $t2),
            asm_md5_op!(I, $d, $a, $b, $c, $xb, 10, 0xbd3af235, $t1, $t2),
            asm_md5_op!(I, $c, $d, $a, $b, $x2, 15, 0x2ad7d2bb, $t1, $t2),
            asm_md5_op!(I, $b, $c, $d, $a, $x9, 21, 0xeb86d391, $t1, $t2),
        )
    }
}

/// x86_64 implementation of MD5 digest.
///
/// # Design
/// * The algorithm is derived from is derived from the [implementation by Project Nayuki](https://www.nayuki.io/page/fast-md5-hash-implementation-in-x86-assembly).
/// * Since we always handle 32-byte data, the following optimizations are made:
///   * Fixed initial state.
///   * All 32-byte are put into 8 registers.
///   * Padding is known and fixed.
///   * For known 0-padding, we can shave an `add` instruction.
#[cfg(target_arch = "x86_64")]
unsafe fn digest_md5_asm(x: [u8; 32]) -> [u8; 16] {
    use std::arch::asm;

    let x: [u32; 8] = transmute(x);
    let mut state: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

    // Due to padding, we have following values:
    // x[0..8] = x[0..8]
    // x[8] = 0x80, aka, first bit 1, otherwise 0
    // x[9..14] = 0
    // x[14] = 0x100
    // x[15] = 0
    // Since 256 = 0x0000_0000_0000_0100
    asm!(
        asm_md5!(
            // states
            {a:e}, {b:e}, {c:e}, {d:e},
            // inputs
            {x0:e}, {x1:e}, {x2:e}, {x3:e},
            {x4:e}, {x5:e}, {x6:e}, {x7:e},
              0x80,      0,      0,      0,
                 0,      0,  0x100,      0,
            {t1:e}, {t2:e}
        ),

        a = inout(reg) state[0],
        b = inout(reg) state[1],
        c = inout(reg) state[2],
        d = inout(reg) state[3],
        t1 = out(reg) _,
        t2 = out(reg) _,
        x0 = in(reg) x[0],
        x1 = in(reg) x[1],
        x2 = in(reg) x[2],
        x3 = in(reg) x[3],
        x4 = in(reg) x[4],
        x5 = in(reg) x[5],
        x6 = in(reg) x[6],
        x7 = in(reg) x[7],
    );

    // Add back initial state
    state[0] = state[0].wrapping_add(0x67452301);
    state[1] = state[1].wrapping_add(0xefcdab89);
    state[2] = state[2].wrapping_add(0x98badcfe);
    state[3] = state[3].wrapping_add(0x10325476);

    transmute(state)
}

/// x86 implementation of MD5 digest.
///
/// # Design
/// * The algorithm is derived from is derived from the [implementation by Project Nayuki](https://www.nayuki.io/page/fast-md5-hash-implementation-in-x86-assembly).
/// * Since we always handle 16-byte data, the following optimizations are made:
///   * Fixed initial state.
///   * Padding is known and fixed.
///   * For known 0-padding, we can shave an `add` instruction.
///   * The number of registers on `x86` is very limited, it does not help when
///     `rustc` does not allow to allocate `ebp`, and sometimes refuses `esi`
///     due to LLVM limitations. Hence we manually specify all the register
///     names, save and restore `esi/ebp` so that we have enough registers to
///     work with.
#[cfg(target_arch = "x86")]
unsafe fn digest_md5_asm(x: [u8; 32]) -> [u8; 16] {
    use std::arch::asm;

    let mut state: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];

    // Due to padding, we have following values:
    // x[0..4] = x[0..4]
    // x[4] = 0x80, aka, first bit 1, otherwise 0
    // x[5..14] = 0
    // x[14] = 0x80
    // x[15] = 0
    // Since 128 = 0x0000_0000_0000_0080
    asm!(
        // Save esi
        "sub esp, 8",
        "mov [esp], esi",
        "mov [esp + 4], ebp",

        // Move out x
        "mov ebp, edi",

        asm_md5!(
            // states
            eax, ebx, ecx, edx,
            // inputs
             [ebp + 0],  [ebp + 4],  [ebp + 8], [ebp + 12],
            [ebp + 16], [ebp + 20], [ebp + 24], [ebp + 28],
                  0x80,          0,          0,          0,
                     0,          0,      0x100,          0,
            esi, edi
        ),

        // Restore esi
        "mov esi, [esp]",
        "mov ebp, [esp + 4]",
        "add esp, 8",

        inout("eax") state[0],
        inout("ebx") state[1],
        inout("ecx") state[2],
        inout("edx") state[3],
        lateout("edi") _,
        in("edi") x.as_ptr(),
    );

    // Add back initial state
    state[0] = state[0].wrapping_add(0x67452301);
    state[1] = state[1].wrapping_add(0xefcdab89);
    state[2] = state[2].wrapping_add(0x98badcfe);
    state[3] = state[3].wrapping_add(0x10325476);

    transmute(state)
}

/// Compute 8 MD5 digests according to the input, using SIMD for speeding up.
///
/// # Safety
/// The algorithms implemented here are SIMD based, requiring CPU support to
/// run. User of this function must detect CPU features before calling
/// by testing `is_x86_feature_detected!("avx") &&
/// is_x86_feature_detected!("avx2")`.
pub fn digest_md5s_simd(srcs: [[u8; 32]; 8]) -> [[u8; 16]; 8] {
    unsafe { digest_md5s_batch_simd(srcs) }
}

/// SIMD implementation of MD5 algorithm for 8 128-bit input at the same time.
#[target_feature(enable = "avx,avx2")]
unsafe fn digest_md5s_batch_simd(srcs: [[u8; 32]; 8]) -> [[u8; 16]; 8] {
    debug_assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2"));

    use_intrinsic! {
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_loadu_si256, _mm256_or_si256,
        _mm256_set1_epi32, _mm256_set1_epi8, _mm256_slli_epi32, _mm256_srli_epi32,
        _mm256_storeu_si256, _mm256_xor_si256,
    }

    unsafe fn rotate_left<const L: i32, const R: i32>(x: __m256i) -> __m256i {
        debug_assert_eq!(L + R, 32);
        let hi = _mm256_slli_epi32(x, L);
        let lo = _mm256_srli_epi32(x, R);
        _mm256_or_si256(hi, lo)
    }

    unsafe fn op_f<const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        x: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp = _mm256_xor_si256(c, d);
        a = _mm256_add_epi32(a, x);
        tmp = _mm256_and_si256(tmp, b);
        tmp = _mm256_xor_si256(tmp, d);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    unsafe fn op_f_c<const C: u32, const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp = _mm256_xor_si256(c, d);
        a = _mm256_add_epi32(a, _mm256_set1_epi32(C as i32));
        tmp = _mm256_and_si256(tmp, b);
        tmp = _mm256_xor_si256(tmp, d);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    unsafe fn op_g<const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        x: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp1 = _mm256_xor_si256(d, _mm256_set1_epi8(0xFFu8 as i8));
        a = _mm256_add_epi32(a, x);
        let tmp2 = _mm256_and_si256(b, d);
        tmp1 = _mm256_and_si256(tmp1, c);
        tmp1 = _mm256_or_si256(tmp1, tmp2);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp1);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    unsafe fn op_g_c<const C: u32, const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp1 = _mm256_xor_si256(d, _mm256_set1_epi8(0xFFu8 as i8));
        a = _mm256_add_epi32(a, _mm256_set1_epi32(C as i32));
        let tmp2 = _mm256_and_si256(b, d);
        tmp1 = _mm256_and_si256(tmp1, c);
        tmp1 = _mm256_or_si256(tmp1, tmp2);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp1);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    unsafe fn op_h<const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        x: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp = _mm256_xor_si256(c, d);
        a = _mm256_add_epi32(a, x);
        tmp = _mm256_xor_si256(tmp, b);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    unsafe fn op_h_c<const C: u32, const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp = _mm256_xor_si256(c, d);
        a = _mm256_add_epi32(a, _mm256_set1_epi32(C as i32));
        tmp = _mm256_xor_si256(tmp, b);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    unsafe fn op_i<const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        x: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp = _mm256_xor_si256(d, _mm256_set1_epi8(0xFFu8 as i8));
        a = _mm256_add_epi32(a, x);
        tmp = _mm256_or_si256(tmp, b);
        tmp = _mm256_xor_si256(tmp, c);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    unsafe fn op_i_c<const C: u32, const L: i32, const R: i32>(
        mut a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        t: u32,
    ) -> __m256i {
        let t = _mm256_set1_epi32(t as i32);
        let mut tmp = _mm256_xor_si256(d, _mm256_set1_epi8(0xFFu8 as i8));
        a = _mm256_add_epi32(a, _mm256_set1_epi32(C as i32));
        tmp = _mm256_or_si256(tmp, b);
        tmp = _mm256_xor_si256(tmp, c);
        a = _mm256_add_epi32(a, t);
        a = _mm256_add_epi32(a, tmp);
        a = rotate_left::<L, R>(a);
        _mm256_add_epi32(a, b)
    }

    let srcs: [[u32; 8]; 8] = transmute(srcs);

    let x0 = srcs.map(|v| v[0]);
    let x0 = _mm256_loadu_si256(x0.as_ptr().cast());
    let x1 = srcs.map(|v| v[1]);
    let x1 = _mm256_loadu_si256(x1.as_ptr().cast());
    let x2 = srcs.map(|v| v[2]);
    let x2 = _mm256_loadu_si256(x2.as_ptr().cast());
    let x3 = srcs.map(|v| v[3]);
    let x3 = _mm256_loadu_si256(x3.as_ptr().cast());
    let x4 = srcs.map(|v| v[4]);
    let x4 = _mm256_loadu_si256(x4.as_ptr().cast());
    let x5 = srcs.map(|v| v[5]);
    let x5 = _mm256_loadu_si256(x5.as_ptr().cast());
    let x6 = srcs.map(|v| v[6]);
    let x6 = _mm256_loadu_si256(x6.as_ptr().cast());
    let x7 = srcs.map(|v| v[7]);
    let x7 = _mm256_loadu_si256(x7.as_ptr().cast());

    let mut a = _mm256_set1_epi32(0x67452301u32 as i32);
    let mut b = _mm256_set1_epi32(0xefcdab89u32 as i32);
    let mut c = _mm256_set1_epi32(0x98badcfeu32 as i32);
    let mut d = _mm256_set1_epi32(0x10325476u32 as i32);

    // round 1
    a = op_f::<7, 25>(a, b, c, d, x0, 0xd76aa478);
    d = op_f::<12, 20>(d, a, b, c, x1, 0xe8c7b756);
    c = op_f::<17, 15>(c, d, a, b, x2, 0x242070db);
    b = op_f::<22, 10>(b, c, d, a, x3, 0xc1bdceee);

    a = op_f::<7, 25>(a, b, c, d, x4, 0xf57c0faf);
    d = op_f::<12, 20>(d, a, b, c, x5, 0x4787c62a);
    c = op_f::<17, 15>(c, d, a, b, x6, 0xa8304613);
    b = op_f::<22, 10>(b, c, d, a, x7, 0xfd469501);

    a = op_f_c::<0x80, 7, 25>(a, b, c, d, 0x698098d8);
    d = op_f_c::<0, 12, 20>(d, a, b, c, 0x8b44f7af);
    c = op_f_c::<0, 17, 15>(c, d, a, b, 0xffff5bb1);
    b = op_f_c::<0, 22, 10>(b, c, d, a, 0x895cd7be);

    a = op_f_c::<0, 7, 25>(a, b, c, d, 0x6b901122);
    d = op_f_c::<0, 12, 20>(d, a, b, c, 0xfd987193);
    c = op_f_c::<0x100, 17, 15>(c, d, a, b, 0xa679438e);
    b = op_f_c::<0, 22, 10>(b, c, d, a, 0x49b40821);

    // round 2
    a = op_g::<5, 27>(a, b, c, d, x1, 0xf61e2562);
    d = op_g::<9, 23>(d, a, b, c, x6, 0xc040b340);
    c = op_g_c::<0, 14, 18>(c, d, a, b, 0x265e5a51);
    b = op_g::<20, 12>(b, c, d, a, x0, 0xe9b6c7aa);

    a = op_g::<5, 27>(a, b, c, d, x5, 0xd62f105d);
    d = op_g_c::<0, 9, 23>(d, a, b, c, 0x02441453);
    c = op_g_c::<0, 14, 18>(c, d, a, b, 0xd8a1e681);
    b = op_g::<20, 12>(b, c, d, a, x4, 0xe7d3fbc8);

    a = op_g_c::<0, 5, 27>(a, b, c, d, 0x21e1cde6);
    d = op_g_c::<0x100, 9, 23>(d, a, b, c, 0xc33707d6);
    c = op_g::<14, 18>(c, d, a, b, x3, 0xf4d50d87);
    b = op_g_c::<0x80, 20, 12>(b, c, d, a, 0x455a14ed);

    a = op_g_c::<0, 5, 27>(a, b, c, d, 0xa9e3e905);
    d = op_g::<9, 23>(d, a, b, c, x2, 0xfcefa3f8);
    c = op_g::<14, 18>(c, d, a, b, x7, 0x676f02d9);
    b = op_g_c::<0, 20, 12>(b, c, d, a, 0x8d2a4c8a);

    // round 3
    a = op_h::<4, 28>(a, b, c, d, x5, 0xfffa3942);
    d = op_h_c::<0x80, 11, 21>(d, a, b, c, 0x8771f681);
    c = op_h_c::<0, 16, 16>(c, d, a, b, 0x6d9d6122);
    b = op_h_c::<0x100, 23, 9>(b, c, d, a, 0xfde5380c);

    a = op_h::<4, 28>(a, b, c, d, x1, 0xa4beea44);
    d = op_h::<11, 21>(d, a, b, c, x4, 0x4bdecfa9);
    c = op_h::<16, 16>(c, d, a, b, x7, 0xf6bb4b60);
    b = op_h_c::<0, 23, 9>(b, c, d, a, 0xbebfbc70);

    a = op_h_c::<0, 4, 28>(a, b, c, d, 0x289b7ec6);
    d = op_h::<11, 21>(d, a, b, c, x0, 0xeaa127fa);
    c = op_h::<16, 16>(c, d, a, b, x3, 0xd4ef3085);
    b = op_h::<23, 9>(b, c, d, a, x6, 0x04881d05);

    a = op_h_c::<0, 4, 28>(a, b, c, d, 0xd9d4d039);
    d = op_h_c::<0, 11, 21>(d, a, b, c, 0xe6db99e5);
    c = op_h_c::<0, 16, 16>(c, d, a, b, 0x1fa27cf8);
    b = op_h::<23, 9>(b, c, d, a, x2, 0xc4ac5665);

    // round 4
    a = op_i::<6, 26>(a, b, c, d, x0, 0xf4292244);
    d = op_i::<10, 22>(d, a, b, c, x7, 0x432aff97);
    c = op_i_c::<0x100, 15, 17>(c, d, a, b, 0xab9423a7);
    b = op_i::<21, 11>(b, c, d, a, x5, 0xfc93a039);

    a = op_i_c::<0, 6, 26>(a, b, c, d, 0x655b59c3);
    d = op_i::<10, 22>(d, a, b, c, x3, 0x8f0ccc92);
    c = op_i_c::<0, 15, 17>(c, d, a, b, 0xffeff47d);
    b = op_i::<21, 11>(b, c, d, a, x1, 0x85845dd1);

    a = op_i_c::<0x80, 6, 26>(a, b, c, d, 0x6fa87e4f);
    d = op_i_c::<0, 10, 22>(d, a, b, c, 0xfe2ce6e0);
    c = op_i::<15, 17>(c, d, a, b, x6, 0xa3014314);
    b = op_i_c::<0, 21, 11>(b, c, d, a, 0x4e0811a1);

    a = op_i::<6, 26>(a, b, c, d, x4, 0xf7537e82);
    d = op_i_c::<0, 10, 22>(d, a, b, c, 0xbd3af235);
    c = op_i::<15, 17>(c, d, a, b, x2, 0x2ad7d2bb);
    b = op_i_c::<0, 21, 11>(b, c, d, a, 0xeb86d391);

    a = _mm256_add_epi32(a, _mm256_set1_epi32(0x67452301u32 as i32));
    b = _mm256_add_epi32(b, _mm256_set1_epi32(0xefcdab89u32 as i32));
    c = _mm256_add_epi32(c, _mm256_set1_epi32(0x98badcfeu32 as i32));
    d = _mm256_add_epi32(d, _mm256_set1_epi32(0x10325476u32 as i32));

    let mut state0: [u32; 8] = [0; 8];
    let mut state1: [u32; 8] = [0; 8];
    let mut state2: [u32; 8] = [0; 8];
    let mut state3: [u32; 8] = [0; 8];

    _mm256_storeu_si256(state0.as_mut_ptr().cast(), a);
    _mm256_storeu_si256(state1.as_mut_ptr().cast(), b);
    _mm256_storeu_si256(state2.as_mut_ptr().cast(), c);
    _mm256_storeu_si256(state3.as_mut_ptr().cast(), d);

    let mut state = [[0; 4]; 8];
    for i in 0..8 {
        state[i] = [state0[i], state1[i], state2[i], state3[i]];
    }

    transmute(state)
}

#[cfg(test)]
mod tests {
    use md5::{Digest, Md5};
    #[cfg(all(
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "sse2"
    ))]
    use rand::thread_rng;

    use super::*;

    #[cfg(all(
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "sse2"
    ))]
    #[test]
    fn test_generate_str() {
        let mut rng = thread_rng();

        // Count the frequency that each byte appears
        // The probability of any given digit or letter not appearing at all is
        // <10^{-27}
        let mut frequency: [usize; 256] = [0; 256];

        for _ in 0..1000 {
            let s = generate_strs::<_, 1>(&mut rng)[0];
            for b in s.into_iter() {
                frequency[b as usize] += 1;
            }
        }

        // Make sure the byte that shouldn't appear didn't appear
        for i in (0..=47).chain(58..=96).chain(123..=255) {
            assert_eq!(frequency[i], 0, "byte `{i:#04x}` should not appear");
        }

        // Make sure the byte that should appear did
        for i in (48..=57).chain(97..=122) {
            assert_ne!(frequency[i], 0, "byte `{i:#04x}` should appear");
        }
    }

    #[test]
    fn test_md5() {
        let testcases = [
            [0x00; 32],
            [0x01; 32],
            [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04,
                0x03, 0x02, 0x01, 0x00,
            ],
            [
                0x61, 0xae, 0xc2, 0x6f, 0x1b, 0x90, 0x95, 0x78, 0xef, 0x63, 0x8a, 0xe0, 0x2d, 0xac,
                0x09, 0x77, 0x61, 0xae, 0xc2, 0x6f, 0x1b, 0x90, 0x95, 0x78, 0xef, 0x63, 0x8a, 0xe0,
                0x2d, 0xac, 0x09, 0x77,
            ],
        ];

        for testcase in testcases.into_iter() {
            assert_eq!(
                Into::<[u8; 16]>::into(Md5::digest(testcase.as_slice())),
                digest_md5s([testcase])[0],
                "testing md5 digest of 0x{}",
                hex::encode(testcase)
            );
        }
    }

    #[test]
    fn test_md5_batch() {
        let testcases = [
            [0x00; 32],
            [0x01; 32],
            [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04,
                0x03, 0x02, 0x01, 0x00,
            ],
            [
                0x61, 0xae, 0xc2, 0x6f, 0x1b, 0x90, 0x95, 0x78, 0xef, 0x63, 0x8a, 0xe0, 0x2d, 0xac,
                0x09, 0x77, 0x61, 0xae, 0xc2, 0x6f, 0x1b, 0x90, 0x95, 0x78, 0xef, 0x63, 0x8a, 0xe0,
                0x2d, 0xac, 0x09, 0x77,
            ],
        ];

        for testcase in testcases.into_iter() {
            let reference: [u8; 16] = Md5::digest(testcase.as_slice()).into();
            let digests = unsafe { digest_md5s_batch_simd([testcase; 8]) };
            for (i, digest) in digests.into_iter().enumerate() {
                assert_eq!(
                    reference,
                    digest,
                    "testing lane {i} on md5 digest of 0x{}",
                    hex::encode(testcase)
                );
            }
        }
    }
}
