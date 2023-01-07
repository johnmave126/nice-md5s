use std::is_x86_feature_detected;

use super::use_intrinsic;
use crate::{EmbarrisinglyParallel, Nibbles, Nicety};

use_intrinsic! {__m256i}

/// An array of 32 nibbles (4-byte).
///
/// Nibbles are tightly packed inside a single [`__m256i`].
///
/// The algorithms implemented here are SIMD based, requiring CPU support to
/// run. User of this struct must detect CPU features before calling any method.
#[derive(Debug, Clone, Copy)]
pub struct Simd(pub(crate) __m256i);

impl Simd {
    /// Mathematical constant `e` in nibbles form.
    ///
    /// `e = 2.71828 18284 59045 23536 02874 71352 6...`
    const MATH_E: [u8; 32] = [
        2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3, 6, 0, 2, 8, 7, 4, 7, 1, 3, 5,
        2, 6,
    ];

    /// Mathematical constant `π` in nibbles form.
    ///
    /// `π = 3.14159 26535 89793 23846 26433 83279 5...`
    const MATH_PI: [u8; 32] = [
        3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7,
        9, 5,
    ];

    /// Construct a new array of nibbles.
    ///
    /// Algorithm is as follows:
    /// - Load 16 bytes into a 128-bit vector: `[nn, nn, nn, nn, ..]`
    /// - Zero extended the vector into a 256-bit vector, making each byte now a
    ///   word: `[00nn, 00nn, 00nn, 00nn, ..]`
    /// - Transform to 2 nibble vectors.
    ///   - Shift each word left by 8 and apply nibble mask, essentially moving
    ///     lo-nibbles to hi-bytes: `x = [0n00, 0n00, 0n00, 0n00, ...]`
    ///   - Shift each word right by 4, essentially moving hi-nibbles to
    ///     lo-bytes: `y = [000n, 000n, 000n, 000n, ..]`
    /// - Combine `x` and `y` by or-ing to get the final nibble vector.
    #[target_feature(enable = "avx,avx2,sse2")]
    pub(crate) unsafe fn new(x: [u8; 16]) -> Self {
        debug_assert!(
            is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("sse2")
        );

        use_intrinsic! {
            _mm256_and_si256, _mm256_cvtepu8_epi16, _mm256_or_si256, _mm256_set1_epi8,
            _mm256_slli_epi16, _mm256_srli_epi16, _mm_loadu_si128,
        }

        let x = _mm_loadu_si128(x.as_ptr().cast());
        // Each byte now occupies 2 bytes
        let x = _mm256_cvtepu8_epi16(x);
        // Shift left to place lo-nibble in hi-byte and clear excess nibbles
        let lo_nibble = _mm256_and_si256(_mm256_slli_epi16(x, 8), _mm256_set1_epi8(0x0Fu8 as i8));
        // Shift right to place hi-nibble in lo-byte
        let hi_nibble = _mm256_srli_epi16(x, 4);

        Self(_mm256_or_si256(hi_nibble, lo_nibble))
    }

    /// Returns number of consecutive leading digits/letters.
    #[target_feature(enable = "avx,avx2")]
    unsafe fn count_leading_digits_and_letters(&self) -> (u8, u8) {
        debug_assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2"));

        use_intrinsic! {_mm256_cmpgt_epi8, _mm256_set1_epi8}

        // The boundary of digit and letter
        let boundary = _mm256_set1_epi8(0x09u8 as i8);

        // Set each byte to 0xFF if it is a letter, 0x00 otherwise.
        let letter_or_digit = _mm256_cmpgt_epi8(self.0, boundary);

        Self::count_trailing_all_zero_and_one_bytes(letter_or_digit)
    }

    /// Returns number of consecutive same nibbles from the start.
    #[target_feature(enable = "avx,avx2")]
    unsafe fn count_leading_homogenous_simd(&self) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2"));

        use_intrinsic! {
            _mm256_permute4x64_epi64, _mm256_setzero_si256, _mm256_shuffle_epi8,
        }
        // Duplicate lo-64 bit cross-lane to all 64-bit lane.
        let leading_byte = _mm256_permute4x64_epi64(self.0, 0);
        // Set all the bytes to the lowest byte in each 128-bit lane, aka, the original
        // lowest byte.
        let leading_byte = _mm256_shuffle_epi8(leading_byte, _mm256_setzero_si256());

        self.count_longest_prefix_simd(&Self(leading_byte))
    }

    /// Returns number of consecutive nibbles equal to mathematical constant `e`
    /// from the start.
    #[target_feature(enable = "avx")]
    unsafe fn count_longest_prefix_e_simd(&self) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx"));

        use_intrinsic! {_mm256_loadu_si256}

        let e = _mm256_loadu_si256(Self::MATH_E.as_ptr().cast());
        self.count_longest_prefix_simd(&Self(e))
    }

    /// Returns number of consecutive nibbles equal to mathematical constant `π`
    /// from the start.
    #[target_feature(enable = "avx")]
    unsafe fn count_longest_prefix_pi_simd(&self) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx"));

        use_intrinsic! {_mm256_loadu_si256}

        let e = _mm256_loadu_si256(Self::MATH_PI.as_ptr().cast());
        self.count_longest_prefix_simd(&Self(e))
    }

    /// Returns number of consecutive nibbles equal to `other` from the start.
    #[target_feature(enable = "avx2")]
    unsafe fn count_longest_prefix_simd(&self, other: &Self) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx2"));

        use_intrinsic! {_mm256_cmpeq_epi8}

        // Set each byte to 0xFF if it is the same between `self` and `other`, 0x00
        // otherwise.
        Self::count_trailing_all_one_bytes(_mm256_cmpeq_epi8(self.0, other.0))
    }

    /// Count the number of consecutive trailing bytes that is 0xFF in a 256-bit
    /// vector.
    ///
    /// The function assumes that each byte of the input vector is either 0xFF
    /// or 0x00.
    #[target_feature(enable = "avx2,bmi1")]
    unsafe fn count_trailing_all_one_bytes(x: __m256i) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi1"));

        use_intrinsic! {_mm256_movemask_epi8, _tzcnt_u32}

        // Gather the hightest bit of each byte, the result is 32-bit.
        let packed_mask = _mm256_movemask_epi8(x) as u32;
        // Count trailing 0's in the complement.
        _tzcnt_u32(!packed_mask) as u8
    }

    /// Count the number of consecutive trailing 0's and 1's in a 256-bit
    /// vector.
    ///
    /// The function assumes that each byte of the input vector is either 0xFF
    /// or 0x00.
    #[target_feature(enable = "avx2,bmi1")]
    unsafe fn count_trailing_all_zero_and_one_bytes(x: __m256i) -> (u8, u8) {
        debug_assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi1"));

        use_intrinsic! {_mm256_movemask_epi8, _tzcnt_u32}

        // Gather the hightest bit of each byte, the result is 32-bit.
        let packed_mask = _mm256_movemask_epi8(x) as u32;
        // Count trailing 0's in `packed_mask` and its complement.
        (
            _tzcnt_u32(packed_mask) as u8,
            _tzcnt_u32(!packed_mask) as u8,
        )
    }
}

impl Nibbles for Simd {
    fn count_leading_digits(&self) -> u8 {
        unsafe { self.count_leading_digits_and_letters().0 }
    }

    fn count_leading_letters(&self) -> u8 {
        unsafe { self.count_leading_digits_and_letters().1 }
    }

    fn count_leading_homogenous(&self) -> u8 {
        unsafe { self.count_leading_homogenous_simd() }
    }

    fn count_longest_prefix_e(&self) -> u8 {
        unsafe { self.count_longest_prefix_e_simd() }
    }

    fn count_longest_prefix_pi(&self) -> u8 {
        unsafe { self.count_longest_prefix_pi_simd() }
    }

    fn count_longest_prefix(&self, other: &Self) -> u8 {
        unsafe { self.count_longest_prefix_simd(other) }
    }

    fn compute_nicety(&self) -> Nicety {
        let (digits, letters) = unsafe { self.count_leading_digits_and_letters() };
        let homogenous = self.count_leading_homogenous();
        let leading_e = self.count_longest_prefix_e();
        let leading_pi = self.count_longest_prefix_pi();
        Nicety {
            digits,
            letters,
            homogenous,
            leading_e,
            leading_pi,
        }
    }
}

impl From<[u8; 16]> for Simd {
    fn from(x: [u8; 16]) -> Self {
        unsafe { Simd::new(x) }
    }
}

impl EmbarrisinglyParallel for Simd {}

#[cfg(all(
    test,
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "bmi1",
    target_feature = "sse2"
))]
mod tests {
    use super::*;
    use_intrinsic! {
        _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_setzero_si256,
        _mm256_xor_si256,
    }

    #[test]
    fn from_u8_array() {
        fn check_eq(x: [u8; 16], y: [u8; 32]) {
            let x: Simd = x.into();
            unsafe {
                let y = _mm256_loadu_si256(y.as_ptr().cast());
                let eq = _mm256_cmpeq_epi8(_mm256_xor_si256(x.0, y), _mm256_setzero_si256());
                let mask = _mm256_movemask_epi8(eq);
                assert_eq!(mask as u32, 0xFFFFFFFF);
            }
        }

        check_eq(
            [
                0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
                0x32, 0x10,
            ],
            [
                0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
                0xf, 0xe, 0xd, 0xc, 0xb, 0xa, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
            ],
        );

        check_eq(
            [
                0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
                0xcd, 0xef,
            ],
            [
                0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
                0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
            ],
        );
    }

    #[test]
    fn count_leading_digits() {
        crate::tests::count_leading_digits::<Simd>(0);
    }

    #[test]
    fn count_leading_letters() {
        crate::tests::count_leading_letters::<Simd>(0);
    }
}
