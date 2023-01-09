use std::{
    is_x86_feature_detected,
    mem::{transmute, MaybeUninit},
};

use super::use_intrinsic;
use crate::{x86::Simd, Nibbles, NibblesBatch, Nicety};

use_intrinsic! {__m256i}

/// **8** arrays of 32 nibbles (4-byte).
///
/// The algorithms implemented here are SIMD based, requiring CPU support to
/// run. User of this struct must detect CPU features before calling any method.
///
/// This struct provides a lossy algorithm for batch operation. For each nicety,
/// the algorithm reports the accurate number only if the nicety is larger than
/// 4, otherwise 0 is reported.
///
/// The general algorithm is to use SIMD to detect whether any of the 8 inputs
/// has nicety greater than 4, and use [`crate::baseline`] algorithm to test
/// those.
#[derive(Debug, Clone, Copy)]
pub struct SimdLossy8(__m256i, [[u8; 16]; 8]);

impl SimdLossy8 {
    /// Mathematical constant `e` in nibbles form for 4 nibbles.
    ///
    /// `e = 2.71828 18284 59045 23536 02874 71352 6...`
    const MATH_E: i32 = 0x08010702u32 as i32;

    /// Mathematical constant `π` in nibbles form for 4 nibbles.
    ///
    /// `π = 3.14159 26535 89793 23846 26433 83279 5...`
    const MATH_PI: i32 = 0x01040103u32 as i32;

    /// The lookup table storing the gathered indices of 32-bit integers which
    /// is all 1
    const LUT_ALL_ONE: [u32; 256] = [
        0x00000000, 0x00000001, 0x00000002, 0x00000021, 0x00000003, 0x00000031, 0x00000032,
        0x00000321, 0x00000004, 0x00000041, 0x00000042, 0x00000421, 0x00000043, 0x00000431,
        0x00000432, 0x00004321, 0x00000005, 0x00000051, 0x00000052, 0x00000521, 0x00000053,
        0x00000531, 0x00000532, 0x00005321, 0x00000054, 0x00000541, 0x00000542, 0x00005421,
        0x00000543, 0x00005431, 0x00005432, 0x00054321, 0x00000006, 0x00000061, 0x00000062,
        0x00000621, 0x00000063, 0x00000631, 0x00000632, 0x00006321, 0x00000064, 0x00000641,
        0x00000642, 0x00006421, 0x00000643, 0x00006431, 0x00006432, 0x00064321, 0x00000065,
        0x00000651, 0x00000652, 0x00006521, 0x00000653, 0x00006531, 0x00006532, 0x00065321,
        0x00000654, 0x00006541, 0x00006542, 0x00065421, 0x00006543, 0x00065431, 0x00065432,
        0x00654321, 0x00000007, 0x00000071, 0x00000072, 0x00000721, 0x00000073, 0x00000731,
        0x00000732, 0x00007321, 0x00000074, 0x00000741, 0x00000742, 0x00007421, 0x00000743,
        0x00007431, 0x00007432, 0x00074321, 0x00000075, 0x00000751, 0x00000752, 0x00007521,
        0x00000753, 0x00007531, 0x00007532, 0x00075321, 0x00000754, 0x00007541, 0x00007542,
        0x00075421, 0x00007543, 0x00075431, 0x00075432, 0x00754321, 0x00000076, 0x00000761,
        0x00000762, 0x00007621, 0x00000763, 0x00007631, 0x00007632, 0x00076321, 0x00000764,
        0x00007641, 0x00007642, 0x00076421, 0x00007643, 0x00076431, 0x00076432, 0x00764321,
        0x00000765, 0x00007651, 0x00007652, 0x00076521, 0x00007653, 0x00076531, 0x00076532,
        0x00765321, 0x00007654, 0x00076541, 0x00076542, 0x00765421, 0x00076543, 0x00765431,
        0x00765432, 0x07654321, 0x00000008, 0x00000081, 0x00000082, 0x00000821, 0x00000083,
        0x00000831, 0x00000832, 0x00008321, 0x00000084, 0x00000841, 0x00000842, 0x00008421,
        0x00000843, 0x00008431, 0x00008432, 0x00084321, 0x00000085, 0x00000851, 0x00000852,
        0x00008521, 0x00000853, 0x00008531, 0x00008532, 0x00085321, 0x00000854, 0x00008541,
        0x00008542, 0x00085421, 0x00008543, 0x00085431, 0x00085432, 0x00854321, 0x00000086,
        0x00000861, 0x00000862, 0x00008621, 0x00000863, 0x00008631, 0x00008632, 0x00086321,
        0x00000864, 0x00008641, 0x00008642, 0x00086421, 0x00008643, 0x00086431, 0x00086432,
        0x00864321, 0x00000865, 0x00008651, 0x00008652, 0x00086521, 0x00008653, 0x00086531,
        0x00086532, 0x00865321, 0x00008654, 0x00086541, 0x00086542, 0x00865421, 0x00086543,
        0x00865431, 0x00865432, 0x08654321, 0x00000087, 0x00000871, 0x00000872, 0x00008721,
        0x00000873, 0x00008731, 0x00008732, 0x00087321, 0x00000874, 0x00008741, 0x00008742,
        0x00087421, 0x00008743, 0x00087431, 0x00087432, 0x00874321, 0x00000875, 0x00008751,
        0x00008752, 0x00087521, 0x00008753, 0x00087531, 0x00087532, 0x00875321, 0x00008754,
        0x00087541, 0x00087542, 0x00875421, 0x00087543, 0x00875431, 0x00875432, 0x08754321,
        0x00000876, 0x00008761, 0x00008762, 0x00087621, 0x00008763, 0x00087631, 0x00087632,
        0x00876321, 0x00008764, 0x00087641, 0x00087642, 0x00876421, 0x00087643, 0x00876431,
        0x00876432, 0x08764321, 0x00008765, 0x00087651, 0x00087652, 0x00876521, 0x00087653,
        0x00876531, 0x00876532, 0x08765321, 0x00087654, 0x00876541, 0x00876542, 0x08765421,
        0x00876543, 0x08765431, 0x08765432, 0x87654321,
    ];

    /// Construct a new array of first 4 nibbles.
    unsafe fn new(x: [[u8; 16]; 8]) -> Self {
        let first_2_bytes = x.map(|v| [v[0], v[1]]);
        Self(Simd::new(transmute(first_2_bytes)).0, x)
    }

    /// Load original inputs as SIMD vector according to the mask
    unsafe fn load_by_mask(&self, mask: u8) -> [MaybeUninit<Simd>; 8] {
        // SAFETY: MaybeUninit is always initialized
        let mut simds: [MaybeUninit<Simd>; 8] = MaybeUninit::uninit().assume_init();

        Self::foreach_mask(mask, |idx| {
            // SAFETY: 0 <= idx < 8
            simds
                .get_unchecked_mut(idx)
                .write(Simd::new(self.1.get_unchecked(idx).clone()));
        });

        simds
    }

    /// Returns number of consecutive leading digits/letters for each input.
    unsafe fn count_leading_digits_and_letters(&self) -> [(u8, u8); 8] {
        let (digits, letters) = self.count_leading_digits_and_letters_mask();
        let loaded = self.load_by_mask(digits | letters);
        Self::count_leading_digits_and_letters_with_potential((digits, letters), &loaded)
    }

    /// Returns whether consecutive leading digits/letters are at least 4 for
    /// each input, in bit mask form.
    #[target_feature(enable = "avx,avx2")]
    unsafe fn count_leading_digits_and_letters_mask(&self) -> (u8, u8) {
        debug_assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2"));

        use_intrinsic! {
            _mm256_cmpeq_epi32, _mm256_cmpgt_epi8, _mm256_set1_epi8, _mm256_setzero_si256,
        }

        let epi8_mask = _mm256_cmpgt_epi8(self.0, _mm256_set1_epi8(0x09u8 as i8));

        let digits =
            Self::gather_all_one_dword_mask(_mm256_cmpeq_epi32(epi8_mask, _mm256_setzero_si256()));
        let letters = Self::gather_all_one_dword_mask(_mm256_cmpeq_epi32(
            epi8_mask,
            _mm256_set1_epi8(0xFFu8 as i8),
        ));

        (digits, letters)
    }

    /// Returns number of consecutive leading digits/letters for each input.
    ///
    /// # Safety
    /// The caller needs to make sure that `simd` is initialized at indices
    /// indicated by `digits` or `letters`
    unsafe fn count_leading_digits_and_letters_with_potential(
        (digits, letters): (u8, u8),
        simd: &[MaybeUninit<Simd>; 8],
    ) -> [(u8, u8); 8] {
        let mut r = [(0, 0); 8];

        Self::foreach_mask(digits, |idx| {
            r.get_unchecked_mut(idx).0 =
                simd.get_unchecked(idx).assume_init().count_leading_digits()
        });

        Self::foreach_mask(letters, |idx| {
            r.get_unchecked_mut(idx).1 = simd
                .get_unchecked(idx)
                .assume_init()
                .count_leading_letters()
        });

        r
    }

    /// Returns number of consecutive same nibbles from the start for each
    /// input.
    unsafe fn count_leading_homogeneous_simd(&self) -> [u8; 8] {
        let homogeneous = self.count_leading_homogeneous_mask();
        let loaded = self.load_by_mask(homogeneous);
        Self::count_leading_homogeneous_with_potential(homogeneous, &loaded)
    }

    /// Returns whether consecutive same nibbles from the start are at least 4
    /// for each input, in bit mask form.
    #[target_feature(enable = "avx,avx2")]
    unsafe fn count_leading_homogeneous_mask(&self) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2"));

        use_intrinsic! {_mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_shuffle_epi8}

        // The mask to broadcast the first nibble in to each byte of a 32-bit integer
        const SHUFFLE_MASK: [u8; 32] = [
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, // lo-16 bytes
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, // hi-16 bytes
        ];

        Self::gather_all_one_dword_mask(_mm256_cmpeq_epi32(
            _mm256_shuffle_epi8(self.0, _mm256_loadu_si256(SHUFFLE_MASK.as_ptr().cast())),
            self.0,
        ))
    }

    /// Returns number of consecutive same nibbles from the start for each
    /// input.
    ///
    /// # Safety
    /// The caller needs to make sure that `simd` is initialized at indices
    /// indicated by `homogeneous`
    fn count_leading_homogeneous_with_potential(
        homogeneous: u8,
        simds: &[MaybeUninit<Simd>; 8],
    ) -> [u8; 8] {
        Self::delegate_by_mask(homogeneous, simds, Simd::count_leading_homogeneous)
    }

    /// Returns number of consecutive nibbles equal to mathematical constant `e`
    /// from the start for each input.
    unsafe fn count_longest_prefix_e_simd(&self) -> [u8; 8] {
        let leading_e = self.count_longest_prefix_single_mask(Self::MATH_E);
        let loaded = self.load_by_mask(leading_e);
        Self::count_longest_prefix_e_with_potential(leading_e, &loaded)
    }

    /// Returns number of consecutive nibbles equal to mathematical constant `e`
    /// for each input.
    ///
    /// # Safety
    /// The caller needs to make sure that `simd` is initialized at indices
    /// indicated by `leading_e`
    fn count_longest_prefix_e_with_potential(
        leading_e: u8,
        simds: &[MaybeUninit<Simd>; 8],
    ) -> [u8; 8] {
        Self::delegate_by_mask(leading_e, simds, Simd::count_longest_prefix_e)
    }

    /// Returns number of consecutive nibbles equal to mathematical constant `π`
    /// from the start for each input.
    #[target_feature(enable = "avx,avx2")]
    unsafe fn count_longest_prefix_pi_simd(&self) -> [u8; 8] {
        let leading_pi = self.count_longest_prefix_single_mask(Self::MATH_PI);
        let loaded = self.load_by_mask(leading_pi);
        Self::count_longest_prefix_pi_with_potential(leading_pi, &loaded)
    }

    /// Returns number of consecutive nibbles equal to mathematical constant `π`
    /// for each input.
    ///
    /// # Safety
    /// The caller needs to make sure that `simd` is initialized at indices
    /// indicated by `leading_pi`
    fn count_longest_prefix_pi_with_potential(
        leading_pi: u8,
        simds: &[MaybeUninit<Simd>; 8],
    ) -> [u8; 8] {
        Self::delegate_by_mask(leading_pi, simds, Simd::count_longest_prefix_pi)
    }

    /// Returns whether consecutive nibbles equal to `prefix` from the start are
    /// at least 4 for each input, in bit mask form.
    #[target_feature(enable = "avx,avx2")]
    unsafe fn count_longest_prefix_single_mask(&self, prefix: i32) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2"));

        use_intrinsic! {_mm256_cmpeq_epi32, _mm256_set1_epi32}

        let prefix = _mm256_set1_epi32(prefix);

        Self::gather_all_one_dword_mask(_mm256_cmpeq_epi32(self.0, prefix))
    }

    /// Returns number of consecutive nibbles equal to `other` from the start
    /// for each input.
    #[target_feature(enable = "avx2")]
    unsafe fn count_longest_prefix_simd(&self, other: &Self) -> [u8; 8] {
        debug_assert!(is_x86_feature_detected!("avx2"));

        use_intrinsic! {_mm256_cmpeq_epi32}

        let prefix = Self::gather_all_one_dword_mask(_mm256_cmpeq_epi32(self.0, other.0));

        let loaded_self = self.load_by_mask(prefix);
        let loaded_other = other.load_by_mask(prefix);

        let mut r = [0; 8];
        Self::foreach_mask(prefix, |idx| {
            *r.get_unchecked_mut(idx) = loaded_self
                .get_unchecked(idx)
                .assume_init_ref()
                .count_longest_prefix(loaded_other.get_unchecked(idx).assume_init_ref())
        });
        r
    }

    /// Returns packed mask of 32-bit integers in a SIMD vector such that the
    /// highest bit is 1.
    #[target_feature(enable = "avx")]
    unsafe fn gather_all_one_dword_mask(x: __m256i) -> u8 {
        debug_assert!(is_x86_feature_detected!("avx"));

        use_intrinsic! {_mm256_cvtepi32_ps, _mm256_movemask_ps}

        _mm256_movemask_ps(_mm256_cvtepi32_ps(x)) as u8
    }

    /// Given a compressed indices vector returned by
    /// [`Self::gather_all_one_dword_mask`], call `f` for the indices of 1's.
    #[inline(always)]
    fn foreach_mask<F>(mask: u8, mut f: F)
    where
        F: FnMut(usize),
    {
        // SAFETY: 0<=x < 256, Self::LUT_ALL_ONE.len() == 256
        let mut indices = unsafe { *Self::LUT_ALL_ONE.get_unchecked(mask as usize) };
        while indices != 0 {
            let idx = (indices & 0xF) as usize - 1;
            f(idx);
            indices >>= 4;
        }
    }

    /// Given a compressed indices vector returned by
    /// [`Self::gather_all_one_dword_mask`], use `f` as the value for those
    /// positions, and leave 0 otherwise.
    #[inline(always)]
    fn delegate_by_mask<F>(mask: u8, simds: &[MaybeUninit<Simd>; 8], f: F) -> [u8; 8]
    where
        F: Fn(&Simd) -> u8,
    {
        let mut r = [0; 8];
        Self::foreach_mask(mask, |idx| {
            unsafe { *r.get_unchecked_mut(idx) = f(simds.get_unchecked(idx).assume_init_ref()) };
        });
        r
    }
}

impl NibblesBatch<8> for SimdLossy8 {
    fn count_leading_digits_batch(x: [[u8; 16]; 8]) -> [u8; 8] {
        unsafe {
            Self::from(x)
                .count_leading_digits_and_letters()
                .map(|(d, _)| d)
        }
    }

    fn count_leading_letters_batch(x: [[u8; 16]; 8]) -> [u8; 8] {
        unsafe {
            Self::from(x)
                .count_leading_digits_and_letters()
                .map(|(_, l)| l)
        }
    }

    fn count_leading_homogeneous_batch(x: [[u8; 16]; 8]) -> [u8; 8] {
        unsafe { Self::from(x).count_leading_homogeneous_simd() }
    }

    fn count_longest_prefix_e_batch(x: [[u8; 16]; 8]) -> [u8; 8] {
        unsafe { Self::from(x).count_longest_prefix_e_simd() }
    }

    fn count_longest_prefix_pi_batch(x: [[u8; 16]; 8]) -> [u8; 8] {
        unsafe { Self::from(x).count_longest_prefix_pi_simd() }
    }

    fn count_longest_prefix_batch(x: [[u8; 16]; 8], y: [[u8; 16]; 8]) -> [u8; 8] {
        unsafe { Self::from(x).count_longest_prefix_simd(&Self::from(y)) }
    }

    fn compute_nicety_batch(x: [[u8; 16]; 8]) -> [Nicety; 8] {
        let x = Self::from(x);

        let mut r = [Nicety::default(); 8];
        unsafe {
            let (digits, letters) = x.count_leading_digits_and_letters_mask();
            let homogeneous = x.count_leading_homogeneous_mask();
            let leading_e = x.count_longest_prefix_single_mask(Self::MATH_E);
            let leading_pi = x.count_longest_prefix_single_mask(Self::MATH_PI);

            let loaded = x.load_by_mask(digits | letters | homogeneous | leading_e | leading_pi);

            let digits_and_letters =
                Self::count_leading_digits_and_letters_with_potential((digits, letters), &loaded);
            let homogeneous = Self::count_leading_homogeneous_with_potential(homogeneous, &loaded);
            let leading_e = Self::count_longest_prefix_e_with_potential(leading_e, &loaded);
            let leading_pi = Self::count_longest_prefix_pi_with_potential(leading_pi, &loaded);

            for i in 0..8 {
                r[i] = Nicety {
                    digits: digits_and_letters[i].0,
                    letters: digits_and_letters[i].1,
                    homogeneous: homogeneous[i],
                    leading_e: leading_e[i],
                    leading_pi: leading_pi[i],
                }
            }
        }
        r
    }
}

impl From<[[u8; 16]; 8]> for SimdLossy8 {
    fn from(x: [[u8; 16]; 8]) -> Self {
        unsafe { Self::new(x) }
    }
}

#[cfg(all(
    test,
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "bmi1",
    target_feature = "sse2"
))]
mod tests {
    use super::*;
    use crate::tests::BatchToSingleHarness;

    type T = BatchToSingleHarness<8, SimdLossy8>;

    #[test]
    fn count_leading_digits() {
        crate::tests::count_leading_digits::<T>(4);
    }

    #[test]
    fn count_leading_letters() {
        crate::tests::count_leading_letters::<T>(4);
    }
}
