use std::mem::transmute;

use md5::{Digest, Md5};
use rand::Rng;

use crate::{EmbarrisinglyParallel, Nibbles};

/// An array of 32 nibbles (4-byte).
///
/// Each nibble is represented by a byte, only lower 4 bits are used.
///
/// The algorithms implemented here are naïvely iterating the nibbles one by
/// one.
#[derive(Debug, Clone, Copy)]
pub struct Baseline([u8; 32]);

impl Baseline {
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

    /// Count the number of consecutive digits (in range `0x00..=0x09`) from the
    /// start, assuming the first `SKIP` nibbles are digits.
    pub(crate) fn count_leading_digits_skipping<const SKIP: usize>(&self) -> u8 {
        self.0.iter().skip(SKIP).take_while(|&&b| b < 0x0A).count() as u8 + SKIP as u8
    }

    /// Count the number of consecutive letters (in range `0x0A..=0x0F`) from
    /// the start, assuming the first `SKIP` nibbles are letters.
    pub(crate) fn count_leading_letters_skipping<const SKIP: usize>(&self) -> u8 {
        self.0.iter().skip(SKIP).take_while(|&&b| b >= 0x0A).count() as u8 + SKIP as u8
    }

    /// Count the number of consecutive nibbles equal to the first nibble from
    /// the start, assuming the first `SKIP` nibbles are equal to the first
    /// nibble.
    pub(crate) fn count_leading_homogenous_skipping<const SKIP: usize>(&self) -> u8 {
        self.0
            .iter()
            .skip(SKIP)
            .take_while(|&&b| b == self.0[0])
            .count() as u8
            + SKIP as u8
    }

    /// Count the length of the longest common prefix between `self` against
    /// mathematical constant `e`, assuming the first `SKIP` nibbles are the
    /// same.
    pub(crate) fn count_longest_prefix_e_skipping<const SKIP: usize>(&self) -> u8 {
        self.0
            .iter()
            .skip(SKIP)
            .zip(Self::MATH_E.iter().skip(SKIP))
            .take_while(|(a, b)| a == b)
            .count() as u8
            + SKIP as u8
    }

    /// Count the length of the longest common prefix between `self` against
    /// mathematical constant `π`, assuming the first `SKIP` nibbles are the
    /// same.
    pub(crate) fn count_longest_prefix_pi_skipping<const SKIP: usize>(&self) -> u8 {
        self.0
            .iter()
            .skip(SKIP)
            .zip(Self::MATH_PI.iter().skip(SKIP))
            .take_while(|(a, b)| a == b)
            .count() as u8
            + SKIP as u8
    }

    /// Count the length of the longest common prefix between `self` and
    /// `other`, assuming the first `SKIP` nibbles are the same.
    pub(crate) fn count_longest_prefix_skipping<const SKIP: usize>(&self, other: &Self) -> u8 {
        self.0
            .iter()
            .skip(SKIP)
            .zip(other.0.iter().skip(SKIP))
            .take_while(|(a, b)| a == b)
            .count() as u8
            + SKIP as u8
    }
}

impl Nibbles for Baseline {
    fn count_leading_digits(&self) -> u8 {
        self.count_leading_digits_skipping::<0>()
    }

    fn count_leading_letters(&self) -> u8 {
        self.count_leading_letters_skipping::<0>()
    }

    fn count_leading_homogenous(&self) -> u8 {
        self.count_leading_homogenous_skipping::<1>()
    }

    fn count_longest_prefix_e(&self) -> u8 {
        self.count_longest_prefix_e_skipping::<0>()
    }

    fn count_longest_prefix_pi(&self) -> u8 {
        self.count_longest_prefix_pi_skipping::<0>()
    }

    fn count_longest_prefix(&self, other: &Self) -> u8 {
        self.count_longest_prefix_skipping::<0>(other)
    }
}

/// Convert 16 bytes to 32 niblles.
///
/// Each byte `b` will be splited as `[b >> 4, b & 0xF]`.
impl From<[u8; 16]> for Baseline {
    fn from(x: [u8; 16]) -> Self {
        let nibbles = x.map(|b| [b >> 4, b & 0xF]);
        // SAFETY: size_of::<[[u8; 2]; 16]>() == size_of::<[u8; 32]>()
        Self(unsafe { transmute(nibbles) })
    }
}

impl EmbarrisinglyParallel for Baseline {}

/// Generate `N` 16-byte array, where each byte is an ASCII character [0-9a-z].
/// For maximum performance, the implementation does not guarantee uniformness
/// of the generated strings.
pub fn generate_strs<R: Rng, const N: usize>(rng: &mut R) -> [[u8; 32]; N] {
    const POOL: [u8; 36] = [
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, // 0-9
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        116, 117, 118, 119, 120, 121, 122, //a-z
    ];
    [0; N].map(|_| {
        let v: [u8; 32] = unsafe {
            transmute([
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ])
        };
        v.map(|b| POOL[(b % 36) as usize])
    })
}

/// Compute `N` MD5 digests according to the input.
pub fn digest_md5s<const N: usize>(srcs: [[u8; 32]; N]) -> [[u8; 16]; N] {
    srcs.map(|v| Md5::digest(v.as_slice()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u8_array() {
        let x: Baseline = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ]
        .into();
        assert_eq!(
            x.0,
            [
                0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
                0xf, 0xe, 0xd, 0xc, 0xb, 0xa, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0
            ]
        );
    }

    #[test]
    fn count_leading_digits() {
        crate::tests::count_leading_digits::<Baseline>(0);
    }

    #[test]
    fn count_leading_letters() {
        crate::tests::count_leading_letters::<Baseline>(0);
    }
}
