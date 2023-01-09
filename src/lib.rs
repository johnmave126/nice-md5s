pub mod baseline;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

/// Nicety of a 128-bit value.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nicety {
    /// Number of consecutive leading digits (0-9)
    pub digits: u8,
    /// Number of consecutive leading letters (a-f)
    pub letters: u8,
    /// Number of consecutive leading nibbles equal to the first nibble
    pub homogeneous: u8,
    /// Number of consecutive leading nibbles matching `e`
    pub leading_e: u8,
    /// Number of consecutive leading nibbles matching `π`
    pub leading_pi: u8,
}

/// Trait representing an array of 32 nibbles (4-byte), providing methods to
/// compute nicety.
pub trait Nibbles {
    /// Count the number of consecutive digits (in range `0x00..=0x09`) from the
    /// start.
    ///
    /// # Example
    /// ```
    /// # use nice_md5s::{baseline::Baseline, Nibbles};
    /// let nibbles: Baseline = [
    ///     0x10u8, 0x22, 0x93, 0x4a, // 7 leading consecutive digits
    ///     0xaa, 0xaa, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
    /// ]
    /// .into();
    /// assert_eq!(nibbles.count_leading_digits(), 7);
    /// ```
    fn count_leading_digits(&self) -> u8;

    /// Count the number of consecutive letters (in range `0x0A..=0x0F`) from
    /// the start.
    ///
    /// # Example
    /// ```
    /// # use nice_md5s::{baseline::Baseline, Nibbles};
    /// let nibbles: Baseline = [
    ///     0xabu8, 0xcd, 0xef, 0xa9, // 7 leading consecutive letters
    ///     0x00, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
    /// ]
    /// .into();
    /// assert_eq!(nibbles.count_leading_letters(), 7);
    /// ```
    fn count_leading_letters(&self) -> u8;

    /// Count the number of consecutive nibbles equal to the first nibble from
    /// the start.
    ///
    /// # Example
    /// ```
    /// # use nice_md5s::{baseline::Baseline, Nibbles};
    /// let nibbles: Baseline = [
    ///     0x22u8, 0x22, 0x22, 0x29, // 7 leading consecutive same nibbles
    ///     0x00, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
    /// ]
    /// .into();
    /// assert_eq!(nibbles.count_leading_homogeneous(), 7);
    /// ```
    fn count_leading_homogeneous(&self) -> u8;

    /// Count the length of the longest common prefix between `self` and the
    /// [mathematical constant `e`](https://en.wikipedia.org/wiki/E_(mathematical_constant)).
    ///
    /// # Example
    /// ```
    /// # use nice_md5s::{baseline::Baseline, Nibbles};
    /// let nibbles: Baseline = [
    ///     0x27u8, 0x18, 0x28, 0x19, // 7 leading consecutive `e`
    ///     0x00, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
    /// ]
    /// .into();
    /// assert_eq!(nibbles.count_longest_prefix_e(), 7);
    /// ```
    fn count_longest_prefix_e(&self) -> u8;

    /// Count the length of the longest common prefix between `self` and the
    /// [mathematical constant `π`](https://en.wikipedia.org/wiki/Pi).
    ///
    /// # Example
    /// ```
    /// # use nice_md5s::{baseline::Baseline, Nibbles};
    /// let nibbles: Baseline = [
    ///     0x31u8, 0x41, 0x59, 0x29, // 7 leading consecutive `π`
    ///     0x00, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
    /// ]
    /// .into();
    /// assert_eq!(nibbles.count_longest_prefix_pi(), 7);
    /// ```
    fn count_longest_prefix_pi(&self) -> u8;

    /// Count the length of the longest common prefix between `self` and
    /// `other`.
    ///
    /// # Example
    /// ```
    /// # use nice_md5s::{baseline::Baseline, Nibbles};
    /// let a: Baseline = [
    ///     0x12u8, 0x34, 0x56, 0x78, // same leading 7 nibbles
    ///     0x00, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
    /// ]
    /// .into();
    /// let b: Baseline = [
    ///     0x12u8, 0x34, 0x56, 0x79, // same leading 7 nibbles
    ///     0x00, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
    /// ]
    /// .into();
    /// assert_eq!(a.count_longest_prefix(&b), 7);
    /// ```
    fn count_longest_prefix(&self, other: &Self) -> u8;

    /// Compute nice pattern values for an 128-bit byte strings.
    fn compute_nicety(&self) -> Nicety {
        let digits = self.count_leading_digits();
        let letters = self.count_leading_letters();
        let homogeneous = self.count_leading_homogeneous();
        let leading_e = self.count_longest_prefix_e();
        let leading_pi = self.count_longest_prefix_pi();
        Nicety {
            digits,
            letters,
            homogeneous,
            leading_e,
            leading_pi,
        }
    }
}
/// Trait representing `N` array of 32 nibbles (4-byte), providing methods to
/// compute niceties in batch.
pub trait NibblesBatch<const N: usize> {
    /// Count the number of consecutive digits (in range `0x00..=0x09`) from the
    /// start in batch.
    fn count_leading_digits_batch(x: [[u8; 16]; N]) -> [u8; N];

    /// Count the number of consecutive letters (in range `0x0A..=0x0F`) from
    /// the start in batch.
    fn count_leading_letters_batch(x: [[u8; 16]; N]) -> [u8; N];

    /// Count the number of consecutive nibbles equal to the first nibble from
    /// the start in batch.
    fn count_leading_homogeneous_batch(x: [[u8; 16]; N]) -> [u8; N];

    /// Count the length of the longest common prefix between `x` and the
    /// [mathematical constant `e`](https://en.wikipedia.org/wiki/E_(mathematical_constant)) in batch.
    fn count_longest_prefix_e_batch(x: [[u8; 16]; N]) -> [u8; N];

    /// Count the length of the longest common prefix between `x` and the
    /// [mathematical constant `π`](https://en.wikipedia.org/wiki/Pi) in batch.
    fn count_longest_prefix_pi_batch(x: [[u8; 16]; N]) -> [u8; N];

    /// Count the length of the longest common prefix between `x` and
    /// `y` in batch.
    fn count_longest_prefix_batch(x: [[u8; 16]; N], y: [[u8; 16]; N]) -> [u8; N];

    /// Compute the
    fn compute_nicety_batch(x: [[u8; 16]; N]) -> [Nicety; N];
}

pub(crate) trait EmbarrisinglyParallel {}

impl<T, const N: usize> NibblesBatch<N> for T
where
    T: Nibbles + From<[u8; 16]> + EmbarrisinglyParallel,
{
    fn count_leading_digits_batch(x: [[u8; 16]; N]) -> [u8; N] {
        x.map(|v| T::from(v).count_leading_digits())
    }

    fn count_leading_letters_batch(x: [[u8; 16]; N]) -> [u8; N] {
        x.map(|v| T::from(v).count_leading_letters())
    }

    fn count_leading_homogeneous_batch(x: [[u8; 16]; N]) -> [u8; N] {
        x.map(|v| T::from(v).count_leading_homogeneous())
    }

    fn count_longest_prefix_e_batch(x: [[u8; 16]; N]) -> [u8; N] {
        x.map(|v| T::from(v).count_longest_prefix_e())
    }

    fn count_longest_prefix_pi_batch(x: [[u8; 16]; N]) -> [u8; N] {
        x.map(|v| T::from(v).count_longest_prefix_pi())
    }

    fn count_longest_prefix_batch(x: [[u8; 16]; N], y: [[u8; 16]; N]) -> [u8; N] {
        let mut r = [0; N];
        for i in 0..N {
            r[i] = T::from(x[i]).count_longest_prefix(&T::from(y[i]));
        }
        r
    }

    fn compute_nicety_batch(x: [[u8; 16]; N]) -> [Nicety; N] {
        x.map(|v| T::from(v).compute_nicety())
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use super::*;

    pub(crate) struct BatchToSingleHarness<const N: usize, T>([u8; 16], PhantomData<T>)
    where
        T: NibblesBatch<N>;

    impl<const N: usize, T> BatchToSingleHarness<N, T>
    where
        T: NibblesBatch<N>,
    {
        #[inline(always)]
        fn test_single<F>(&self, f: F) -> u8
        where
            F: Fn([[u8; 16]; N]) -> [u8; N],
        {
            // Test all at the same time
            let all = f([self.0; N]);

            for i in 1..N {
                assert_eq!(all[0], all[i]);
            }

            // Test each individually
            for i in 0..N {
                let mut input = [[0; 16]; N];
                input[i] = self.0;
                assert_eq!(f(input)[i], all[0]);
            }

            all[0]
        }
    }

    impl<const N: usize, T> Nibbles for BatchToSingleHarness<N, T>
    where
        T: NibblesBatch<N>,
    {
        fn count_leading_digits(&self) -> u8 {
            self.test_single(T::count_leading_digits_batch)
        }

        fn count_leading_letters(&self) -> u8 {
            self.test_single(T::count_leading_letters_batch)
        }

        fn count_leading_homogeneous(&self) -> u8 {
            self.test_single(T::count_leading_homogeneous_batch)
        }

        fn count_longest_prefix_e(&self) -> u8 {
            self.test_single(T::count_longest_prefix_e_batch)
        }

        fn count_longest_prefix_pi(&self) -> u8 {
            self.test_single(T::count_longest_prefix_pi_batch)
        }

        fn count_longest_prefix(&self, other: &Self) -> u8 {
            let result = T::count_longest_prefix_batch([self.0; N], [other.0; N]);

            for i in 1..N {
                assert_eq!(result[0], result[i])
            }

            result[0]
        }
    }

    impl<const N: usize, T: NibblesBatch<N>> From<[u8; 16]> for BatchToSingleHarness<N, T> {
        fn from(value: [u8; 16]) -> Self {
            Self(value, PhantomData)
        }
    }

    pub(crate) fn count_leading_digits<T: Nibbles + From<[u8; 16]>>(slack: u8) {
        let test_cases = [
            // No digit
            (
                [
                    0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                0,
            ),
            (
                [
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff,
                ],
                0,
            ),
            (
                [
                    0xa0, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                0,
            ),
            (
                [
                    0xa1, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                0,
            ),
            (
                [
                    0xa9, 0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                0,
            ),
            // 1 digit
            (
                [
                    0x9a, 0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                1,
            ),
            (
                [
                    0x0a, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                1,
            ),
            (
                [
                    0x1a, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                1,
            ),
            // 2 digits
            (
                [
                    0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                2,
            ),
            (
                [
                    0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                2,
            ),
            (
                [
                    0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                2,
            ),
            // 4 digits
            (
                [
                    0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                4,
            ),
            (
                [
                    0x99, 0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                4,
            ),
            (
                [
                    0x11, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                4,
            ),
            // 5 digits
            (
                [
                    0x00, 0x00, 0x0a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                5,
            ),
            (
                [
                    0x99, 0x99, 0x9a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                5,
            ),
            (
                [
                    0x11, 0x11, 0x1a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                5,
            ),
            // 32 digits
            (
                [
                    0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
                    0x11, 0x11, 0x11,
                ],
                32,
            ),
            (
                [
                    0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
                    0x99, 0x99, 0x99,
                ],
                32,
            ),
            (
                [
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ],
                32,
            ),
        ];

        for (v, expected) in test_cases.into_iter() {
            let x = T::from(v);
            if expected < slack {
                assert!(x.count_leading_digits() <= expected);
            } else {
                assert_eq!(x.count_leading_digits(), expected);
            }
        }
    }

    pub(crate) fn count_leading_letters<T: Nibbles + From<[u8; 16]>>(slack: u8) {
        let test_cases = [
            // no letters
            (
                [
                    0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
                    0x99, 0x99, 0x99,
                ],
                0,
            ),
            (
                [
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00,
                ],
                0,
            ),
            (
                [
                    0x9a, 0xaa, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
                    0x99, 0x99, 0x99,
                ],
                0,
            ),
            (
                [
                    0x0f, 0xff, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x99,
                    0x99, 0x99, 0x99,
                ],
                0,
            ),
            // 1 letter
            (
                [
                    0xa1, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                1,
            ),
            (
                [
                    0xf9, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff,
                ],
                1,
            ),
            (
                [
                    0xa0, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                1,
            ),
            (
                [
                    0xa1, 0x11, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                1,
            ),
            (
                [
                    0xa9, 0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                1,
            ),
            // 2 letters
            (
                [
                    0xaa, 0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                2,
            ),
            (
                [
                    0xff, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                2,
            ),
            (
                [
                    0xab, 0x0a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                2,
            ),
            // 4 letters
            (
                [
                    0xaa, 0xaa, 0x0a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                4,
            ),
            (
                [
                    0xff, 0xaa, 0x9a, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                4,
            ),
            (
                [
                    0xff, 0xaa, 0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                4,
            ),
            // 5 letters
            (
                [
                    0xaa, 0xaa, 0xa9, 0x99, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                5,
            ),
            (
                [
                    0xff, 0xff, 0xf0, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                5,
            ),
            (
                [
                    0xbb, 0xaa, 0xa5, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                5,
            ),
            // all letters
            (
                [
                    0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                    0xaa, 0xaa, 0xaa,
                ],
                32,
            ),
            (
                [
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff,
                ],
                32,
            ),
        ];

        for (v, expected) in test_cases.into_iter() {
            let x = T::from(v);
            if expected < slack {
                assert!(
                    x.count_leading_letters() <= expected,
                    "testing leading letter of {}",
                    hex::encode(v)
                );
            } else {
                assert_eq!(
                    x.count_leading_letters(),
                    expected,
                    "testing leading letter of {}",
                    hex::encode(v)
                );
            }
        }
    }
}
