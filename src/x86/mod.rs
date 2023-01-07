mod generation;
pub mod simd;
pub mod simd_lossy;

pub use generation::{digest_md5s, digest_md5s_simd, generate_strs};
pub use simd::Simd;
pub use simd_lossy::SimdLossy8;

macro_rules! use_intrinsic {
    ($($item: tt), + $(,)?) => {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{$($item), +};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{$($item), +};
    };
}

pub(crate) use use_intrinsic;
