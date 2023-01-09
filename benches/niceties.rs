use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize::SmallInput, BenchmarkGroup,
    BenchmarkId, Criterion, Throughput::Elements,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use nice_md5s::x86::{Simd, SimdLossy8};
use nice_md5s::{baseline::Baseline, NibblesBatch};
use rand::{thread_rng, RngCore};

trait Runnable<const N: usize, Input, Output> {
    fn setup() -> Input;
    fn run(&mut self, x: Input) -> Output;
}

impl<const N: usize, T, Output> Runnable<N, [[u8; 16]; N], [Output; N]> for T
where
    T: FnMut([[u8; 16]; N]) -> [Output; N],
{
    fn setup() -> [[u8; 16]; N] {
        let mut rng = thread_rng();
        let mut rnd_bytes = [[0; 16]; N];
        for v in rnd_bytes.iter_mut() {
            rng.fill_bytes(v);
        }
        rnd_bytes
    }

    #[inline(always)]
    fn run(&mut self, x: [[u8; 16]; N]) -> [Output; N] {
        self(x)
    }
}

impl<const N: usize, T, Output> Runnable<N, ([[u8; 16]; N], [[u8; 16]; N]), [Output; N]> for T
where
    T: FnMut([[u8; 16]; N], [[u8; 16]; N]) -> [Output; N],
{
    fn setup() -> ([[u8; 16]; N], [[u8; 16]; N]) {
        let mut rng = thread_rng();
        let mut a = [[0; 16]; N];
        let mut b = [[0; 16]; N];
        for v in a.iter_mut() {
            rng.fill_bytes(v);
        }
        for v in b.iter_mut() {
            rng.fill_bytes(v);
        }
        (a, b)
    }

    #[inline(always)]
    fn run(&mut self, (x, y): ([[u8; 16]; N], [[u8; 16]; N])) -> [Output; N] {
        self(x, y)
    }
}

fn bench_function<const N: usize, F, I, O>(
    group: &mut BenchmarkGroup<WallTime>,
    fn_name: &str,
    mut f: F,
) where
    F: Copy + Runnable<N, I, O>,
{
    group.bench_function(BenchmarkId::new(fn_name, N), move |b| {
        b.iter_batched(F::setup, |i| f.run(i), SmallInput)
    });
}

macro_rules! bench_batch {
    ($N: literal, $group: expr, $fn: ident) => {
        $group.throughput(Elements($N as u64));

        bench_function::<$N, _, _, _>(&mut $group, "Baseline", Baseline::$fn);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("bmi1")
                && is_x86_feature_detected!("sse2")
            {
                bench_function::<$N, _, _, _>(&mut $group, "SIMD", Simd::$fn);
            }
            if $N == 8 && is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
                bench_function::<8, _, _, _>(&mut $group, "SIMD Lossy", SimdLossy8::$fn);
            }
        }
    };
}

macro_rules! make_bench {
    ($fn: ident, $title: literal) => {
        fn $fn(c: &mut Criterion) {
            let mut group = c.benchmark_group($title);

            bench_batch!(1, group, $fn);
            bench_batch!(2, group, $fn);
            bench_batch!(4, group, $fn);
            bench_batch!(8, group, $fn);
            bench_batch!(16, group, $fn);

            group.finish()
        }
    };
}

make_bench!(compute_nicety_batch, "Compute Nicety");
make_bench!(count_leading_digits_batch, "Count Leading Digits");
make_bench!(count_leading_letters_batch, "Count Leading Letters");
make_bench!(count_leading_homogeneous_batch, "Count Homogenous Prefix");
make_bench!(count_longest_prefix_e_batch, "Count `e` prefix");
make_bench!(count_longest_prefix_pi_batch, "Count `π` prefix");
make_bench!(count_longest_prefix_batch, "Count Common prefix");

criterion_group!(
    name    = niceties;
    config  = Criterion::default()
              .noise_threshold(0.05)
              .sample_size(2000)
              .measurement_time(Duration::from_secs(10));
    targets = compute_nicety_batch,
              count_leading_digits_batch,
              count_leading_letters_batch,
              count_leading_homogeneous_batch,
              count_longest_prefix_e_batch,
              count_longest_prefix_pi_batch,
              count_longest_prefix_batch
);
criterion_main!(niceties);
