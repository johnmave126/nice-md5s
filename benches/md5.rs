use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize::SmallInput, BenchmarkGroup,
    BenchmarkId, Criterion, Throughput::Elements,
};
use nice_md5s::baseline;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use nice_md5s::x86;
use rand::{rngs::SmallRng, SeedableRng};

fn bench_source_generation<const N: usize>(group: &mut BenchmarkGroup<WallTime>) {
    let mut rng = SmallRng::from_entropy();
    group.throughput(Elements(N as u64));
    group.bench_function(BenchmarkId::new("Baseline", N), |b| {
        b.iter(|| baseline::generate_strs::<_, N>(&mut rng))
    });

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse2") {
            group.bench_function(BenchmarkId::new("SIMD", N), |b| {
                b.iter(|| x86::generate_strs::<_, N>(&mut rng))
            });
        }
    }
}

fn bench_md5_digest<const N: usize>(group: &mut BenchmarkGroup<WallTime>) {
    let mut rng = SmallRng::from_entropy();
    group.throughput(Elements(N as u64));
    group.bench_function(BenchmarkId::new("Baseline", N), |b| {
        b.iter_batched(
            || baseline::generate_strs::<_, N>(&mut rng),
            baseline::digest_md5s,
            SmallInput,
        )
    });

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        group.bench_function(BenchmarkId::new("Assembly", N), |b| {
            b.iter_batched(
                || baseline::generate_strs::<_, N>(&mut rng),
                x86::digest_md5s,
                SmallInput,
            )
        });
        if N == 8 && is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            group.bench_function(BenchmarkId::new("SIMD", N), |b| {
                b.iter_batched(
                    || baseline::generate_strs::<_, 8>(&mut rng),
                    x86::digest_md5s_simd,
                    SmallInput,
                )
            });
        }
    }
}

fn bench_md5_generation<const N: usize>(group: &mut BenchmarkGroup<WallTime>) {
    let mut rng = SmallRng::from_entropy();
    group.throughput(Elements(N as u64));
    group.bench_function(BenchmarkId::new("Baseline", N), |b| {
        b.iter(|| baseline::digest_md5s(baseline::generate_strs::<_, N>(&mut rng)))
    });

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse2") {
            group.bench_function(BenchmarkId::new("SIMD + Assembly", N), |b| {
                b.iter(|| x86::digest_md5s(x86::generate_strs::<_, N>(&mut rng)))
            });
        }

        if N == 8 && is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
            group.bench_function(BenchmarkId::new("SIMD + SIMD", N), |b| {
                b.iter(|| x86::digest_md5s_simd(x86::generate_strs::<_, 8>(&mut rng)))
            });
        }
    }
}

fn bench_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("Source Generation");
    bench_source_generation::<1>(&mut group);
    bench_source_generation::<2>(&mut group);
    bench_source_generation::<4>(&mut group);
    bench_source_generation::<8>(&mut group);
    bench_source_generation::<16>(&mut group);
    bench_source_generation::<32>(&mut group);
    group.finish();

    let mut group = c.benchmark_group("MD5 Digest");
    bench_md5_digest::<1>(&mut group);
    bench_md5_digest::<2>(&mut group);
    bench_md5_digest::<4>(&mut group);
    bench_md5_digest::<8>(&mut group);
    bench_md5_digest::<16>(&mut group);
    bench_md5_digest::<32>(&mut group);
    group.finish();

    let mut group = c.benchmark_group("MD5 Generation");
    bench_md5_generation::<1>(&mut group);
    bench_md5_generation::<2>(&mut group);
    bench_md5_generation::<4>(&mut group);
    bench_md5_generation::<8>(&mut group);
    bench_md5_generation::<16>(&mut group);
    bench_md5_generation::<32>(&mut group);
    group.finish();
}

criterion_group!(
    name    = md5;
    config  = Criterion::default()
              .noise_threshold(0.05)
              .sample_size(2000)
              .measurement_time(Duration::from_secs(10));
    targets = bench_all
);
criterion_main!(md5);
