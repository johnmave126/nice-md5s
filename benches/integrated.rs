use std::{fmt::Display, time::Duration};

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput::Elements,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use nice_md5s::x86::{self, Simd};
use nice_md5s::{baseline, baseline::Baseline, NibblesBatch, Nicety};
use rand::{rngs::SmallRng, SeedableRng};

struct Schedule {
    total: usize,
    str: usize,
    digest: usize,
    nicety: usize,
}

impl Display for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}x{} -> {}x{} -> {}x{}",
            self.str,
            self.total / self.str,
            self.digest,
            self.total / self.digest,
            self.nicety,
            self.total / self.nicety
        ))
    }
}

fn bench_baseline<
    const TOTAL: usize,
    const STR: usize,
    const DIGEST: usize,
    const NICETY: usize,
>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    let mut rng = SmallRng::from_entropy();
    group.throughput(Elements(TOTAL as u64));
    assert_eq!(TOTAL % STR, 0);
    assert_eq!(TOTAL % DIGEST, 0);
    assert_eq!(TOTAL % NICETY, 0);

    let shedule = Schedule {
        total: TOTAL,
        str: STR,
        digest: DIGEST,
        nicety: NICETY,
    };

    group.bench_function(BenchmarkId::new("Baseline", shedule), |b| {
        b.iter(|| {
            let mut srcs = [[0; 32]; TOTAL];
            for chunk in srcs.chunks_exact_mut(STR) {
                chunk.copy_from_slice(&baseline::generate_strs::<_, STR>(&mut rng));
            }
            let mut digests = [[0; 16]; TOTAL];
            for (src, digest) in srcs
                .chunks_exact(DIGEST)
                .zip(digests.chunks_exact_mut(DIGEST))
            {
                digest.copy_from_slice(&baseline::digest_md5s::<DIGEST>(src.try_into().unwrap()));
            }
            let mut niceties = [Nicety::default(); TOTAL];
            for (digest, nicety) in digests
                .chunks_exact(NICETY)
                .zip(niceties.chunks_exact_mut(NICETY))
            {
                nicety.copy_from_slice(&Baseline::compute_nicety_batch(
                    TryInto::<[[u8; 16]; NICETY]>::try_into(digest).unwrap(),
                ));
            }
            niceties
        })
    });
}

fn bench_simd<const TOTAL: usize, const STR: usize, const DIGEST: usize, const NICETY: usize>(
    group: &mut BenchmarkGroup<WallTime>,
) {
    let mut rng = SmallRng::from_entropy();
    group.throughput(Elements(TOTAL as u64));
    assert_eq!(TOTAL % STR, 0);
    assert_eq!(TOTAL % DIGEST, 0);
    assert_eq!(TOTAL % NICETY, 0);

    let shedule = Schedule {
        total: TOTAL,
        str: STR,
        digest: DIGEST,
        nicety: NICETY,
    };

    group.bench_function(BenchmarkId::new("SIMD + ASSEMBLY + SIMD", shedule), |b| {
        b.iter(|| {
            let mut srcs = [[0; 32]; TOTAL];
            for chunk in srcs.chunks_exact_mut(STR) {
                chunk.copy_from_slice(&x86::generate_strs::<_, STR>(&mut rng));
            }
            let mut digests = [[0; 16]; TOTAL];
            for (src, digest) in srcs
                .chunks_exact(DIGEST)
                .zip(digests.chunks_exact_mut(DIGEST))
            {
                digest.copy_from_slice(&x86::digest_md5s::<DIGEST>(src.try_into().unwrap()));
            }
            let mut niceties = [Nicety::default(); TOTAL];
            for (digest, nicety) in digests
                .chunks_exact(NICETY)
                .zip(niceties.chunks_exact_mut(NICETY))
            {
                nicety.copy_from_slice(&Simd::compute_nicety_batch(
                    TryInto::<[[u8; 16]; NICETY]>::try_into(digest).unwrap(),
                ));
            }
            niceties
        })
    });

    if DIGEST == 8 {
        let shedule = Schedule {
            total: TOTAL,
            str: STR,
            digest: 8,
            nicety: NICETY,
        };
        group.bench_function(BenchmarkId::new("SIMD", shedule), |b| {
            b.iter(|| {
                let mut srcs = [[0; 32]; TOTAL];
                for chunk in srcs.chunks_exact_mut(STR) {
                    chunk.copy_from_slice(&x86::generate_strs::<_, STR>(&mut rng));
                }
                let mut digests = [[0; 16]; TOTAL];
                for (src, digest) in srcs.chunks_exact(8).zip(digests.chunks_exact_mut(8)) {
                    digest.copy_from_slice(&x86::digest_md5s_simd(src.try_into().unwrap()));
                }
                let mut niceties = [Nicety::default(); TOTAL];
                for (digest, nicety) in digests
                    .chunks_exact(NICETY)
                    .zip(niceties.chunks_exact_mut(NICETY))
                {
                    nicety.copy_from_slice(&Simd::compute_nicety_batch(
                        TryInto::<[[u8; 16]; NICETY]>::try_into(digest).unwrap(),
                    ));
                }
                niceties
            })
        });
    }
}

fn bench_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("Integrated");

    bench_baseline::<1, 1, 1, 1>(&mut group);
    bench_baseline::<8, 8, 8, 8>(&mut group);
    bench_baseline::<8, 8, 8, 1>(&mut group);
    bench_baseline::<8, 1, 8, 1>(&mut group);
    bench_baseline::<16, 16, 16, 16>(&mut group);
    bench_baseline::<16, 16, 16, 1>(&mut group);
    bench_baseline::<16, 8, 16, 1>(&mut group);
    bench_baseline::<16, 8, 8, 1>(&mut group);
    bench_baseline::<16, 16, 8, 1>(&mut group);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("sse2")
    {
        bench_simd::<1, 1, 1, 1>(&mut group);
        bench_simd::<8, 8, 8, 8>(&mut group);
        bench_simd::<8, 8, 8, 1>(&mut group);
        bench_simd::<8, 1, 8, 1>(&mut group);
        bench_simd::<16, 16, 16, 16>(&mut group);
        bench_simd::<16, 16, 16, 1>(&mut group);
        bench_simd::<16, 8, 16, 1>(&mut group);
        bench_simd::<16, 8, 8, 1>(&mut group);
        bench_simd::<16, 16, 8, 1>(&mut group);
    }
    group.finish();
}

criterion_group!(
    name    = integrated;
    config  = Criterion::default()
              .noise_threshold(0.05)
              .sample_size(2000)
              .measurement_time(Duration::from_secs(10));
    targets = bench_all
);
criterion_main!(integrated);
