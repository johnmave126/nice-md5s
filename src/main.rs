use std::{
    cmp::max,
    fmt::Display,
    num::NonZeroUsize,
    str,
    sync::mpsc::{sync_channel, SyncSender},
    thread::{self, available_parallelism},
    time::{Duration, Instant},
};

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use nice_md5s::x86::{self, Simd};
use nice_md5s::{
    baseline::{self, Baseline},
    NibblesBatch, Nicety,
};
use rand::{
    rngs::{SmallRng, ThreadRng},
    thread_rng, SeedableRng,
};

const RENDER_FREQUENCY: Duration = Duration::from_secs(1);

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Number of threads to use
    #[arg(
        short,
        long,
        value_name = "NUM",
        default_value_t = available_parallelism().map(NonZeroUsize::get).unwrap_or(1)
    )]
    worker: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
struct MD5Pair {
    value: u8,
    src: [u8; 32],
    digest: [u8; 16],
}

impl Display for MD5Pair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{:>2}, src: \"{}\", digest: 0x",
            self.value,
            str::from_utf8(&self.src).unwrap()
        ))?;
        for b in self.digest.iter() {
            f.write_fmt(format_args!("{b:02x}"))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct Report {
    digits: MD5Pair,
    letters: MD5Pair,
    homogenous: MD5Pair,
    leading_e: MD5Pair,
    leading_pi: MD5Pair,
}

impl Report {
    fn new(niceties: &[Nicety], srcs: &[[u8; 32]], digests: &[[u8; 16]]) -> Self {
        Self {
            digits: Self::get_by(niceties, srcs, digests, |n| n.digits),
            letters: Self::get_by(niceties, srcs, digests, |n| n.letters),
            homogenous: Self::get_by(niceties, srcs, digests, |n| n.homogenous),
            leading_e: Self::get_by(niceties, srcs, digests, |n| n.leading_e),
            leading_pi: Self::get_by(niceties, srcs, digests, |n| n.leading_pi),
        }
    }

    fn get_by<F: Fn(&Nicety) -> u8>(
        niceties: &[Nicety],
        srcs: &[[u8; 32]],
        digests: &[[u8; 16]],
        key_fn: F,
    ) -> MD5Pair {
        niceties
            .iter()
            .zip(srcs.iter())
            .zip(digests.iter())
            .max_by_key(|((n, _), _)| key_fn(n))
            .map(|((n, s), d)| MD5Pair {
                value: key_fn(n),
                src: s.clone(),
                digest: d.clone(),
            })
            .unwrap()
    }
}

// Per thread schedule
const BLOCK: usize = 2048;
const STR_BLOCK: usize = 32;
const DIGEST_BLOCK: usize = 8;
const COMPUTE_BLOCK: usize = 1;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SIMD_DIGEST_BLOCK: usize = 8;

fn new_thread(report_send: &SyncSender<Report>, trng: &mut ThreadRng) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("sse2")
    {
        thread::spawn({
            let report_send = report_send.clone();
            let mut rng = SmallRng::from_rng(trng).expect("failed to initialize RNG");

            move || {
                loop {
                    let mut srcs = [[0; 32]; BLOCK];
                    for chunk in srcs.chunks_exact_mut(STR_BLOCK) {
                        chunk.copy_from_slice(&x86::generate_strs::<_, STR_BLOCK>(&mut rng));
                    }
                    let mut digests = [[0; 16]; BLOCK];
                    for (src, digest) in srcs
                        .chunks_exact(SIMD_DIGEST_BLOCK)
                        .zip(digests.chunks_exact_mut(SIMD_DIGEST_BLOCK))
                    {
                        digest.copy_from_slice(&x86::digest_md5s_simd(src.try_into().unwrap()));
                    }
                    let mut niceties = [Nicety::default(); BLOCK];
                    for (digest, nicety) in digests
                        .chunks_exact(COMPUTE_BLOCK)
                        .zip(niceties.chunks_exact_mut(COMPUTE_BLOCK))
                    {
                        nicety.copy_from_slice(&Simd::compute_nicety_batch(
                            TryInto::<[[u8; 16]; COMPUTE_BLOCK]>::try_into(digest).unwrap(),
                        ));
                    }

                    if report_send
                        .send(Report::new(&niceties, &srcs, &digests))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        });
        return;
    }

    thread::spawn({
        let report_send = report_send.clone();
        let mut rng = SmallRng::from_rng(trng).expect("failed to initialize RNG");

        move || {
            loop {
                let mut srcs = [[0; 32]; BLOCK];
                for chunk in srcs.chunks_exact_mut(STR_BLOCK) {
                    chunk.copy_from_slice(&baseline::generate_strs::<_, STR_BLOCK>(&mut rng));
                }
                let mut digests = [[0; 16]; BLOCK];
                for (src, digest) in srcs
                    .chunks_exact(DIGEST_BLOCK)
                    .zip(digests.chunks_exact_mut(DIGEST_BLOCK))
                {
                    digest.copy_from_slice(&baseline::digest_md5s::<DIGEST_BLOCK>(
                        src.try_into().unwrap(),
                    ));
                }
                let mut niceties = [Nicety::default(); BLOCK];
                for (digest, nicety) in digests
                    .chunks_exact(COMPUTE_BLOCK)
                    .zip(niceties.chunks_exact_mut(COMPUTE_BLOCK))
                {
                    nicety.copy_from_slice(&Baseline::compute_nicety_batch(
                        TryInto::<[[u8; 16]; COMPUTE_BLOCK]>::try_into(digest).unwrap(),
                    ));
                }

                if report_send
                    .send(Report::new(&niceties, &srcs, &digests))
                    .is_err()
                {
                    break;
                }
            }
        }
    });
}

fn main() {
    let cli = Cli::parse();

    // spawn command line renderer
    let (ui_send, ui_recv) = sync_channel::<(u64, Report)>(10);

    thread::spawn(move || {
        let style = ProgressStyle::with_template("  {prefix:>10.cyan}: {msg}").unwrap();

        let digits_bar = ProgressBar::new(u64::MAX)
            .with_style(style.clone())
            .with_prefix("digits")
            .with_message("awaiting update...");
        let letters_bar = ProgressBar::new(u64::MAX)
            .with_style(style.clone())
            .with_prefix("letters")
            .with_message("awaiting update...");
        let homogenous_bar = ProgressBar::new(u64::MAX)
            .with_style(style.clone())
            .with_prefix("homogenous")
            .with_message("awaiting update...");
        let e_bar = ProgressBar::new(u64::MAX)
            .with_style(style.clone())
            .with_prefix("e")
            .with_message("awaiting update...");
        let pi_bar = ProgressBar::new(u64::MAX)
            .with_style(style.clone())
            .with_prefix("π")
            .with_message("awaiting update...");

        let multi = MultiProgress::new();
        let digits_bar = multi.add(digits_bar);
        let letters_bar = multi.add(letters_bar);
        let homogenous_bar = multi.add(homogenous_bar);
        let e_bar = multi.add(e_bar);
        let pi_bar = multi.add(pi_bar);

        let summary = ProgressBar::new(u64::MAX).with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {per_sec:.green} processed {human_pos}",
            )
            .unwrap(),
        );
        let summary = multi.add(summary);
        multi
            .println("Current Best:")
            .expect("failed to write to console");

        while let Ok((total, report)) = ui_recv.recv() {
            summary.set_position(total);
            digits_bar.set_message(report.digits.to_string());
            letters_bar.set_message(report.letters.to_string());
            homogenous_bar.set_message(report.homogenous.to_string());
            e_bar.set_message(report.leading_e.to_string());
            pi_bar.set_message(report.leading_pi.to_string());
        }
    });
    // spawn workers

    // Allow staging of 4 message per thread
    let (report_send, report_recv) = sync_channel(4 * cli.worker);

    let mut trng = thread_rng();
    for _ in 0..cli.worker {
        new_thread(&report_send, &mut trng);
    }

    let mut master_report = Report::default();
    let mut processed = 0;

    let mut next_report = Instant::now() + RENDER_FREQUENCY;
    let mut last_reported_processed = 0;
    let mut est_interval = 0;
    while let Ok(report) = report_recv.recv() {
        processed += BLOCK as u64;
        master_report.digits = max(master_report.digits, report.digits);
        master_report.letters = max(master_report.letters, report.letters);
        master_report.homogenous = max(master_report.homogenous, report.homogenous);
        master_report.leading_e = max(master_report.leading_e, report.leading_e);
        master_report.leading_pi = max(master_report.leading_pi, report.leading_pi);

        if processed >= last_reported_processed + est_interval {
            let now = Instant::now();
            if now >= next_report {
                if ui_send.send((processed, master_report.clone())).is_err() {
                    break;
                }
                est_interval = est_interval / 3 + (processed - last_reported_processed) * 2 / 3;
                last_reported_processed = processed;
                next_report = now + RENDER_FREQUENCY;
            }
        }
    }
}
