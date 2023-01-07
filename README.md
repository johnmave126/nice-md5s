# Nice MD5s
This is an attempt to write performant code to find [nice MD5s](https://github.com/zvibazak/Nice-MD5s) to test out Rust's SIMD and inline assembly stories.

## What does that even mean?
See [zvibazak/Nice-MD5s](https://github.com/zvibazak/Nice-MD5s) for an explanation.

## Running the binary
```
A terminal app to find nice MD5s

Usage: nice-md5s.exe [OPTIONS]

Options:
  -w, --worker <NUM>  Number of threads to use [default: <your core count>]
  -h, --help          Print help information
  -V, --version       Print version information
```
The exectuble will try to use SIMD algorithms if your CPU supports it. It is recommended to compile the binary with `RUSTFLAGS="-C target-cpu=native"`.

![screenshot of executable running](/assets/bin-screenshot.png)

## Structure
We consider 3 different tasks to find nice MD5s:
 * Generate random strings/byte arrays.
 * MD5 digest.
 * Compute different metrics of "how nice" a digest is.

`baseline.rs` contains code as a baseline for performance measurement:
 * `generate_strs()` generates strings by simply mapping random bytes to a random character in [0-9a-z]. The algorithm is not uniformly distributed though.
 * `digest_md5s()` produces MD5 digest using existing library [md-5](https://crates.io/crates/md-5).
 * `Baseline` provides a struct to compute different metrics. The computation naïvely iterates through the bytes.

Module `x86` contains code specifically for `x86` or `x86_64` optimizations.

`x86/generation.rs` contains code for string generation and MD5 digest.
 * `generate_strs()` uses the same algorithm as in `baseline.rs`, but in SIMD.
 * `digest_md5s()` uses inline-assembly to implement MD5, derived from [Fast MD5 hash implementation in x86 assembly](https://www.nayuki.io/page/fast-md5-hash-implementation-in-x86-assembly).
 * `digest_md5s_simd()` uses SIMD to simutaneously compute 8 MD5s using a single pass of MD5 algorithm.

`x86/simd.rs` contains code for computing metrics.
* `Simd` provides a struct to compute different metrics using SIMD. The main idea is to use AVX/AVX2/SSE2 intrinsics to check all the nibbles simutaneously and use `_mm256_movemask` and `_tzcnt` to collect the result.

## Performance
My system:

| | |
|---|---|
| CPU | AMD Ryzen 9 5900X |
| RAM | 64GB DDR4 3200MHz |
| OS | Windows 10 22H2 |

With 24 worker threads, about 0.5B/s strings are generated, digested, and with metrics computed.

We also have `criterion`-backed benchmark for each individual tasks.

## License
Dual licensed under the Apache 2.0 license and the MIT license.
