//! Standalone benchmark comparison binary.
//!
//! Usage:
//!   cargo run --release --bin bench-compare --features bench -- [--iterations 50] [--warmup 5] [--validate]

use std::path::PathBuf;
use std::time::Instant;

const SHORT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn medium_text() -> String {
    SHORT_TEXT.repeat(20)
}

fn long_text() -> String {
    SHORT_TEXT.repeat(500)
}

struct BenchResult {
    p50_us: f64,
    p99_us: f64,
    mean_us: f64,
    n_tokens: usize,
}

fn bench_fn<F: FnMut() -> usize>(mut f: F, warmup: usize, iterations: usize) -> BenchResult {
    // Warmup
    let mut n_tokens = 0;
    for _ in 0..warmup {
        n_tokens = f();
    }

    // Timed runs
    let mut times_us = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        n_tokens = f();
        times_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_us = times_us.iter().sum::<f64>() / times_us.len() as f64;
    let p50_us = times_us[times_us.len() / 2];
    let p99_us = times_us[(times_us.len() as f64 * 0.99) as usize];

    BenchResult {
        p50_us,
        p99_us,
        mean_us,
        n_tokens,
    }
}

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata")
}

fn main() {
    let mut iterations: usize = 50;
    let mut warmup: usize = 5;
    let mut validate = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                i += 1;
                iterations = args[i].parse().unwrap_or(50);
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().unwrap_or(5);
            }
            "--validate" => validate = true,
            _ => {}
        }
        i += 1;
    }

    println!("Benchmark configuration: iterations={iterations}, warmup={warmup}");
    println!();

    // Load IREE tokenizer
    let iree_tok = iree_tokenizer::Tokenizer::from_file(
        testdata().join("bpe_bytelevel_minimal.json"),
    );
    let iree_tok = match iree_tok {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("Warning: IREE tokenizer failed to load: {e}");
            None
        }
    };

    // Load HuggingFace tokenizer (optional)
    #[cfg(feature = "bench")]
    let hf_tok = {
        match tokenizers::Tokenizer::from_pretrained("openai-community/gpt2", None) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("Warning: HuggingFace tokenizer failed to load: {e}");
                None
            }
        }
    };
    #[cfg(not(feature = "bench"))]
    let hf_tok: Option<()> = None;

    // Load tiktoken (optional)
    #[cfg(feature = "bench")]
    let tt_bpe = {
        match tiktoken_rs::get_bpe_from_model("gpt-3.5-turbo") {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("Warning: tiktoken failed to load: {e}");
                None
            }
        }
    };
    #[cfg(not(feature = "bench"))]
    let tt_bpe: Option<()> = None;

    let corpora = [
        ("short", SHORT_TEXT.to_string()),
        ("medium", medium_text()),
        ("long", long_text()),
    ];

    // --- Encode Latency ---
    println!("=== Encode Latency ===");
    println!(
        "{:<14} {:<8} {:>8} {:>8} {:>10} {:>10} {:>14}",
        "Backend", "Corpus", "Chars", "Tokens", "p50 (us)", "p99 (us)", "Tokens/sec"
    );
    println!("{}", "-".repeat(78));

    for (name, text) in &corpora {
        // IREE
        if let Some(tok) = &iree_tok {
            let text_ref = text.as_str();
            let result = bench_fn(
                || tok.encode(text_ref, false).unwrap().len(),
                warmup,
                iterations,
            );
            let tps = if result.mean_us > 0.0 {
                result.n_tokens as f64 / (result.mean_us / 1_000_000.0)
            } else {
                0.0
            };
            println!(
                "{:<14} {:<8} {:>8} {:>8} {:>10.1} {:>10.1} {:>14.0}",
                "iree", name, text.len(), result.n_tokens, result.p50_us, result.p99_us, tps
            );
        }

        // HuggingFace
        #[cfg(feature = "bench")]
        if let Some(tok) = &hf_tok {
            let text_ref = text.as_str();
            let result = bench_fn(
                || {
                    tok.encode(text_ref, false)
                        .map(|e| e.get_ids().len())
                        .unwrap_or(0)
                },
                warmup,
                iterations,
            );
            let tps = if result.mean_us > 0.0 {
                result.n_tokens as f64 / (result.mean_us / 1_000_000.0)
            } else {
                0.0
            };
            println!(
                "{:<14} {:<8} {:>8} {:>8} {:>10.1} {:>10.1} {:>14.0}",
                "huggingface", name, text.len(), result.n_tokens, result.p50_us, result.p99_us, tps
            );
        }

        // tiktoken
        #[cfg(feature = "bench")]
        if let Some(bpe) = &tt_bpe {
            let text_ref = text.as_str();
            let result = bench_fn(
                || bpe.encode_with_special_tokens(text_ref).len(),
                warmup,
                iterations,
            );
            let tps = if result.mean_us > 0.0 {
                result.n_tokens as f64 / (result.mean_us / 1_000_000.0)
            } else {
                0.0
            };
            println!(
                "{:<14} {:<8} {:>8} {:>8} {:>10.1} {:>10.1} {:>14.0}",
                "tiktoken", name, text.len(), result.n_tokens, result.p50_us, result.p99_us, tps
            );
        }
    }

    // --- Decode Latency ---
    println!();
    println!("=== Decode Latency ===");
    println!(
        "{:<14} {:<8} {:>8} {:>10} {:>10}",
        "Backend", "Corpus", "Tokens", "p50 (us)", "p99 (us)"
    );
    println!("{}", "-".repeat(56));

    for (name, text) in &corpora {
        if let Some(tok) = &iree_tok {
            let ids = tok.encode(text, false).unwrap();
            let ids_ref = ids.as_slice();
            let result =
                bench_fn(|| tok.decode(ids_ref, false).unwrap().len(), warmup, iterations);
            println!(
                "{:<14} {:<8} {:>8} {:>10.1} {:>10.1}",
                "iree", name, ids.len(), result.p50_us, result.p99_us
            );
        }

        #[cfg(feature = "bench")]
        if let Some(tok) = &hf_tok {
            let ids = tok
                .encode(text.as_str(), false)
                .unwrap()
                .get_ids()
                .to_vec();
            let result = bench_fn(
                || tok.decode(&ids, true).map(|s| s.len()).unwrap_or(0),
                warmup,
                iterations,
            );
            println!(
                "{:<14} {:<8} {:>8} {:>10.1} {:>10.1}",
                "huggingface", name, ids.len(), result.p50_us, result.p99_us
            );
        }

        #[cfg(feature = "bench")]
        if let Some(bpe) = &tt_bpe {
            let ids = bpe.encode_with_special_tokens(text);
            let ids_ref: Vec<usize> = ids.clone();
            let result = bench_fn(
                || bpe.decode(ids_ref.clone()).map(|s| s.len()).unwrap_or(0),
                warmup,
                iterations,
            );
            println!(
                "{:<14} {:<8} {:>8} {:>10.1} {:>10.1}",
                "tiktoken", name, ids.len(), result.p50_us, result.p99_us
            );
        }
    }

    // --- Batch Encode ---
    println!();
    println!("=== Batch Encode ===");
    println!(
        "{:<14} {:>8} {:>10} {:>10} {:>10} {:>14}",
        "Backend", "Batch", "Tokens", "p50 (us)", "p99 (us)", "Tokens/sec"
    );
    println!("{}", "-".repeat(72));

    for batch_size in [1, 10, 100] {
        let texts: Vec<&str> = (0..batch_size).map(|_| SHORT_TEXT).collect();

        if let Some(tok) = &iree_tok {
            let texts_ref = &texts;
            let result = bench_fn(
                || {
                    tok.encode_batch(texts_ref, false)
                        .unwrap()
                        .iter()
                        .map(|v| v.len())
                        .sum()
                },
                warmup,
                iterations,
            );
            let tps = if result.mean_us > 0.0 {
                result.n_tokens as f64 / (result.mean_us / 1_000_000.0)
            } else {
                0.0
            };
            println!(
                "{:<14} {:>8} {:>10} {:>10.1} {:>10.1} {:>14.0}",
                "iree", batch_size, result.n_tokens, result.p50_us, result.p99_us, tps
            );
        }

        #[cfg(feature = "bench")]
        if let Some(tok) = &hf_tok {
            let texts_ref = &texts;
            let result = bench_fn(
                || {
                    texts_ref
                        .iter()
                        .map(|t| tok.encode(*t, false).map(|e| e.get_ids().len()).unwrap_or(0))
                        .sum()
                },
                warmup,
                iterations,
            );
            let tps = if result.mean_us > 0.0 {
                result.n_tokens as f64 / (result.mean_us / 1_000_000.0)
            } else {
                0.0
            };
            println!(
                "{:<14} {:>8} {:>10} {:>10.1} {:>10.1} {:>14.0}",
                "huggingface", batch_size, result.n_tokens, result.p50_us, result.p99_us, tps
            );
        }

        #[cfg(feature = "bench")]
        if let Some(bpe) = &tt_bpe {
            let texts_ref = &texts;
            let result = bench_fn(
                || {
                    texts_ref
                        .iter()
                        .map(|t| bpe.encode_with_special_tokens(t).len())
                        .sum()
                },
                warmup,
                iterations,
            );
            let tps = if result.mean_us > 0.0 {
                result.n_tokens as f64 / (result.mean_us / 1_000_000.0)
            } else {
                0.0
            };
            println!(
                "{:<14} {:>8} {:>10} {:>10.1} {:>10.1} {:>14.0}",
                "tiktoken", batch_size, result.n_tokens, result.p50_us, result.p99_us, tps
            );
        }
    }

    // --- Validation ---
    if validate {
        println!();
        println!("=== Validation ===");

        if let Some(tok) = &iree_tok {
            // Roundtrip validation
            for (name, text) in &corpora {
                let ids = tok.encode(text, false).unwrap();
                let decoded = tok.decode(&ids, false).unwrap();
                if decoded == *text {
                    println!("  PASS: IREE encode→decode roundtrip ({name})");
                } else {
                    println!(
                        "  FAIL: IREE roundtrip mismatch ({name}): expected {} bytes, got {} bytes",
                        text.len(),
                        decoded.len()
                    );
                }
            }

            // Batch consistency
            let texts: Vec<&str> = corpora.iter().map(|(_, t)| t.as_str()).collect();
            let batch = tok.encode_batch(&texts, false).unwrap();
            let mut batch_ok = true;
            for (i, (_, text)) in corpora.iter().enumerate() {
                let single = tok.encode(text, false).unwrap();
                if batch[i] != single {
                    println!("  FAIL: Batch[{i}] differs from single encode");
                    batch_ok = false;
                }
            }
            if batch_ok {
                println!("  PASS: IREE batch encode matches single encode");
            }
        }
    }

    // Suppress unused variable warnings when bench feature is off.
    let _ = &hf_tok;
    let _ = &tt_bpe;
}
