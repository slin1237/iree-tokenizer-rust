#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::semicolon_if_nothing_returned,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::{fs, path::PathBuf};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ---------------------------------------------------------------------------
// Tokenizer download + cache (duplicated from tests/common for bench binary)
// ---------------------------------------------------------------------------

const QWEN_TOKENIZER_URL: &str =
    "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer.json";

fn cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/.cache")
}

fn qwen_tokenizer_path() -> PathBuf {
    let dir = cache_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qwen2.5_tokenizer.json");
    if path.exists() {
        return path;
    }
    println!("Downloading Qwen2.5 tokenizer...");
    let resp = ureq::get(QWEN_TOKENIZER_URL)
        .call()
        .expect("failed to download tokenizer");
    let body = resp
        .into_body()
        .read_to_vec()
        .expect("failed to read response body");
    fs::write(&path, &body).unwrap();
    println!("Cached at {}", path.display());
    path
}

// ---------------------------------------------------------------------------
// Corpus — diverse, realistic text at multiple scales
// ---------------------------------------------------------------------------

const SHORT_TEXT: &str =
    "The Rust programming language helps you write faster, more reliable software.";

const MEDIUM_TEXT: &str = "\
Transformers have revolutionized natural language processing since the publication \
of \"Attention Is All You Need\" in 2017. The key innovation is the self-attention \
mechanism, which allows the model to weigh the importance of different parts of the \
input sequence when producing each output token. Unlike RNNs, transformers process \
all positions in parallel, leading to significant speedups during training. Modern \
LLMs like GPT-4, Claude, and Qwen2.5 are all based on this architecture, typically \
using decoder-only designs with billions of parameters. Fine-tuning techniques such \
as LoRA (Low-Rank Adaptation) and RLHF (Reinforcement Learning from Human Feedback) \
have made these models increasingly capable at following instructions and generating \
high-quality text. The tokenizer is a critical component: it converts raw text into \
integer token IDs that the model can process, and its vocabulary size directly affects \
both model capacity and inference speed.";

const LONG_TEXT: &str = "\
Tokenization is the process of converting raw text into a sequence of integer token \
IDs that a language model can process. It is one of the most fundamental yet often \
overlooked components of the modern NLP pipeline. The choice of tokenizer affects \
model quality, inference latency, and the effective context window available to users. \
BPE is the most widely used tokenization algorithm in modern LLMs. Originally proposed \
for data compression by Philip Gage in 1994, it was adapted for NLP by Sennrich et al. \
in 2016. The algorithm works by iteratively merging the most frequent pair of adjacent \
tokens in a training corpus until the desired vocabulary size is reached. For example, \
starting with individual characters, BPE might first merge 't' and 'h' into 'th', then \
merge 'th' and 'e' into 'the'. This creates a vocabulary that efficiently represents \
common subwords while still being able to encode any text by falling back to individual \
bytes. The vocabulary size is a critical hyperparameter. Smaller vocabularies produce \
longer sequences but generalize better to rare words. Larger vocabularies produce shorter \
sequences and can represent more words as single tokens, but require more model parameters \
in the embedding layer. Modern tokenizers like those used in GPT-4 and Qwen2.5 use \
byte-level BPE, meaning they can represent any byte sequence. This eliminates unknown \
tokens entirely. Tokenizer performance matters more than many practitioners realize. In a \
typical LLM inference pipeline, tokenization happens on the critical path before any GPU \
computation begins. A slow tokenizer can add significant latency, especially for streaming \
applications where the first token latency is critical. The IREE tokenizer addresses this \
by implementing the entire tokenization pipeline in optimized C code with zero allocations \
per token. Streaming tokenization allows you to feed text incrementally and receive tokens \
as they become available. This is useful for real-time transcription, interactive editors, \
pipeline processing, and memory efficiency. The IREE tokenizer provides first-class \
streaming support through its EncodeStream and DecodeStream APIs. Special tokens are used \
for model control: BOS, EOS, PAD, UNK, SEP, CLS, and MASK. These tokens have dedicated \
IDs in the vocabulary and can be optionally added during encoding.";

fn very_long_text() -> String {
    let mut text = String::with_capacity(55000);
    for i in 0..10 {
        if i > 0 {
            text.push_str("\n\n---\n\n");
        }
        text.push_str(&format!("## Section {}\n\n", i + 1));
        text.push_str(LONG_TEXT);
    }
    text
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    let tokenizer_path = qwen_tokenizer_path();

    let very_long = very_long_text();
    let corpora: &[(&str, &str)] = &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
        ("very_long", &very_long),
    ];

    // IREE
    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(&tokenizer_path) {
        for &(name, text) in corpora {
            group.bench_with_input(BenchmarkId::new("iree", name), text, |b, text| {
                b.iter(|| tok.encode(black_box(text), false).unwrap())
            });
        }
    }

    // HuggingFace tokenizers
    if let Ok(tok) = tokenizers::Tokenizer::from_file(&tokenizer_path) {
        for &(name, text) in corpora {
            group.bench_with_input(BenchmarkId::new("huggingface", name), text, |b, text| {
                b.iter(|| tok.encode(black_box(text), false).unwrap())
            });
        }
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    let tokenizer_path = qwen_tokenizer_path();

    let very_long = very_long_text();
    let corpora: &[(&str, &str)] = &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
        ("very_long", &very_long),
    ];

    // IREE
    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(&tokenizer_path) {
        for &(name, text) in corpora {
            let ids = tok.encode(text, false).unwrap();
            group.bench_with_input(BenchmarkId::new("iree", name), &ids, |b, ids| {
                b.iter(|| tok.decode(black_box(ids), false).unwrap())
            });
        }
    }

    // HuggingFace tokenizers
    if let Ok(tok) = tokenizers::Tokenizer::from_file(&tokenizer_path) {
        for &(name, text) in corpora {
            let ids: Vec<u32> = tok.encode(text, false).unwrap().get_ids().to_vec();
            group.bench_with_input(BenchmarkId::new("huggingface", name), &ids, |b, ids| {
                b.iter(|| tok.decode(black_box(ids), true).unwrap())
            });
        }
    }

    group.finish();
}

fn bench_batch_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_encode");
    let tokenizer_path = qwen_tokenizer_path();

    // IREE (native batch)
    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(&tokenizer_path) {
        for batch_size in [1, 10, 100] {
            let texts: Vec<&str> = (0..batch_size).map(|_| MEDIUM_TEXT).collect();
            group.bench_with_input(
                BenchmarkId::new("iree", format!("batch_{batch_size}")),
                &texts,
                |b, texts| b.iter(|| tok.encode_batch(black_box(texts), false).unwrap()),
            );
        }
    }

    // HuggingFace (sequential, no native batch in this API)
    if let Ok(tok) = tokenizers::Tokenizer::from_file(&tokenizer_path) {
        for batch_size in [1, 10, 100] {
            let texts: Vec<&str> = (0..batch_size).map(|_| MEDIUM_TEXT).collect();
            group.bench_with_input(
                BenchmarkId::new("huggingface", format!("batch_{batch_size}")),
                &texts,
                |b, texts| {
                    b.iter(|| {
                        texts
                            .iter()
                            .map(|t| tok.encode(black_box(*t), false).unwrap())
                            .collect::<Vec<_>>()
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_streaming_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_encode");
    let tokenizer_path = qwen_tokenizer_path();

    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(&tokenizer_path) {
        // Compare one-shot vs streaming at different chunk sizes
        let text = LONG_TEXT;

        group.bench_function("iree/oneshot", |b| {
            b.iter(|| tok.encode(black_box(text), false).unwrap())
        });

        for chunk_size in [64, 256, 1024] {
            group.bench_function(
                BenchmarkId::new("iree/stream", format!("chunk_{chunk_size}")),
                |b| {
                    b.iter(|| {
                        let mut stream = tok.encode_stream(false).unwrap();
                        let mut all_ids = Vec::new();
                        for chunk in text.as_bytes().chunks(chunk_size) {
                            let s = std::str::from_utf8(chunk).unwrap();
                            all_ids.extend(stream.feed(black_box(s)).unwrap());
                        }
                        all_ids.extend(stream.finalize().unwrap());
                        all_ids
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_batch_encode,
    bench_streaming_encode
);
criterion_main!(benches);
