use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;

const SHORT_TEXT: &str = "The quick brown fox jumps over the lazy dog.";

fn medium_text() -> String {
    SHORT_TEXT.repeat(20)
}

fn long_text() -> String {
    SHORT_TEXT.repeat(500)
}

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata")
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    let tokenizer_path = testdata().join("bpe_bytelevel_minimal.json");

    let medium = medium_text();
    let long = long_text();
    let corpora: &[(&str, &str)] = &[
        ("short", SHORT_TEXT),
        ("medium", &medium),
        ("long", &long),
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
    let tokenizer_path = testdata().join("bpe_bytelevel_minimal.json");

    let medium = medium_text();
    let long = long_text();
    let corpora: &[(&str, &str)] = &[
        ("short", SHORT_TEXT),
        ("medium", &medium),
        ("long", &long),
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
    let tokenizer_path = testdata().join("bpe_bytelevel_minimal.json");

    // IREE (native batch)
    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(&tokenizer_path) {
        for batch_size in [1, 10, 100] {
            let texts: Vec<&str> = (0..batch_size).map(|_| SHORT_TEXT).collect();
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
            let texts: Vec<&str> = (0..batch_size).map(|_| SHORT_TEXT).collect();
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

criterion_group!(benches, bench_encode, bench_decode, bench_batch_encode);
criterion_main!(benches);
