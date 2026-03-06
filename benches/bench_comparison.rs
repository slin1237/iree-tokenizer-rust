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

    // IREE
    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(
        testdata().join("bpe_bytelevel_minimal.json"),
    ) {
        let medium = medium_text();
        let long = long_text();
        group.bench_with_input(BenchmarkId::new("iree", "short"), SHORT_TEXT, |b, text| {
            b.iter(|| tok.encode(black_box(text), false).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("iree", "medium"), &medium, |b, text| {
            b.iter(|| tok.encode(black_box(text.as_str()), false).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("iree", "long"), &long, |b, text| {
            b.iter(|| tok.encode(black_box(text.as_str()), false).unwrap())
        });
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(
        testdata().join("bpe_bytelevel_minimal.json"),
    ) {
        let medium = medium_text();
        let long = long_text();
        let ids_short = tok.encode(SHORT_TEXT, false).unwrap();
        let ids_medium = tok.encode(&medium, false).unwrap();
        let ids_long = tok.encode(&long, false).unwrap();

        group.bench_with_input(
            BenchmarkId::new("iree", "short"),
            &ids_short,
            |b, ids| b.iter(|| tok.decode(black_box(ids), false).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("iree", "medium"),
            &ids_medium,
            |b, ids| b.iter(|| tok.decode(black_box(ids), false).unwrap()),
        );
        group.bench_with_input(BenchmarkId::new("iree", "long"), &ids_long, |b, ids| {
            b.iter(|| tok.decode(black_box(ids), false).unwrap())
        });
    }

    group.finish();
}

fn bench_batch_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_encode");

    if let Ok(tok) = iree_tokenizer::Tokenizer::from_file(
        testdata().join("bpe_bytelevel_minimal.json"),
    ) {
        for batch_size in [1, 10, 100] {
            let texts: Vec<&str> = (0..batch_size).map(|_| SHORT_TEXT).collect();
            group.bench_with_input(
                BenchmarkId::new("iree", format!("batch_{batch_size}")),
                &texts,
                |b, texts| b.iter(|| tok.encode_batch(black_box(texts), false).unwrap()),
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_batch_encode);
criterion_main!(benches);
