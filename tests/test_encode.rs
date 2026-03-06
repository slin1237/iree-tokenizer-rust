#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use iree_tokenizer::Tokenizer;

fn bpe_tokenizer() -> Tokenizer {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/bpe_bytelevel_minimal.json");
    Tokenizer::from_file(path).unwrap()
}

#[test]
fn test_encode_basic() {
    let tok = bpe_tokenizer();
    let ids = tok.encode("Hello world", false).unwrap();
    assert_eq!(ids, vec![39, 68, 105, 110]);
}

#[test]
fn test_encode_empty() {
    let tok = bpe_tokenizer();
    let ids = tok.encode("", false).unwrap();
    assert!(ids.is_empty());
}

#[test]
fn test_encode_roundtrip() {
    let tok = bpe_tokenizer();
    let text = "Hello world";
    let ids = tok.encode(text, false).unwrap();
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_encode_rich_with_offsets() {
    let tok = bpe_tokenizer();
    let enc = tok.encode_rich("Hello world", false, true).unwrap();
    assert_eq!(enc.ids, vec![39, 68, 105, 110]);
    assert!(enc.offsets.is_some());
    let offsets = enc.offsets.unwrap();
    assert_eq!(offsets.len(), enc.ids.len());
    assert_eq!(offsets, vec![(0, 1), (1, 2), (2, 5), (5, 11)]);
    assert_eq!(enc.type_ids, vec![0, 0, 0, 0]);
}

#[test]
fn test_encode_rich_without_offsets() {
    let tok = bpe_tokenizer();
    let enc = tok.encode_rich("Hello world", false, false).unwrap();
    assert!(!enc.is_empty());
    assert!(enc.offsets.is_none());
}

#[test]
fn test_encode_rich_display() {
    let tok = bpe_tokenizer();
    let enc = tok.encode_rich("Hello", false, false).unwrap();
    let s = enc.to_string();
    assert!(s.contains("n_tokens="), "got: {s}");
}
