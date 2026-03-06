#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use iree_tokenizer::Tokenizer;

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata")
}

fn tiktoken_path() -> PathBuf {
    testdata().join("tiktoken_gpt2.tiktoken")
}

fn tiktoken_tokenizer() -> Tokenizer {
    Tokenizer::from_tiktoken_file(tiktoken_path(), "gpt2").unwrap()
}

#[test]
fn test_from_tiktoken() {
    let tok = Tokenizer::from_tiktoken_file(tiktoken_path(), "gpt2").unwrap();
    assert_eq!(tok.vocab_size(), 261);
}

#[test]
fn test_from_tiktoken_str() {
    let data = std::fs::read_to_string(tiktoken_path()).unwrap();
    let tok = Tokenizer::from_tiktoken_str(&data, "gpt2").unwrap();
    assert_eq!(tok.vocab_size(), 261);
}

#[test]
fn test_from_tiktoken_bytes() {
    let data = std::fs::read(tiktoken_path()).unwrap();
    let tok = Tokenizer::from_tiktoken_bytes(&data, "gpt2").unwrap();
    assert_eq!(tok.vocab_size(), 261);
}

#[test]
fn test_tiktoken_invalid_encoding() {
    let result = Tokenizer::from_tiktoken_file(tiktoken_path(), "nonexistent");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("Unknown tiktoken encoding"),
        "got: {err}"
    );
}

#[test]
fn test_tiktoken_model_type() {
    let tok = tiktoken_tokenizer();
    assert_eq!(tok.model_type(), "BPE");
}

#[test]
fn test_tiktoken_encode_decode_roundtrip() {
    let tok = tiktoken_tokenizer();
    let text = "Hello world";
    let ids = tok.encode(text, false).unwrap();
    assert_eq!(
        ids,
        vec![72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
    );
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_tiktoken_encode_empty() {
    let tok = tiktoken_tokenizer();
    let ids = tok.encode("", false).unwrap();
    assert!(ids.is_empty());
}

#[test]
fn test_tiktoken_encode_batch() {
    let tok = tiktoken_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world"];
    let batch = tok.encode_batch(&texts, false).unwrap();
    assert_eq!(batch[0], vec![72, 101, 108, 108, 111]);
    assert_eq!(batch[1], vec![119, 111, 114, 108, 100]);
}
