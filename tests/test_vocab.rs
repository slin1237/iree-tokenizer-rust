#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use iree_tokenizer::Tokenizer;

fn bpe_tokenizer() -> Tokenizer {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/bpe_bytelevel_minimal.json");
    Tokenizer::from_file(path).unwrap()
}

#[test]
fn test_vocab_size() {
    let tok = bpe_tokenizer();
    assert_eq!(tok.vocab_size(), 112);
}

#[test]
fn test_model_type() {
    let tok = bpe_tokenizer();
    assert_eq!(tok.model_type(), "BPE");
}

#[test]
fn test_token_to_id() {
    let tok = bpe_tokenizer();
    assert_eq!(tok.token_to_id("H"), Some(39));
}

#[test]
fn test_token_to_id_not_found() {
    let tok = bpe_tokenizer();
    assert_eq!(tok.token_to_id("\x00\x01\x02\x03nonexistent"), None);
}

#[test]
fn test_id_to_token() {
    let tok = bpe_tokenizer();
    assert_eq!(tok.id_to_token(0), Some("!"));
}

#[test]
fn test_id_to_token_out_of_range() {
    let tok = bpe_tokenizer();
    assert_eq!(tok.id_to_token(999999), None);
}

#[test]
fn test_id_to_token_negative() {
    let tok = bpe_tokenizer();
    assert_eq!(tok.id_to_token(-1), None);
}

#[test]
fn test_roundtrip_token_id() {
    let tok = bpe_tokenizer();
    let token = tok.id_to_token(0).unwrap();
    let id = tok.token_to_id(token).unwrap();
    assert_eq!(id, 0);
}
