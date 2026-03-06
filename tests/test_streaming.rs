#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use iree_tokenizer::{decode_stream_iter, Tokenizer};

fn bpe_tokenizer() -> Tokenizer {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/bpe_bytelevel_minimal.json");
    Tokenizer::from_file(path).unwrap()
}

#[test]
fn test_encode_stream_basic() {
    let tok = bpe_tokenizer();
    let mut stream = tok.encode_stream(false).unwrap();
    assert!(stream.is_open());
    let mut all_ids = Vec::new();
    all_ids.extend(stream.feed("Hello ").unwrap());
    all_ids.extend(stream.feed("world").unwrap());
    all_ids.extend(stream.finalize().unwrap());
    // Should produce same tokens as one-shot.
    let expected = tok.encode("Hello world", false).unwrap();
    assert_eq!(all_ids, expected);
}

#[test]
fn test_decode_stream_basic() {
    let tok = bpe_tokenizer();
    let ids = tok.encode("Hello world", false).unwrap();
    let mut stream = tok.decode_stream(false).unwrap();
    assert!(stream.is_open());
    let mut text = String::new();
    // Feed one token at a time.
    for &token_id in &ids {
        text.push_str(&stream.feed(&[token_id]).unwrap());
    }
    text.push_str(&stream.finalize().unwrap());
    assert_eq!(text, "Hello world");
}

#[test]
fn test_encode_stream_close() {
    let tok = bpe_tokenizer();
    let mut stream = tok.encode_stream(false).unwrap();
    assert!(stream.is_open());
    // Finalize closes the stream.
    let _ = stream.finalize().unwrap();
    assert!(!stream.is_open());
    let result = stream.feed("text");
    assert!(result.is_err());
}

#[test]
fn test_decode_stream_close() {
    let tok = bpe_tokenizer();
    let mut stream = tok.decode_stream(false).unwrap();
    assert!(stream.is_open());
    let _ = stream.finalize().unwrap();
    assert!(!stream.is_open());
    let result = stream.feed(&[1]);
    assert!(result.is_err());
}

#[test]
fn test_decode_stream_iter_basic() {
    let tok = bpe_tokenizer();
    let ids = tok.encode("Hello world", false).unwrap();
    let chunks: Vec<String> = decode_stream_iter(&tok, ids, false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let full_text: String = chunks.join("");
    assert_eq!(full_text, "Hello world");
}

#[test]
fn test_decode_stream_iter_empty() {
    let tok = bpe_tokenizer();
    let chunks: Vec<String> = decode_stream_iter(&tok, vec![], false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn test_double_drop_safe() {
    // Dropping an encode stream that was never finalized should be safe.
    let tok = bpe_tokenizer();
    {
        let mut stream = tok.encode_stream(false).unwrap();
        let _ = stream.feed("Hello").unwrap();
        // Drop without finalize - should not crash.
    }
}
