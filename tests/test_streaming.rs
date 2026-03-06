#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{minimal_tokenizer, qwen_tokenizer};
use iree_tokenizer::decode_stream_iter;

// ---------------------------------------------------------------------------
// Minimal tokenizer tests (edge cases)
// ---------------------------------------------------------------------------

#[test]
fn test_encode_stream_basic() {
    let tok = minimal_tokenizer();
    let mut stream = tok.encode_stream(false).unwrap();
    assert!(stream.is_open());
    let mut all_ids = Vec::new();
    all_ids.extend(stream.feed("Hello ").unwrap());
    all_ids.extend(stream.feed("world").unwrap());
    all_ids.extend(stream.finalize().unwrap());
    let expected = tok.encode("Hello world", false).unwrap();
    assert_eq!(all_ids, expected);
}

#[test]
fn test_decode_stream_basic() {
    let tok = minimal_tokenizer();
    let ids = tok.encode("Hello world", false).unwrap();
    let mut stream = tok.decode_stream(false).unwrap();
    assert!(stream.is_open());
    let mut text = String::new();
    for &token_id in &ids {
        text.push_str(&stream.feed(&[token_id]).unwrap());
    }
    text.push_str(&stream.finalize().unwrap());
    assert_eq!(text, "Hello world");
}

#[test]
fn test_encode_stream_close() {
    let tok = minimal_tokenizer();
    let mut stream = tok.encode_stream(false).unwrap();
    assert!(stream.is_open());
    let _ = stream.finalize().unwrap();
    assert!(!stream.is_open());
    let result = stream.feed("text");
    assert!(result.is_err());
}

#[test]
fn test_decode_stream_close() {
    let tok = minimal_tokenizer();
    let mut stream = tok.decode_stream(false).unwrap();
    assert!(stream.is_open());
    let _ = stream.finalize().unwrap();
    assert!(!stream.is_open());
    let result = stream.feed(&[1]);
    assert!(result.is_err());
}

#[test]
fn test_decode_stream_iter_basic() {
    let tok = minimal_tokenizer();
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
    let tok = minimal_tokenizer();
    let chunks: Vec<String> = decode_stream_iter(&tok, vec![], false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn test_double_drop_safe() {
    let tok = minimal_tokenizer();
    {
        let mut stream = tok.encode_stream(false).unwrap();
        let _ = stream.feed("Hello").unwrap();
        // Drop without finalize - should not crash.
    }
}

// ---------------------------------------------------------------------------
// Qwen2.5 tokenizer streaming tests (production-scale)
// ---------------------------------------------------------------------------

#[test]
fn test_qwen_stream_matches_oneshot() {
    let tok = qwen_tokenizer();
    let text = common::MEDIUM_TEXT;

    // One-shot encode
    let expected = tok.encode(text, false).unwrap();

    // Streaming encode — feed in chunks of ~100 chars
    let mut stream = tok.encode_stream(false).unwrap();
    let mut all_ids = Vec::new();
    for chunk in text.as_bytes().chunks(100) {
        let chunk_str = std::str::from_utf8(chunk).unwrap();
        all_ids.extend(stream.feed(chunk_str).unwrap());
    }
    all_ids.extend(stream.finalize().unwrap());

    assert_eq!(all_ids, expected);
}

#[test]
fn test_qwen_stream_long_text() {
    let tok = qwen_tokenizer();
    let text = common::LONG_TEXT;

    let expected = tok.encode(text, false).unwrap();

    let mut stream = tok.encode_stream(false).unwrap();
    let mut all_ids = Vec::new();
    // Feed in larger chunks (~500 chars)
    for chunk in text.as_bytes().chunks(500) {
        let chunk_str = std::str::from_utf8(chunk).unwrap();
        all_ids.extend(stream.feed(chunk_str).unwrap());
    }
    all_ids.extend(stream.finalize().unwrap());

    assert_eq!(all_ids, expected);
}

#[test]
fn test_qwen_stream_multilingual() {
    let tok = qwen_tokenizer();
    let text = common::MULTILINGUAL_TEXT;

    let expected = tok.encode(text, false).unwrap();

    // Feed character-by-character (stress test for multibyte handling)
    let mut stream = tok.encode_stream(false).unwrap();
    let mut all_ids = Vec::new();
    // Feed in char-boundary-safe chunks
    let chars: Vec<char> = text.chars().collect();
    for chunk in chars.chunks(20) {
        let s: String = chunk.iter().collect();
        all_ids.extend(stream.feed(&s).unwrap());
    }
    all_ids.extend(stream.finalize().unwrap());

    assert_eq!(all_ids, expected);
}

#[test]
fn test_qwen_decode_stream_long() {
    let tok = qwen_tokenizer();
    let text = common::MEDIUM_TEXT;
    let ids = tok.encode(text, false).unwrap();

    let mut stream = tok.decode_stream(false).unwrap();
    let mut decoded = String::new();
    // Feed tokens in small batches
    for chunk in ids.chunks(10) {
        decoded.push_str(&stream.feed(chunk).unwrap());
    }
    decoded.push_str(&stream.finalize().unwrap());

    assert_eq!(decoded, text);
}

#[test]
fn test_qwen_decode_stream_iter_long() {
    let tok = qwen_tokenizer();
    let text = common::MEDIUM_TEXT;
    let ids = tok.encode(text, false).unwrap();

    let chunks: Vec<String> = decode_stream_iter(&tok, ids, false)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let full_text: String = chunks.join("");
    assert_eq!(full_text, text);
}

#[test]
fn test_qwen_stream_code() {
    let tok = qwen_tokenizer();
    let text = common::CODE_TEXT;

    let expected = tok.encode(text, false).unwrap();

    let mut stream = tok.encode_stream(false).unwrap();
    let mut all_ids = Vec::new();
    // Feed line-by-line
    for line in text.lines() {
        let mut line_with_newline = line.to_string();
        line_with_newline.push('\n');
        all_ids.extend(stream.feed(&line_with_newline).unwrap());
    }
    all_ids.extend(stream.finalize().unwrap());

    assert_eq!(all_ids, expected);
}
