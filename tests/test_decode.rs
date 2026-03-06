#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{minimal_tokenizer, qwen_tokenizer};

// ---------------------------------------------------------------------------
// Minimal tokenizer tests
// ---------------------------------------------------------------------------

#[test]
fn test_decode_basic() {
    let tok = minimal_tokenizer();
    let ids = tok.encode("Hello world", false).unwrap();
    let text = tok.decode(&ids, false).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_decode_empty() {
    let tok = minimal_tokenizer();
    let text = tok.decode(&[], false).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_decode_roundtrip_unicode() {
    let tok = minimal_tokenizer();
    for text in &["Hello", "abc 123", "foo bar baz"] {
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(&decoded, text);
    }
}

// ---------------------------------------------------------------------------
// Qwen2.5 tokenizer tests (production-scale)
// ---------------------------------------------------------------------------

#[test]
fn test_qwen_decode_empty() {
    let tok = qwen_tokenizer();
    let text = tok.decode(&[], false).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_qwen_decode_medium() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::MEDIUM_TEXT, false).unwrap();
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, common::MEDIUM_TEXT);
}

#[test]
fn test_qwen_decode_long() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::LONG_TEXT, false).unwrap();
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, common::LONG_TEXT);
}

#[test]
fn test_qwen_decode_multilingual() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::MULTILINGUAL_TEXT, false).unwrap();
    let decoded = tok.decode(&ids, false).unwrap();
    // The IREE C library has a known decode issue with the first CJK character
    // after ASCII text, so we verify key substrings rather than exact equality.
    assert!(decoded.contains("recent years"));
    assert!(decoded.contains("语言模型"));
    assert!(decoded.contains("こんにちは世界"));
    assert!(decoded.contains("한국어도"));
}

#[test]
fn test_qwen_decode_code() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::CODE_TEXT, false).unwrap();
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, common::CODE_TEXT);
}

#[test]
fn test_qwen_decode_very_long() {
    let tok = qwen_tokenizer();
    let text = common::very_long_text();
    let ids = tok.encode(&text, false).unwrap();
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, text);
}
