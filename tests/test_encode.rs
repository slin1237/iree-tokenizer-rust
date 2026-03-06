#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{minimal_tokenizer, qwen_tokenizer};

// ---------------------------------------------------------------------------
// Minimal tokenizer tests (edge cases, known IDs)
// ---------------------------------------------------------------------------

#[test]
fn test_encode_basic() {
    let tok = minimal_tokenizer();
    let ids = tok.encode("Hello world", false).unwrap();
    assert_eq!(ids, vec![39, 68, 105, 110]);
}

#[test]
fn test_encode_empty() {
    let tok = minimal_tokenizer();
    let ids = tok.encode("", false).unwrap();
    assert!(ids.is_empty());
}

#[test]
fn test_encode_roundtrip() {
    let tok = minimal_tokenizer();
    let text = "Hello world";
    let ids = tok.encode(text, false).unwrap();
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_encode_rich_with_offsets() {
    let tok = minimal_tokenizer();
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
    let tok = minimal_tokenizer();
    let enc = tok.encode_rich("Hello world", false, false).unwrap();
    assert!(!enc.is_empty());
    assert!(enc.offsets.is_none());
}

#[test]
fn test_encode_rich_display() {
    let tok = minimal_tokenizer();
    let enc = tok.encode_rich("Hello", false, false).unwrap();
    let s = enc.to_string();
    assert!(s.contains("n_tokens="), "got: {s}");
}

// ---------------------------------------------------------------------------
// Qwen2.5 tokenizer tests (production-scale, diverse text)
// ---------------------------------------------------------------------------

#[test]
fn test_qwen_encode_short() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::SHORT_TEXT, false).unwrap();
    assert!(!ids.is_empty());
    // Qwen should tokenize a short English sentence into a reasonable number of tokens
    assert!(ids.len() < 30, "expected < 30 tokens, got {}", ids.len());
}

#[test]
fn test_qwen_encode_medium() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::MEDIUM_TEXT, false).unwrap();
    assert!(ids.len() > 50, "expected > 50 tokens, got {}", ids.len());
}

#[test]
fn test_qwen_encode_long() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::LONG_TEXT, false).unwrap();
    assert!(ids.len() > 500, "expected > 500 tokens, got {}", ids.len());
}

#[test]
fn test_qwen_encode_very_long() {
    let tok = qwen_tokenizer();
    let text = common::very_long_text();
    let ids = tok.encode(&text, false).unwrap();
    assert!(
        ids.len() > 5000,
        "expected > 5000 tokens, got {}",
        ids.len()
    );
}

#[test]
fn test_qwen_encode_multilingual() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::MULTILINGUAL_TEXT, false).unwrap();
    assert!(!ids.is_empty());
    // Qwen handles Chinese, Japanese, and Korean well
    assert!(
        ids.len() > 50,
        "multilingual text should produce many tokens"
    );
}

#[test]
fn test_qwen_encode_code() {
    let tok = qwen_tokenizer();
    let ids = tok.encode(common::CODE_TEXT, false).unwrap();
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids, false).unwrap();
    assert_eq!(decoded, common::CODE_TEXT);
}

#[test]
fn test_qwen_roundtrip_all_corpora() {
    let tok = qwen_tokenizer();
    for (name, text) in [
        ("short", common::SHORT_TEXT),
        ("medium", common::MEDIUM_TEXT),
        ("long", common::LONG_TEXT),
        ("code", common::CODE_TEXT),
    ] {
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(decoded, text, "roundtrip failed for {name}");
    }
}

#[test]
fn test_qwen_encode_rich_long() {
    let tok = qwen_tokenizer();
    let enc = tok.encode_rich(common::MEDIUM_TEXT, false, true).unwrap();
    assert!(!enc.is_empty());
    let offsets = enc.offsets.unwrap();
    assert_eq!(offsets.len(), enc.ids.len());
    // Offsets should be monotonically non-decreasing in start position
    for window in offsets.windows(2) {
        assert!(
            window[1].0 >= window[0].0,
            "offsets not monotonic: {:?} -> {:?}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn test_qwen_encode_with_special_tokens() {
    let tok = qwen_tokenizer();
    let ids_without = tok.encode("Hello world", false).unwrap();
    let ids_with = tok.encode("Hello world", true).unwrap();
    // With special tokens should produce at least as many tokens
    assert!(ids_with.len() >= ids_without.len());
}
