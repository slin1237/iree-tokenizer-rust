#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{minimal_tokenizer, qwen_tokenizer};

// ---------------------------------------------------------------------------
// Minimal tokenizer tests (known values)
// ---------------------------------------------------------------------------

#[test]
fn test_vocab_size() {
    let tok = minimal_tokenizer();
    assert_eq!(tok.vocab_size(), 112);
}

#[test]
fn test_model_type() {
    let tok = minimal_tokenizer();
    assert_eq!(tok.model_type(), "BPE");
}

#[test]
fn test_token_to_id() {
    let tok = minimal_tokenizer();
    assert_eq!(tok.token_to_id("H"), Some(39));
}

#[test]
fn test_token_to_id_not_found() {
    let tok = minimal_tokenizer();
    assert_eq!(tok.token_to_id("\x00\x01\x02\x03nonexistent"), None);
}

#[test]
fn test_id_to_token() {
    let tok = minimal_tokenizer();
    assert_eq!(tok.id_to_token(0), Some("!"));
}

#[test]
fn test_id_to_token_out_of_range() {
    let tok = minimal_tokenizer();
    assert_eq!(tok.id_to_token(999_999), None);
}

#[test]
fn test_id_to_token_negative() {
    let tok = minimal_tokenizer();
    assert_eq!(tok.id_to_token(-1), None);
}

#[test]
fn test_roundtrip_token_id() {
    let tok = minimal_tokenizer();
    let token = tok.id_to_token(0).unwrap();
    let id = tok.token_to_id(token).unwrap();
    assert_eq!(id, 0);
}

// ---------------------------------------------------------------------------
// Qwen2.5 tokenizer tests (production-scale vocabulary)
// ---------------------------------------------------------------------------

#[test]
fn test_qwen_vocab_size() {
    let tok = qwen_tokenizer();
    // Qwen2.5 has 151K+ tokens
    assert!(
        tok.vocab_size() > 150_000,
        "expected > 150K, got {}",
        tok.vocab_size()
    );
}

#[test]
fn test_qwen_model_type() {
    let tok = qwen_tokenizer();
    assert_eq!(tok.model_type(), "BPE");
}

#[test]
fn test_qwen_special_tokens() {
    let tok = qwen_tokenizer();
    // Qwen2.5 uses <|endoftext|> as its EOS-like token
    let eos_id = tok.token_to_id("<|endoftext|>");
    assert!(eos_id.is_some(), "Qwen should have <|endoftext|> token");
}

#[test]
fn test_qwen_token_to_id_common_words() {
    let tok = qwen_tokenizer();
    // Common English words should be in the vocabulary
    for word in &["the", "is", "of", "and", "to"] {
        let id = tok.token_to_id(word);
        assert!(id.is_some(), "expected '{word}' in Qwen vocabulary");
    }
}

#[test]
fn test_qwen_id_to_token_roundtrip() {
    let tok = qwen_tokenizer();
    // Pick some IDs and verify roundtrip
    for id in [0, 1, 100, 1000, 10000] {
        if let Some(token) = tok.id_to_token(id) {
            let back_id = tok.token_to_id(token);
            assert_eq!(
                back_id,
                Some(id),
                "roundtrip failed for id={id}, token={token:?}"
            );
        }
    }
}

#[test]
fn test_qwen_id_to_token_out_of_range() {
    let tok = qwen_tokenizer();
    let token = tok.id_to_token(999_999_999);
    assert!(token.is_none());
}
