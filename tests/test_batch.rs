#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{minimal_tokenizer, qwen_tokenizer};

// ---------------------------------------------------------------------------
// Minimal tokenizer tests (known IDs)
// ---------------------------------------------------------------------------

#[test]
fn test_encode_batch() {
    let tok = minimal_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world", "foo bar"];
    let batch = tok.encode_batch(&texts, false).unwrap();
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0], vec![39, 68, 105]);
    assert_eq!(batch[1], vec![86, 108]);
    assert_eq!(batch[2], vec![69, 78, 78, 94, 65, 64, 81]);
}

#[test]
fn test_encode_batch_matches_single() {
    let tok = minimal_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world"];
    let batch = tok.encode_batch(&texts, false).unwrap();
    for (text, batch_ids) in texts.iter().zip(batch.iter()) {
        let single_ids = tok.encode(text, false).unwrap();
        assert_eq!(batch_ids, &single_ids);
    }
}

#[test]
fn test_encode_batch_empty() {
    let tok = minimal_tokenizer();
    let result = tok.encode_batch(&[], false).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_decode_batch() {
    let tok = minimal_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world"];
    let batch_ids = tok.encode_batch(&texts, false).unwrap();
    let refs: Vec<&[i32]> = batch_ids.iter().map(|v| v.as_slice()).collect();
    let decoded = tok.decode_batch(&refs, false).unwrap();
    assert_eq!(decoded, texts);
}

#[test]
fn test_decode_batch_empty() {
    let tok = minimal_tokenizer();
    let result = tok.decode_batch(&[], false).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_encode_batch_flat() {
    let tok = minimal_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world"];
    let (flat, lengths) = tok.encode_batch_flat(&texts, false).unwrap();
    assert_eq!(lengths, vec![3, 2]);
    assert_eq!(flat, vec![39, 68, 105, 86, 108]);
}

// ---------------------------------------------------------------------------
// Qwen2.5 tokenizer tests (production-scale)
// ---------------------------------------------------------------------------

#[test]
fn test_qwen_batch_matches_single() {
    let tok = qwen_tokenizer();
    let texts: Vec<&str> = vec![
        common::SHORT_TEXT,
        "Hello world",
        "Tokenization is important.",
    ];
    let batch = tok.encode_batch(&texts, false).unwrap();
    assert_eq!(batch.len(), texts.len());
    for (text, batch_ids) in texts.iter().zip(batch.iter()) {
        let single_ids = tok.encode(text, false).unwrap();
        assert_eq!(batch_ids, &single_ids, "batch != single for: {text}");
    }
}

#[test]
fn test_qwen_batch_diverse_texts() {
    let tok = qwen_tokenizer();
    let texts: Vec<&str> = vec![
        common::SHORT_TEXT,
        common::MEDIUM_TEXT,
        common::MULTILINGUAL_TEXT,
        common::CODE_TEXT,
    ];
    let batch = tok.encode_batch(&texts, false).unwrap();
    assert_eq!(batch.len(), 4);
    // Each should produce tokens
    for (i, ids) in batch.iter().enumerate() {
        assert!(!ids.is_empty(), "batch[{i}] produced no tokens");
    }
}

#[test]
fn test_qwen_batch_roundtrip() {
    let tok = qwen_tokenizer();
    let texts: Vec<&str> = vec![
        "The quick brown fox jumps over the lazy dog.",
        common::SHORT_TEXT,
        common::MEDIUM_TEXT,
    ];
    let batch_ids = tok.encode_batch(&texts, false).unwrap();
    let refs: Vec<&[i32]> = batch_ids.iter().map(|v| v.as_slice()).collect();
    let decoded = tok.decode_batch(&refs, false).unwrap();
    assert_eq!(decoded, texts);
}

#[test]
fn test_qwen_batch_large() {
    let tok = qwen_tokenizer();
    let texts: Vec<&str> = (0..50).map(|_| common::SHORT_TEXT).collect();
    let batch = tok.encode_batch(&texts, false).unwrap();
    assert_eq!(batch.len(), 50);
    // All should be identical
    for ids in &batch {
        assert_eq!(ids, &batch[0]);
    }
}

#[test]
fn test_qwen_batch_flat() {
    let tok = qwen_tokenizer();
    let texts: Vec<&str> = vec!["Hello world", "Tokenizer test"];
    let (flat, lengths) = tok.encode_batch_flat(&texts, false).unwrap();
    assert_eq!(lengths.len(), 2);
    let total: usize = lengths
        .iter()
        .map(|l| *l as usize)
        .collect::<Vec<_>>()
        .iter()
        .sum();
    assert_eq!(flat.len(), total);
}

#[test]
fn test_qwen_batch_mixed_lengths() {
    let tok = qwen_tokenizer();
    let texts: Vec<&str> = vec![
        "Hi",                // very short
        common::SHORT_TEXT,  // short
        common::MEDIUM_TEXT, // medium
    ];
    let batch = tok.encode_batch(&texts, false).unwrap();
    // Token counts should increase with text length
    assert!(batch[0].len() < batch[1].len());
    assert!(batch[1].len() < batch[2].len());
}
