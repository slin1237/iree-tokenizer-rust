#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use iree_tokenizer::Tokenizer;

fn bpe_tokenizer() -> Tokenizer {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/bpe_bytelevel_minimal.json");
    Tokenizer::from_file(path).unwrap()
}

#[test]
fn test_encode_batch() {
    let tok = bpe_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world", "foo bar"];
    let batch = tok.encode_batch(&texts, false).unwrap();
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0], vec![39, 68, 105]);
    assert_eq!(batch[1], vec![86, 108]);
    assert_eq!(batch[2], vec![69, 78, 78, 94, 65, 64, 81]);
}

#[test]
fn test_encode_batch_matches_single() {
    let tok = bpe_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world"];
    let batch = tok.encode_batch(&texts, false).unwrap();
    for (text, batch_ids) in texts.iter().zip(batch.iter()) {
        let single_ids = tok.encode(text, false).unwrap();
        assert_eq!(batch_ids, &single_ids);
    }
}

#[test]
fn test_encode_batch_empty() {
    let tok = bpe_tokenizer();
    let result = tok.encode_batch(&[], false).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_decode_batch() {
    let tok = bpe_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world"];
    let batch_ids = tok.encode_batch(&texts, false).unwrap();
    let refs: Vec<&[i32]> = batch_ids.iter().map(|v| v.as_slice()).collect();
    let decoded = tok.decode_batch(&refs, false).unwrap();
    assert_eq!(decoded, texts);
}

#[test]
fn test_decode_batch_empty() {
    let tok = bpe_tokenizer();
    let result = tok.decode_batch(&[], false).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_encode_batch_flat() {
    let tok = bpe_tokenizer();
    let texts: Vec<&str> = vec!["Hello", "world"];
    let (flat, lengths) = tok.encode_batch_flat(&texts, false).unwrap();
    assert_eq!(lengths, vec![3, 2]);
    assert_eq!(flat, vec![39, 68, 105, 86, 108]);
}
