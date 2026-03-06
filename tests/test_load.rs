use iree_tokenizer::Tokenizer;
use std::path::PathBuf;

fn testdata() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata")
}

fn bpe_path() -> PathBuf {
    testdata().join("bpe_bytelevel_minimal.json")
}

#[test]
fn test_from_file() {
    let tok = Tokenizer::from_file(bpe_path()).unwrap();
    assert_eq!(tok.vocab_size(), 112);
}

#[test]
fn test_from_str() {
    let json_str = std::fs::read_to_string(bpe_path()).unwrap();
    let tok = Tokenizer::from_str(&json_str).unwrap();
    assert_eq!(tok.vocab_size(), 112);
}

#[test]
fn test_from_bytes() {
    let data = std::fs::read(bpe_path()).unwrap();
    let tok = Tokenizer::from_bytes(&data).unwrap();
    assert_eq!(tok.vocab_size(), 112);
}

#[test]
fn test_from_file_not_found() {
    let result = Tokenizer::from_file("/nonexistent/path.json");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Cannot open"), "got: {err}");
}

#[test]
fn test_from_str_invalid_json() {
    let result = Tokenizer::from_str("not valid json");
    assert!(result.is_err());
}

#[test]
fn test_display() {
    let tok = Tokenizer::from_file(bpe_path()).unwrap();
    let s = tok.to_string();
    assert!(s.contains("BPE"), "got: {s}");
    assert!(s.contains("vocab_size"), "got: {s}");
}
